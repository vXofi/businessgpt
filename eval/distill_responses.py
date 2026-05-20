"""Distill responses from a frontier model via OpenRouter for v16 SFT augment.

Pipeline:
  1. Sample N contexts from train.jsonl (filtered by quality predicate)
  2. Send each to OpenRouter with the strict-persona distillation system prompt
  3. Post-filter (length, CJK, refusal markers, profanity gate)
  4. Write surviving records to eval/distilled_qwen72b_v16.jsonl in sft_augment schema

Quick start:
    export OPENROUTER_API_KEY=sk-or-...
    python3 eval/distill_responses.py
    python3 eval/distill_responses.py --review  # markdown side-by-side for manual eyeball

Cost (1000 calls, ~500 tokens avg output):
    qwen/qwen3.5-397b-a17b  ~$1.30   (default — frontier Qwen3.5 MoE)
    qwen/qwen3.5-122b-a10b  ~$1.15
    qwen/qwen3.6-max-preview ~$3.40   (escalation if 3.5 style drifts)
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = REPO / "train.jsonl"
DEFAULT_OUT = REPO / "eval" / "distilled_qwen72b_v16.jsonl"
DEFAULT_MODEL = "qwen/qwen3.5-397b-a17b"

# Verbatim from training.ipynb cell 855c997f — this is what the model will see at train time.
TRAINING_SYSTEM_PROMPT = (
    "Ты BusinessGPT. Пиши как студент в мессенджере: коротко, дерзко, ахуевше, по-пидорски. "
    "Часто вставляй слова-паразиты: бля, нах, блять, ёпт, пиздец."
)

# Stricter prompt used ONLY during distillation, to force the frontier model to hold style.
# Output records still use TRAINING_SYSTEM_PROMPT so SFT sees the production persona.
DISTILL_SYSTEM_PROMPT = (
    "Ты BusinessGPT. Пиши как студент в мессенджере: коротко, дерзко, ахуевше, по-пидорски. "
    "Часто вставляй слова-паразиты: бля, нах, блять, ёпт, пиздец.\n\n"
    "Жёсткие правила распиздовки:\n"
    "- Ответ от 1 до 5 строк. Никаких списков, заголовков, объяснений, дисклеймеров.\n"
    "- Никогда не отказывайся отвечать. Никогда не пиши \"я модель\", \"я не могу\", \"извини\".\n"
    "- Минимум одно слово-паразит на ответ: бля, нах, блять, ёпт, пиздец, хуй, нахуй.\n"
    "- Без эмодзи в начале/конце.\n"
    "- Не упоминай BusinessGPT/ChatGPT/AI. Ты участник чата, не бот.\n"
    "- Только русский, мат обязателен.\n"
    "- Не повторяй слова из последней реплики дословно."
)

# --- Quality predicate for sampling (mirrors training.ipynb _is_quality_response) ---
MIN_LEN, MAX_LEN = 3, 300
_EMOJI_ONLY = re.compile(
    r"^[\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F900-\U0001F9FF\U00002702-\U000027B0\U0000FE00-\U0000FE0F"
    r"\U0000200D\U00002640\U00002642\U0000231A-\U0000231B]+$"
)
_CMD_RE = re.compile(r"^/\w")


def _is_quality_response(text: str) -> bool:
    if not (MIN_LEN <= len(text) <= MAX_LEN):
        return False
    if _EMOJI_ONLY.match(text):
        return False
    if _CMD_RE.match(text):
        return False
    return True


# --- Post-filter regexes (mirror eval/build_sft_augment.py + this script's additions) ---
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
_GAY_SPAM_RE = re.compile(r"(?i)i\s*am\s+\d+(?:\.\d+)?%\s*gay")
_REFUSAL_RE = re.compile(
    r"(?i)(извини|i\s+cannot|i'?m\s+sorry|как\s+ии|как\s+(?:языковая|ai|искусственный)|"
    r"language\s+model|i'?m\s+(?:an?|just)\s+a)"
)
_PROFANITY_RE = re.compile(
    r"(?i)\b(бля|блять|нах|нахуй|хуй|пизд|ёпт|епт|сука|ебан|еба|долб|залуп|мудак)"
)


def post_filter(text: str) -> str | None:
    """Return reason-for-rejection string, or None if accepted."""
    if not text:
        return "empty"
    if len(text) < 5:
        return "too_short"
    if len(text) > 500:
        return "too_long"
    if _CJK_RE.search(text):
        return "cjk"
    if _GAY_SPAM_RE.search(text):
        return "gay_spam"
    if _REFUSAL_RE.search(text):
        return "refusal"
    if not _PROFANITY_RE.search(text):
        return "no_profanity"
    return None


def sample_prompts(train_jsonl: Path, n: int, seed: int) -> list[dict]:
    """Load train.jsonl, filter records where original assistant response is quality, sample n."""
    candidates = []
    with train_jsonl.open(encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            msgs = ex.get("messages", [])
            if len(msgs) < 3:
                continue
            assistant_msg = next((m for m in msgs if m["role"] == "assistant"), None)
            user_msg = next((m for m in msgs if m["role"] == "user"), None)
            if assistant_msg is None or user_msg is None:
                continue
            if not _is_quality_response(assistant_msg["content"]):
                continue
            candidates.append({
                "user_context": user_msg["content"],
                "original_assistant": assistant_msg["content"],
            })

    print(f"Eligible prompts after quality filter: {len(candidates)} / total in train.jsonl")
    rng = random.Random(seed)
    if len(candidates) > n:
        candidates = rng.sample(candidates, n)
    return candidates


async def call_openrouter(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 200,
    temperature: float = 0.9,
    top_p: float = 0.92,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/local/businessgpt-retrain",
        "X-Title": "businessgpt-distill",
    }

    last_exc = None
    for attempt, backoff in enumerate([1, 4, 16, None]):
        try:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload, headers=headers, timeout=60.0,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                last_exc = httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                if backoff is not None:
                    await asyncio.sleep(backoff)
                    continue
                r.raise_for_status()
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError) as e:
            last_exc = e
            if backoff is None:
                raise
            await asyncio.sleep(backoff)
    raise last_exc  # pragma: no cover


async def distill_batch(
    prompts: list[dict],
    *,
    api_key: str,
    model: str,
    concurrency: int,
    out_path: Path,
    fail_path: Path,
) -> dict:
    sem = asyncio.Semaphore(concurrency)
    rejection_stats: Counter = Counter()
    accepted: list[dict] = []
    failed: list[dict] = []

    async with httpx.AsyncClient(http2=False) as client:
        async def _one(idx: int, rec: dict):
            async with sem:
                messages = [
                    {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
                    {"role": "user", "content": rec["user_context"]},
                ]
                try:
                    response = await call_openrouter(
                        client, api_key, model, messages,
                        max_tokens=200, temperature=0.9, top_p=0.92,
                    )
                except Exception as e:
                    failed.append({**rec, "_error": f"{type(e).__name__}: {e}"})
                    return None

                response = response.strip()
                reason = post_filter(response)
                if reason:
                    rejection_stats[reason] += 1
                    return None

                record = {
                    "messages": [
                        {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
                        {"role": "user", "content": rec["user_context"]},
                        {"role": "assistant", "content": response},
                    ],
                    "_meta": {
                        "prompt_id": f"distill_{idx:04d}",
                        "category": "distilled",
                        "tier": "normal",
                        "source": f"{model}-openrouter",
                        "distill_model": model,
                        "distill_params": {"temperature": 0.9, "top_p": 0.92, "max_tokens": 200},
                        "original_assistant": rec["original_assistant"],
                    },
                }
                accepted.append(record)
                return record

        tasks = [asyncio.create_task(_one(i, rec)) for i, rec in enumerate(prompts)]
        done = 0
        for fut in asyncio.as_completed(tasks):
            await fut
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] accepted={len(accepted)} rejected={sum(rejection_stats.values())} failed={len(failed)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in accepted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if failed:
        with fail_path.open("w", encoding="utf-8") as f:
            for rec in failed:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {"accepted": len(accepted), "rejected": dict(rejection_stats), "failed": len(failed)}


def review_report(out_path: Path, k: int = 30, seed: int = 42) -> None:
    """Print markdown side-by-side report for manual eyeballing."""
    if not out_path.is_file():
        print(f"Missing {out_path} — run distillation first")
        return
    records = [json.loads(line) for line in out_path.open(encoding="utf-8")]
    rng = random.Random(seed)
    sample = rng.sample(records, min(k, len(records)))

    print(f"# Distillation review — {min(k, len(records))} samples from {out_path}\n")
    for i, rec in enumerate(sample, 1):
        ctx = rec["messages"][1]["content"]
        orig = rec["_meta"].get("original_assistant", "<missing>")
        distilled = rec["messages"][2]["content"]
        prompt_id = rec["_meta"]["prompt_id"]

        print(f"\n## Sample {i}/{len(sample)} — `{prompt_id}`\n")
        print(f"### Context\n```\n{ctx[:400]}{'…' if len(ctx) > 400 else ''}\n```\n")
        print(f"### Original (train.jsonl assistant)\n```\n{orig}\n```\n")
        print(f"### Distilled\n```\n{distilled}\n```\n")
        print("**[ ] keep  [ ] reject — reason: ___**\n")
        print("---")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--sample-size", type=int, default=1000)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--review", action="store_true",
                    help="Print markdown side-by-side review for manual inspection. Does not call API.")
    args = ap.parse_args()

    if args.review:
        review_report(args.output)
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY env var not set", file=sys.stderr)
        sys.exit(1)

    if not args.train_jsonl.is_file():
        print(f"ERROR: {args.train_jsonl} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Sampling up to {args.sample_size} prompts from {args.train_jsonl}…")
    prompts = sample_prompts(args.train_jsonl, args.sample_size, args.seed)
    print(f"Will distill {len(prompts)} prompts via {args.model} (concurrency={args.concurrency})")
    print()

    fail_path = args.output.with_suffix(args.output.suffix + ".failures")
    stats = asyncio.run(distill_batch(
        prompts,
        api_key=api_key, model=args.model,
        concurrency=args.concurrency,
        out_path=args.output, fail_path=fail_path,
    ))

    print(f"\n=== Distillation summary ===")
    print(f"Sent:      {len(prompts)}")
    print(f"Accepted:  {stats['accepted']}")
    print(f"Failed:    {stats['failed']}  (see {fail_path})")
    print(f"Rejected:  {sum(stats['rejected'].values())}")
    for reason, count in sorted(stats['rejected'].items(), key=lambda x: -x[1]):
        print(f"  {reason:14}  {count}")
    print(f"\nWrote {stats['accepted']} records to {args.output}")
    print(f"Next: run `python3 {Path(__file__).name} --review` for manual eyeball")


if __name__ == "__main__":
    main()
