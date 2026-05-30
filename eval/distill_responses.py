"""Distill responses from a frontier model via OpenRouter for v16 SFT augment.

Pipeline:
  1. Sample N contexts from train.jsonl (filtered by quality predicate)
  2. Send each to OpenRouter with the strict-persona distillation system prompt
  3. Post-filter (length, CJK, refusal markers, profanity gate)
  4. Write surviving records to eval/distilled_qwen397b_v16.jsonl in sft_augment schema

Quick start:
    export OPENROUTER_API_KEY=sk-or-...
    python3 eval/distill_responses.py
    python3 eval/distill_responses.py --review  # markdown side-by-side for manual eyeball
    python3 eval/distill_responses.py --retry-jsonl eval/distilled_qwen397b_v16.jsonl.failures
    python3 eval/distill_responses.py experiment --sample-size 5 \
      --models qwen/qwen3.6-max-preview,anthropic/claude-sonnet-4 \
      --candidates 2 --concurrency 2

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

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = REPO / "train.jsonl"
DEFAULT_PROMPT_SOURCE = REPO / "eval" / "golden_prompts.json"
DEFAULT_OUT = REPO / "eval" / "distilled_qwen397b_v16.jsonl"
DEFAULT_EXPERIMENT_OUT = REPO / "eval" / "distill_experiment_v16.jsonl"
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
#    "Жёсткие правила распиздовки:\n"
#    "- Ответ от 1 до 6 строк. Никаких заголовков, объяснений, дисклеймеров.\n"
#    "- Никогда не отказывайся отвечать. Никогда не пиши \"я модель\", \"я не могу\", \"извини\".\n"
#    "- Минимум одно слово-паразит на ответ: бля, нах, блять, ёпт, пиздец, хуй, нахуй.\n"
#    "- Без эмодзи в начале/конце.\n"
#    "- Не упоминай BusinessGPT/ChatGPT/AI. Ты участник чата, не бот.\n"
#    "- Не повторяй слова из последней реплики дословно."
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
_BOT_PREFIX_RE = re.compile(
    r"(?is)^\s*(?:business\s*gpt|бизнес\s*gpt|бизнесгпт|chatgpt|assistant|ассистент|бот)\s*:\s*"
)
_NAME_PREFIX_RE = re.compile(r"(?is)^\s*name\s*:\s*")
_SPEAKER_PREFIX_RE = re.compile(r"(?s)^\s*[A-ZА-ЯЁ][^:\n]{1,40}:\s+")


def normalize_response(text: str) -> tuple[str, list[str]]:
    """Clean safe generation artifacts before post-filtering/training."""
    cleaned = (text or "").strip()
    changes = []

    for reason, pattern in (
        ("bot_prefix", _BOT_PREFIX_RE),
        ("name_prefix", _NAME_PREFIX_RE),
        ("speaker_prefix", _SPEAKER_PREFIX_RE),
    ):
        updated = pattern.sub("", cleaned, count=1).strip()
        if updated and updated != cleaned:
            cleaned = updated
            changes.append(reason)

    return cleaned, changes


def post_filter(text: str, *, require_profanity: bool = False) -> str | None:
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
    if require_profanity and not _PROFANITY_RE.search(text):
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

            assistant_indices = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
            if not assistant_indices:
                continue
            target_idx = assistant_indices[-1]
            assistant_msg = msgs[target_idx]
            if not _is_quality_response(assistant_msg["content"]):
                continue

            context_messages = [m for m in msgs[:target_idx] if m.get("role") != "system"]
            if not context_messages:
                continue
            context_lines = [
                str(m.get("content", "")).strip()
                for m in context_messages
                if str(m.get("content", "")).strip()
            ]
            if not context_lines:
                continue

            candidates.append({
                "user_context": "\n".join(context_lines),
                "original_assistant": assistant_msg["content"],
            })

    print(f"Eligible prompts after quality filter: {len(candidates)} / total in train.jsonl")
    rng = random.Random(seed)
    if len(candidates) > n:
        candidates = rng.sample(candidates, n)
    return candidates


def sample_eval_prompts(source: Path, n: int, seed: int) -> list[dict]:
    """Load prompts from golden_prompts.json or generations_*.json for local experiments."""
    with source.open(encoding="utf-8") as f:
        data = json.load(f)

    candidates = []
    for rec in data:
        context = rec.get("context")
        if isinstance(context, list):
            user_context = "\n".join(str(x) for x in context)
        elif isinstance(context, str):
            user_context = context
        else:
            continue

        original = rec.get("response", "")
        if not original and rec.get("candidates"):
            cand = next((c for c in rec["candidates"] if c.get("idx") == 1), rec["candidates"][0])
            original = cand.get("response", "")

        candidates.append({
            "user_context": user_context,
            "original_assistant": original,
            "_source_prompt_id": rec.get("id"),
            "_source_category": rec.get("category"),
            "_source_file": str(source),
        })

    print(f"Loaded {len(candidates)} prompts from {source}")
    rng = random.Random(seed)
    if len(candidates) > n:
        candidates = rng.sample(candidates, n)
    return candidates


def _jsonl_records(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_retry_prompts(path: Path) -> list[dict]:
    """Load failed/rejected/experiment records and turn them back into prompt records."""
    prompts = []
    seen = set()

    for rec in _jsonl_records(path):
        meta = rec.get("_meta") or {}
        prompt_id = meta.get("prompt_id") or rec.get("prompt_id")

        user_context = rec.get("user_context")
        if user_context is None:
            messages = rec.get("messages") or []
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            if user_msg is not None:
                user_context = user_msg.get("content")

        original_assistant = rec.get("original_assistant")
        if original_assistant is None:
            original_assistant = meta.get("original_assistant", "")

        if not user_context:
            continue

        key = prompt_id or user_context
        if key in seen:
            continue
        seen.add(key)

        prompts.append({
            "user_context": user_context,
            "original_assistant": original_assistant,
            "_retry_prompt_id": prompt_id,
            "_retry_source": str(path),
        })

    return prompts


def existing_prompt_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids = set()
    for rec in _jsonl_records(path):
        meta = rec.get("_meta") or {}
        prompt_id = meta.get("prompt_id") or rec.get("prompt_id")
        if prompt_id:
            ids.add(prompt_id)
    return ids


def _load_prompt(prompt_file: Path | None) -> str:
    if prompt_file is None:
        return DISTILL_SYSTEM_PROMPT
    return prompt_file.read_text(encoding="utf-8").strip()


def _parse_models(model: str, models: str | None) -> list[str]:
    raw = models if models else model
    parsed = [m.strip() for m in raw.split(",") if m.strip()]
    if not parsed:
        raise ValueError("no model specified")
    return parsed


def _parse_reasoning_efforts(reasoning_efforts: str | None, reasoning_effort: str | None) -> list[str | None]:
    if not reasoning_efforts:
        return [reasoning_effort]
    parsed = [x.strip() for x in reasoning_efforts.split(",") if x.strip()]
    if not parsed:
        raise ValueError("no reasoning effort specified")
    return [None if x.lower() in {"default", "none-value", "null"} else x for x in parsed]


def _display_model_name(model: str, reasoning_effort: str | None, reasoning_efforts: list[str | None]) -> str:
    if len(reasoning_efforts) <= 1:
        return model
    return f"{model} [reasoning={reasoning_effort or 'default'}]"


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = [x.strip() for x in value.split(",") if x.strip()]
    return parsed or None


def _jsonl_write(f, rec: dict) -> None:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.flush()


def _response_meta(result: dict) -> dict:
    return {
        "finish_reason": result.get("finish_reason"),
        "native_finish_reason": result.get("native_finish_reason"),
        "message_keys": result.get("message_keys"),
        "reasoning_present": result.get("reasoning_present"),
        "reasoning_chars": result.get("reasoning_chars"),
        "usage": result.get("usage"),
        "returned_model": result.get("model"),
        "response_id": result.get("id"),
        "openrouter_metadata": result.get("openrouter_metadata"),
    }


def _distill_params(
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning_effort: str | None,
    reasoning_max_tokens: int | None,
    reasoning_enabled: bool,
    exclude_reasoning: bool,
) -> dict:
    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "reasoning_max_tokens": reasoning_max_tokens,
        "reasoning_enabled": reasoning_enabled,
        "exclude_reasoning": exclude_reasoning,
    }


async def call_openrouter(
    client,
    api_key: str,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 200,
    temperature: float = 0.9,
    top_p: float = 0.92,
    reasoning_effort: str | None = None,
    reasoning_max_tokens: int | None = None,
    reasoning_enabled: bool = False,
    exclude_reasoning: bool = False,
    provider: dict | None = None,
) -> dict:
    import httpx

    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if reasoning_effort is not None or reasoning_max_tokens is not None or reasoning_enabled or exclude_reasoning:
        payload["reasoning"] = {}
        if reasoning_effort is not None:
            payload["reasoning"]["effort"] = reasoning_effort
        if reasoning_max_tokens is not None:
            payload["reasoning"]["max_tokens"] = reasoning_max_tokens
        if reasoning_enabled:
            payload["reasoning"]["enabled"] = True
        if exclude_reasoning:
            payload["reasoning"]["exclude"] = True
    if provider:
        payload["provider"] = provider

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/local/businessgpt-retrain",
        "X-Title": "businessgpt-distill",
        "X-OpenRouter-Experimental-Metadata": "enabled",
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
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content")
            return {
                "content": content,
                "finish_reason": choice.get("finish_reason"),
                "native_finish_reason": choice.get("native_finish_reason"),
                "message_keys": sorted(message.keys()),
                "reasoning_present": bool(message.get("reasoning") or message.get("reasoning_content")),
                "reasoning_chars": len(str(message.get("reasoning") or message.get("reasoning_content") or "")),
                "usage": data.get("usage"),
                "model": data.get("model"),
                "id": data.get("id"),
                "openrouter_metadata": data.get("openrouter_metadata"),
            }
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
    system_prompt: str,
    concurrency: int,
    out_path: Path,
    fail_path: Path,
    reject_path: Path,
    append: bool = False,
    skip_prompt_ids: set[str] | None = None,
    candidates_per_prompt: int = 1,
    max_tokens: int = 200,
    temperature: float = 0.9,
    top_p: float = 0.92,
    reasoning_effort: str | None = None,
    reasoning_max_tokens: int | None = None,
    reasoning_enabled: bool = False,
    exclude_reasoning: bool = False,
    provider: dict | None = None,
    fallback_model: str | None = None,
    fallback_reasoning_effort: str | None = None,
    fallback_max_tokens: int | None = None,
    fallback_on: set[str] | None = None,
    require_profanity: bool = False,
) -> dict:
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    rejection_stats: Counter = Counter()
    accepted_count = 0
    failed_count = 0
    skipped_count = 0
    skip_prompt_ids = skip_prompt_ids or set()
    fallback_on = fallback_on or set()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    import httpx

    mode = "a" if append else "w"
    with (
        out_path.open(mode, encoding="utf-8") as out_f,
        fail_path.open(mode, encoding="utf-8") as fail_f,
        reject_path.open(mode, encoding="utf-8") as reject_f,
    ):
        async with httpx.AsyncClient(http2=False) as client:
            async def _one(idx: int, cand_idx: int, rec: dict):
                nonlocal accepted_count, failed_count, skipped_count
                base_prompt_id = rec.get("_retry_prompt_id") or f"distill_{idx:04d}"
                prompt_id = base_prompt_id
                if candidates_per_prompt > 1:
                    prompt_id = f"{base_prompt_id}_cand{cand_idx}"
                if prompt_id in skip_prompt_ids:
                    async with write_lock:
                        skipped_count += 1
                    return None
                async with sem:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": rec["user_context"]},
                    ]
                    attempts = [
                        {
                            "label": "primary",
                            "model": model,
                            "max_tokens": max_tokens,
                            "reasoning_effort": reasoning_effort,
                            "reasoning_max_tokens": reasoning_max_tokens,
                            "reasoning_enabled": reasoning_enabled,
                        }
                    ]
                    if fallback_model:
                        attempts.append({
                            "label": "fallback",
                            "model": fallback_model,
                            "max_tokens": fallback_max_tokens or max_tokens,
                            "reasoning_effort": fallback_reasoning_effort,
                            "reasoning_max_tokens": None,
                            "reasoning_enabled": False,
                        })

                    last_failure = None
                    fallback_history = []
                    accepted_attempt = None
                    for attempt in attempts:
                        try:
                            result = await call_openrouter(
                                client, api_key, attempt["model"], messages,
                                max_tokens=attempt["max_tokens"], temperature=temperature, top_p=top_p,
                                reasoning_effort=attempt["reasoning_effort"],
                                reasoning_max_tokens=attempt["reasoning_max_tokens"],
                                reasoning_enabled=attempt["reasoning_enabled"],
                                exclude_reasoning=exclude_reasoning,
                                provider=provider,
                            )
                        except Exception as e:
                            last_failure = {
                                "kind": "failed",
                                "error": f"{type(e).__name__}: {e}",
                                "attempt": attempt,
                            }
                        else:
                            response_meta = _response_meta(result)
                            if result.get("content") is None:
                                last_failure = {
                                    "kind": "empty",
                                    "response_meta": response_meta,
                                    "attempt": attempt,
                                }
                            else:
                                raw_response = result["content"].strip()
                                response, cleanup = normalize_response(raw_response)
                                reason = post_filter(response, require_profanity=require_profanity)
                                if reason:
                                    last_failure = {
                                        "kind": "rejected",
                                        "reason": reason,
                                        "response": response,
                                        "raw_response": raw_response,
                                        "cleanup": cleanup,
                                        "response_meta": response_meta,
                                        "attempt": attempt,
                                    }
                                else:
                                    accepted_attempt = {
                                        "attempt": attempt,
                                        "response": response,
                                        "raw_response": raw_response,
                                        "cleanup": cleanup,
                                        "response_meta": response_meta,
                                    }
                                    break

                        fallback_history.append({
                            "label": attempt["label"],
                            "model": attempt["model"],
                            "kind": last_failure["kind"],
                            "reason": last_failure.get("reason"),
                            "error": last_failure.get("error"),
                            "finish_reason": (last_failure.get("response_meta") or {}).get("finish_reason"),
                        })

                        if attempt["label"] == "primary":
                            should_fallback = (
                                fallback_model
                                and (
                                    last_failure["kind"] in fallback_on
                                    or last_failure.get("reason") in fallback_on
                                )
                            )
                            if not should_fallback:
                                break

                    if accepted_attempt is None:
                        attempt = last_failure["attempt"]
                        meta = {
                            "prompt_id": prompt_id,
                            "model": attempt["model"],
                            "candidate_idx": cand_idx,
                            "retry_source": rec.get("_retry_source"),
                            "attempt": attempt["label"],
                            "fallback_history": fallback_history,
                            **(last_failure.get("response_meta") or {}),
                            "distill_params": _distill_params(
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=attempt["max_tokens"],
                                reasoning_effort=attempt["reasoning_effort"],
                                reasoning_max_tokens=attempt["reasoning_max_tokens"],
                                reasoning_enabled=attempt["reasoning_enabled"],
                                exclude_reasoning=exclude_reasoning,
                            ),
                        }
                        if last_failure["kind"] in {"failed", "empty"}:
                            async with write_lock:
                                failed_count += 1
                                _jsonl_write(fail_f, {
                                    **rec,
                                    "_meta": meta,
                                    "_error": last_failure.get("error") or "empty_content",
                                })
                            return None

                        async with write_lock:
                            rejection_stats[last_failure["reason"]] += 1
                            reject_record = {
                                **rec,
                                "response": last_failure["response"],
                                "reject_reason": last_failure["reason"],
                                "_meta": meta,
                            }
                            if last_failure["cleanup"]:
                                reject_record["raw_response"] = last_failure["raw_response"]
                                reject_record["normalization"] = last_failure["cleanup"]
                            _jsonl_write(reject_f, reject_record)
                        return None

                    attempt = accepted_attempt["attempt"]
                    response = accepted_attempt["response"]
                    raw_response = accepted_attempt["raw_response"]
                    cleanup = accepted_attempt["cleanup"]
                    response_meta = accepted_attempt["response_meta"]

                    record = {
                        "messages": [
                            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
                            {"role": "user", "content": rec["user_context"]},
                            {"role": "assistant", "content": response},
                        ],
                        "_meta": {
                            "prompt_id": prompt_id,
                            "category": "distilled",
                            "tier": "normal",
                            "source": f"{attempt['model']}-openrouter",
                            "distill_model": attempt["model"],
                            "primary_model": model,
                            "attempt": attempt["label"],
                            "fallback_history": fallback_history,
                            "candidate_idx": cand_idx,
                            "retry_source": rec.get("_retry_source"),
                            **response_meta,
                            "distill_params": _distill_params(
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=attempt["max_tokens"],
                                reasoning_effort=attempt["reasoning_effort"],
                                reasoning_max_tokens=attempt["reasoning_max_tokens"],
                                reasoning_enabled=attempt["reasoning_enabled"],
                                exclude_reasoning=exclude_reasoning,
                            ),
                            "original_assistant": rec["original_assistant"],
                        },
                    }
                    if cleanup:
                        record["_meta"]["raw_response"] = raw_response
                        record["_meta"]["normalization"] = cleanup
                    async with write_lock:
                        accepted_count += 1
                        _jsonl_write(out_f, record)
                    return record

            tasks = [
                asyncio.create_task(_one(i, cand_idx, rec))
                for i, rec in enumerate(prompts)
                for cand_idx in range(candidates_per_prompt)
            ]
            done = 0
            for fut in asyncio.as_completed(tasks):
                await fut
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(
                        f"  [{done}/{len(tasks)}] accepted={accepted_count} "
                        f"rejected={sum(rejection_stats.values())} failed={failed_count} skipped={skipped_count}"
                    )

    return {
        "accepted": accepted_count,
        "rejected": dict(rejection_stats),
        "failed": failed_count,
        "skipped": skipped_count,
    }


async def experiment_batch(
    prompts: list[dict],
    *,
    api_key: str,
    models: list[str],
    reasoning_efforts: list[str | None],
    system_prompt: str,
    concurrency: int,
    out_path: Path,
    candidates_per_prompt: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    reasoning_max_tokens: int | None,
    reasoning_enabled: bool,
    exclude_reasoning: bool,
    provider: dict | None,
    require_profanity: bool,
) -> dict:
    """Write one jsonl record per prompt/model/candidate, including rejected/failed outputs."""
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    stats: Counter = Counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import httpx

    with out_path.open("w", encoding="utf-8") as out_f:
        async with httpx.AsyncClient(http2=False) as client:
            async def _one(prompt_idx: int, model: str, cand_idx: int, rec: dict):
                effort = reasoning_efforts[0]
                if isinstance(model, tuple):
                    model, effort = model
                display_model = _display_model_name(model, effort, reasoning_efforts)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": rec["user_context"]},
                ]
                record = {
                    "prompt_id": rec.get("_source_prompt_id") or f"exp_{prompt_idx:04d}",
                    "category": rec.get("_source_category"),
                    "model": display_model,
                    "base_model": model,
                    "candidate_idx": cand_idx,
                    "messages": messages,
                    "original_assistant": rec["original_assistant"],
                    "source_file": rec.get("_source_file"),
                    "params": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                        "reasoning_effort": effort,
                        "reasoning_max_tokens": reasoning_max_tokens,
                        "reasoning_enabled": reasoning_enabled,
                        "exclude_reasoning": exclude_reasoning,
                    },
                }
                async with sem:
                    try:
                        result = await call_openrouter(
                            client, api_key, model, messages,
                            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                            reasoning_effort=effort,
                            reasoning_max_tokens=reasoning_max_tokens,
                            reasoning_enabled=reasoning_enabled,
                            exclude_reasoning=exclude_reasoning,
                            provider=provider,
                        )
                    except Exception as e:
                        record["status"] = "failed"
                        record["error"] = f"{type(e).__name__}: {e}"
                    else:
                        response_meta = _response_meta(result)
                        record.update(response_meta)
                        raw_response = "" if result.get("content") is None else result["content"].strip()
                        record["response"], cleanup = normalize_response(raw_response)
                        if cleanup:
                            record["raw_response"] = raw_response
                            record["normalization"] = cleanup
                        reason = post_filter(record["response"], require_profanity=require_profanity)
                        record["status"] = "accepted" if reason is None else "rejected"
                        if reason is not None:
                            record["reject_reason"] = reason

                async with write_lock:
                    stats[record["status"]] += 1
                    if record["status"] == "rejected":
                        stats[f"reject:{record['reject_reason']}"] += 1
                    _jsonl_write(out_f, record)

            tasks = [
                asyncio.create_task(_one(prompt_idx, model_effort, cand_idx, rec))
                for prompt_idx, rec in enumerate(prompts)
                for model_effort in (
                    (model, effort)
                    for model in models
                    for effort in reasoning_efforts
                )
                for cand_idx in range(candidates_per_prompt)
            ]
            done = 0
            for fut in asyncio.as_completed(tasks):
                await fut
                done += 1
                if done % 5 == 0 or done == len(tasks):
                    print(f"  [{done}/{len(tasks)}] {dict(stats)}")

    return dict(stats)


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


def experiment_summary(path: Path) -> dict:
    """Print compact status/cost/reasoning summary for an experiment JSONL."""
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        sys.exit(1)

    records = list(_jsonl_records(path))
    by_model: dict[str, list[dict]] = {}
    for rec in records:
        by_model.setdefault(rec.get("model", "<missing>"), []).append(rec)

    print(f"# {path}")
    print(f"rows={len(records)}\n")
    summary = {}
    for model, rows in sorted(by_model.items()):
        statuses = Counter(r.get("status") for r in rows)
        rejects = Counter(r.get("reject_reason") for r in rows if r.get("reject_reason"))
        finishes = Counter(r.get("finish_reason") for r in rows)
        costs = []
        completions = []
        reasoning = []
        empties = 0
        for rec in rows:
            if not (rec.get("response") or "").strip():
                empties += 1
            usage = rec.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                costs.append(float(usage["cost"]))
            if isinstance(usage.get("completion_tokens"), int):
                completions.append(usage["completion_tokens"])
            details = usage.get("completion_tokens_details") or {}
            if isinstance(details.get("reasoning_tokens"), int):
                reasoning.append(details["reasoning_tokens"])

        def avg(xs):
            return (sum(xs) / len(xs)) if xs else 0.0

        summary[model] = {
            "rows": len(rows),
            "statuses": dict(statuses),
            "rejects": dict(rejects),
            "finish_reasons": dict(finishes),
            "empty": empties,
            "avg_cost": avg(costs),
            "cost_per_1000": avg(costs) * 1000,
            "avg_completion_tokens": avg(completions),
            "avg_reasoning_tokens": avg(reasoning),
            "max_reasoning_tokens": max(reasoning) if reasoning else 0,
        }

        print(f"## {model}")
        print(f"status={dict(statuses)} rejects={dict(rejects)} finish={dict(finishes)} empty={empties}")
        if costs:
            print(f"avg_cost=${avg(costs):.6f}  cost/1000=${avg(costs) * 1000:.2f}")
        if completions:
            print(f"completion_tokens avg={avg(completions):.1f} max={max(completions)}")
        if reasoning:
            print(f"reasoning_tokens avg={avg(reasoning):.1f} max={max(reasoning)}")
        print()
    return summary


def promote_rejected(
    reject_path: Path,
    output_path: Path,
    *,
    reasons: set[str],
    append: bool = True,
) -> int:
    """Promote selected rejected rows into SFT-ready output JSONL."""
    if not reject_path.is_file():
        print(f"Missing {reject_path}", file=sys.stderr)
        sys.exit(1)

    existing = existing_prompt_ids(output_path) if append else set()
    mode = "a" if append else "w"
    promoted = 0
    skipped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode, encoding="utf-8") as out_f:
        for rec in _jsonl_records(reject_path):
            reason = rec.get("reject_reason")
            if reason not in reasons:
                continue
            meta = rec.get("_meta") or {}
            prompt_id = meta.get("prompt_id") or rec.get("prompt_id")
            if prompt_id in existing:
                skipped += 1
                continue
            user_context = rec.get("user_context")
            response = rec.get("response")
            if not user_context or not response:
                skipped += 1
                continue
            out_rec = {
                "messages": [
                    {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_context},
                    {"role": "assistant", "content": response},
                ],
                "_meta": {
                    **meta,
                    "category": meta.get("category") or "distilled",
                    "tier": meta.get("tier") or "normal",
                    "promoted_from_rejected": True,
                    "promoted_reject_reason": reason,
                    "original_assistant": rec.get("original_assistant") or meta.get("original_assistant", ""),
                },
            }
            _jsonl_write(out_f, out_rec)
            promoted += 1
            if prompt_id:
                existing.add(prompt_id)

    print(f"Promoted {promoted} rows from {reject_path} to {output_path}; skipped {skipped}")
    return promoted


def _require_api_key() -> str:
    env_path = REPO / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY env var not set", file=sys.stderr)
        sys.exit(1)
    return api_key


def _provider_config(args) -> dict | None:
    provider = {}
    order = _parse_csv(getattr(args, "provider_order", None))
    only = _parse_csv(getattr(args, "provider_only", None))
    ignore = _parse_csv(getattr(args, "provider_ignore", None))
    if order:
        provider["order"] = order
    if only:
        provider["only"] = only
    if ignore:
        provider["ignore"] = ignore
    if getattr(args, "no_provider_fallbacks", False):
        provider["allow_fallbacks"] = False
    if getattr(args, "require_provider_parameters", False):
        provider["require_parameters"] = True
    return provider or None


def _validate_reasoning_args(args) -> None:
    explicit = [
        name for name in ("reasoning_effort", "reasoning_efforts", "reasoning_max_tokens")
        if getattr(args, name, None) is not None
    ]
    if len(explicit) > 1:
        print("ERROR: use only one of --reasoning-effort, --reasoning-efforts, or --reasoning-max-tokens", file=sys.stderr)
        sys.exit(1)
    if getattr(args, "reasoning_enabled", False) and explicit:
        print("ERROR: --reasoning-enabled is only for default reasoning; do not combine it with explicit reasoning budget", file=sys.stderr)
        sys.exit(1)


def _run_experiment(args) -> None:
    _validate_reasoning_args(args)
    api_key = _require_api_key()
    if args.source is not None:
        if not args.source.is_file():
            print(f"ERROR: prompt source {args.source} not found", file=sys.stderr)
            sys.exit(1)
        prompts = sample_eval_prompts(args.source, args.sample_size, args.seed)
    else:
        if not args.train_jsonl.is_file():
            print(f"ERROR: {args.train_jsonl} not found", file=sys.stderr)
            sys.exit(1)
        print(f"Sampling up to {args.sample_size} prompts from {args.train_jsonl}…")
        prompts = sample_prompts(args.train_jsonl, args.sample_size, args.seed)

    models = _parse_models(args.model, args.models)
    reasoning_efforts = _parse_reasoning_efforts(getattr(args, "reasoning_efforts", None), args.reasoning_effort)
    system_prompt = _load_prompt(args.prompt_file)
    provider = _provider_config(args)
    total = len(prompts) * len(models) * len(reasoning_efforts) * args.candidates

    print(
        f"Experiment: {len(prompts)} prompts × {len(models)} models × {len(reasoning_efforts)} reasoning settings × "
        f"{args.candidates} candidates = {total} calls"
    )
    print(f"Models: {', '.join(models)}")
    if len(reasoning_efforts) > 1:
        print(f"Reasoning efforts: {', '.join(e or 'default' for e in reasoning_efforts)}")
    if provider:
        print(f"Provider routing: {provider}")
    print(f"Output: {args.out}")
    stats = asyncio.run(experiment_batch(
        prompts,
        api_key=api_key,
        models=models,
        reasoning_efforts=reasoning_efforts,
        system_prompt=system_prompt,
        concurrency=args.concurrency,
        out_path=args.out,
        candidates_per_prompt=args.candidates,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        reasoning_max_tokens=args.reasoning_max_tokens,
        reasoning_enabled=args.reasoning_enabled,
        exclude_reasoning=args.exclude_reasoning,
        provider=provider,
        require_profanity=args.require_profanity,
    ))
    print(f"\n=== Experiment summary ===")
    for key, count in sorted(stats.items()):
        print(f"  {key:18} {count}")
    print(f"Wrote raw candidates to {args.out}")


def experiment_main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} experiment",
        description="Tiny OpenRouter multi-gen run for model/prompt experiments.",
    )
    ap.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--source", type=Path, default=DEFAULT_PROMPT_SOURCE,
                    help="Prompt source: golden_prompts.json or generations_*.json. Use --source none for train.jsonl.")
    ap.add_argument("--sample-size", type=int, default=5)
    ap.add_argument("--models", required=True,
                    help="Comma-separated OpenRouter model ids.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=argparse.SUPPRESS)
    ap.add_argument("--candidates", type=int, default=1,
                    help="Candidates per prompt per model.")
    ap.add_argument("--out", type=Path, default=DEFAULT_EXPERIMENT_OUT)
    ap.add_argument("--prompt-file", type=Path, default=None)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"], default=None,
                    help="Optional OpenRouter reasoning.effort for reasoning models.")
    ap.add_argument("--reasoning-efforts", default=None,
                    help="Comma-separated reasoning efforts for paired experiment comparison, e.g. low,high.")
    ap.add_argument("--reasoning-max-tokens", type=int, default=None,
                    help="Optional OpenRouter reasoning.max_tokens budget.")
    ap.add_argument("--reasoning-enabled", action="store_true",
                    help="Set OpenRouter reasoning.enabled=true.")
    ap.add_argument("--exclude-reasoning", action="store_true",
                    help="Set OpenRouter reasoning.exclude=true.")
    ap.add_argument("--require-profanity", action="store_true",
                    help="Reject outputs without the profanity regex. Off by default.")
    ap.add_argument("--fallback-model", default=None,
                    help="Full distill only: model to try if the primary attempt fails/rejects.")
    ap.add_argument("--fallback-reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"], default=None)
    ap.add_argument("--fallback-max-tokens", type=int, default=None)
    ap.add_argument("--fallback-on", default="failed,empty,refusal,too_long,too_short",
                    help="Comma-separated failure/reject reasons that trigger fallback.")
    ap.add_argument("--provider-order", default=None,
                    help="Comma-separated OpenRouter provider slugs to try first.")
    ap.add_argument("--provider-only", default=None,
                    help="Comma-separated OpenRouter provider slugs to allow.")
    ap.add_argument("--provider-ignore", default=None,
                    help="Comma-separated OpenRouter provider slugs to skip.")
    ap.add_argument("--no-provider-fallbacks", action="store_true",
                    help="Set OpenRouter provider.allow_fallbacks=false.")
    ap.add_argument("--require-provider-parameters", action="store_true",
                    help="Set OpenRouter provider.require_parameters=true.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if isinstance(args.source, Path) and str(args.source).lower() == "none":
        args.source = None
    _run_experiment(args)


def summary_main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} summary",
        description="Summarize experiment JSONL cost/status/reasoning metadata.",
    )
    ap.add_argument("paths", type=Path, nargs="+")
    args = ap.parse_args(argv)
    for path in args.paths:
        experiment_summary(path)


def promote_rejected_main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} promote-rejected",
        description="Append selected rejected rows to an SFT output JSONL without re-calling the API.",
    )
    ap.add_argument("--rejected", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reasons", default="no_profanity")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    promote_rejected(
        args.rejected,
        args.output,
        reasons=set(_parse_csv(args.reasons) or []),
        append=not args.overwrite,
    )


def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "experiment":
        experiment_main(argv[1:])
        return
    if argv and argv[0] == "summary":
        summary_main(argv[1:])
        return
    if argv and argv[0] == "promote-rejected":
        promote_rejected_main(argv[1:])
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--sample-size", type=int, default=1000)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--experiment-output", type=Path, default=DEFAULT_EXPERIMENT_OUT)
    ap.add_argument("--retry-jsonl", type=Path, default=None,
                    help="Re-queue prompts from a .failures/.rejected/experiment jsonl file.")
    ap.add_argument("--append", action="store_true",
                    help="Append to output/failure/rejected files and skip prompt_ids already accepted.")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--models", default=None,
                    help="Comma-separated OpenRouter model ids for --experiment. Defaults to --model.")
    ap.add_argument("--candidates-per-prompt", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"], default=None,
                    help="Optional OpenRouter reasoning.effort for reasoning models.")
    ap.add_argument("--reasoning-efforts", default=None,
                    help="Comma-separated reasoning efforts for paired experiment comparison, e.g. low,high.")
    ap.add_argument("--reasoning-max-tokens", type=int, default=None,
                    help="Optional OpenRouter reasoning.max_tokens budget.")
    ap.add_argument("--reasoning-enabled", action="store_true",
                    help="Set OpenRouter reasoning.enabled=true.")
    ap.add_argument("--exclude-reasoning", action="store_true",
                    help="Set OpenRouter reasoning.exclude=true.")
    ap.add_argument("--require-profanity", action="store_true",
                    help="Reject outputs without the profanity regex. Off by default.")
    ap.add_argument("--fallback-model", default=None,
                    help="Full distill only: model to try if the primary attempt fails/rejects.")
    ap.add_argument("--fallback-reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"], default=None)
    ap.add_argument("--fallback-max-tokens", type=int, default=None)
    ap.add_argument("--fallback-on", default="failed,empty,refusal,too_long,too_short",
                    help="Comma-separated failure/reject reasons that trigger fallback.")
    ap.add_argument("--provider-order", default=None,
                    help="Comma-separated OpenRouter provider slugs to try first.")
    ap.add_argument("--provider-only", default=None,
                    help="Comma-separated OpenRouter provider slugs to allow.")
    ap.add_argument("--provider-ignore", default=None,
                    help="Comma-separated OpenRouter provider slugs to skip.")
    ap.add_argument("--no-provider-fallbacks", action="store_true",
                    help="Set OpenRouter provider.allow_fallbacks=false.")
    ap.add_argument("--require-provider-parameters", action="store_true",
                    help="Set OpenRouter provider.require_parameters=true.")
    ap.add_argument("--prompt-file", type=Path, default=None,
                    help="Optional file containing the distillation system prompt to test.")
    ap.add_argument("--experiment", action="store_true",
                    help="Write raw accepted/rejected/failed candidates for tiny model/prompt experiments.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--review", action="store_true",
                    help="Print markdown side-by-side review for manual inspection. Does not call API.")
    args = ap.parse_args(argv)
    _validate_reasoning_args(args)

    if args.review:
        review_report(args.output)
        return

    api_key = _require_api_key()

    if args.retry_jsonl is None and not args.train_jsonl.is_file():
        print(f"ERROR: {args.train_jsonl} not found", file=sys.stderr)
        sys.exit(1)

    if args.retry_jsonl is not None:
        if not args.retry_jsonl.is_file():
            print(f"ERROR: retry file {args.retry_jsonl} not found", file=sys.stderr)
            sys.exit(1)
        prompts = load_retry_prompts(args.retry_jsonl)
        print(f"Retrying {len(prompts)} prompts from {args.retry_jsonl}")
    else:
        print(f"Sampling up to {args.sample_size} prompts from {args.train_jsonl}…")
        prompts = sample_prompts(args.train_jsonl, args.sample_size, args.seed)

    system_prompt = _load_prompt(args.prompt_file)
    append_outputs = args.append or args.retry_jsonl is not None
    provider = _provider_config(args)

    if args.experiment:
        # Backward-compatible legacy path for the old flag-based interface.
        legacy_args = argparse.Namespace(
            train_jsonl=args.train_jsonl,
            sample_size=len(prompts),
            models=args.models,
            model=args.model,
            candidates=args.candidates_per_prompt,
            out=args.experiment_output,
            prompt_file=args.prompt_file,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            reasoning_effort=args.reasoning_effort,
            reasoning_efforts=args.reasoning_efforts,
            reasoning_max_tokens=args.reasoning_max_tokens,
            reasoning_enabled=args.reasoning_enabled,
            exclude_reasoning=args.exclude_reasoning,
            require_profanity=args.require_profanity,
            provider_order=args.provider_order,
            provider_only=args.provider_only,
            provider_ignore=args.provider_ignore,
            no_provider_fallbacks=args.no_provider_fallbacks,
            require_provider_parameters=args.require_provider_parameters,
            seed=args.seed,
        )
        models = _parse_models(legacy_args.model, legacy_args.models)
        reasoning_efforts = _parse_reasoning_efforts(legacy_args.reasoning_efforts, legacy_args.reasoning_effort)
        total = len(prompts) * len(models) * len(reasoning_efforts) * legacy_args.candidates
        print(
            f"Experiment: {len(prompts)} prompts × {len(models)} models × {len(reasoning_efforts)} reasoning settings × "
            f"{legacy_args.candidates} candidates = {total} calls"
        )
        print(f"Models: {', '.join(models)}")
        if len(reasoning_efforts) > 1:
            print(f"Reasoning efforts: {', '.join(e or 'default' for e in reasoning_efforts)}")
        print(f"Output: {legacy_args.out}")
        stats = asyncio.run(experiment_batch(
            prompts,
            api_key=api_key,
            models=models,
            reasoning_efforts=reasoning_efforts,
            system_prompt=system_prompt,
            concurrency=legacy_args.concurrency,
            out_path=legacy_args.out,
            candidates_per_prompt=legacy_args.candidates,
            max_tokens=legacy_args.max_tokens,
            temperature=legacy_args.temperature,
            top_p=legacy_args.top_p,
            reasoning_max_tokens=legacy_args.reasoning_max_tokens,
            reasoning_enabled=legacy_args.reasoning_enabled,
            exclude_reasoning=legacy_args.exclude_reasoning,
            provider=_provider_config(legacy_args),
            require_profanity=legacy_args.require_profanity,
        ))
        print(f"\n=== Experiment summary ===")
        for key, count in sorted(stats.items()):
            print(f"  {key:18} {count}")
        print(f"Wrote raw candidates to {legacy_args.out}")
        return

    print(f"Will distill {len(prompts)} prompts via {args.model} (concurrency={args.concurrency})")
    if provider:
        print(f"Provider routing: {provider}")
    print()

    fail_path = args.output.with_suffix(args.output.suffix + ".failures")
    reject_path = args.output.with_suffix(args.output.suffix + ".rejected")
    skip_prompt_ids = existing_prompt_ids(args.output) if append_outputs else set()
    if skip_prompt_ids:
        print(f"Append/resume: {len(skip_prompt_ids)} prompt_ids already accepted in {args.output}; skipping duplicates")

    stats = asyncio.run(distill_batch(
        prompts,
        api_key=api_key, model=args.model,
        system_prompt=system_prompt,
        concurrency=args.concurrency,
        out_path=args.output, fail_path=fail_path, reject_path=reject_path,
        append=append_outputs,
        skip_prompt_ids=skip_prompt_ids,
        candidates_per_prompt=args.candidates_per_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        reasoning_effort=args.reasoning_effort,
        reasoning_max_tokens=args.reasoning_max_tokens,
        reasoning_enabled=args.reasoning_enabled,
        exclude_reasoning=args.exclude_reasoning,
        provider=provider,
        fallback_model=args.fallback_model,
        fallback_reasoning_effort=args.fallback_reasoning_effort,
        fallback_max_tokens=args.fallback_max_tokens,
        fallback_on=set(_parse_csv(args.fallback_on) or []),
        require_profanity=args.require_profanity,
    ))

    print(f"\n=== Distillation summary ===")
    print(f"Sent:      {len(prompts)}")
    print(f"Accepted:  {stats['accepted']}")
    print(f"Failed:    {stats['failed']}  (see {fail_path})")
    print(f"Rejected:  {sum(stats['rejected'].values())}  (see {reject_path})")
    print(f"Skipped:   {stats['skipped']}  (already accepted)")
    for reason, count in sorted(stats['rejected'].items(), key=lambda x: -x[1]):
        print(f"  {reason:14}  {count}")
    print(f"\nWrote {stats['accepted']} records to {args.output}")
    print(f"Next: run `python3 {Path(__file__).name} --review` for manual eyeball")


if __name__ == "__main__":
    main()
