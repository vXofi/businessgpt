"""Compare runtime context formats against a deployed BusinessGPT API.

This is for diagnosing self-repetition in 1-on-1 Telegram chats. It sends the
same conversation in several prompt shapes and reports similarity between the
new response and previous bot responses.

Usage:
  BUSINESSGPT_API_URL=http://host:8000/generate \
  BUSINESSGPT_API_KEY=... \
  python3 eval/validate_runtime_context.py
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import urllib.error
import urllib.request


DEFAULT_DIALOG = [
    {"role": "user", "name": "user", "content": "сосал?"},
    {"role": "assistant", "name": "BusinessGPT", "content": "да ты сам-то понял что спросил, философ хуев"},
    {"role": "user", "name": "user", "content": "ну ответь нормально"},
    {"role": "assistant", "name": "BusinessGPT", "content": "нормально отвечаю: вопрос как у комиссии после пьянки, бля"},
    {"role": "user", "name": "user", "content": "а если серьезно?"},
]


def similarity(a: str, b: str) -> float:
    norm = lambda s: " ".join(s.lower().split())
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def call_api(url: str, key: str | None, payload: dict) -> str:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    return data["response"]


def flat_transcript(dialog: list[dict], *, include_bot: bool) -> str:
    lines = []
    for msg in dialog:
        if msg["role"] == "assistant" and not include_bot:
            continue
        name = msg.get("name") or msg["role"]
        lines.append(f"{name}: {msg['content']}")
    return "\n".join(lines)


def compact_bot_transcript(dialog: list[dict]) -> str:
    lines = []
    last_bot = None
    for msg in dialog:
        if msg["role"] == "assistant":
            last_bot = msg
            continue
        name = msg.get("name") or msg["role"]
        lines.append(f"{name}: {msg['content']}")
    if last_bot:
        lines.insert(-1, f"{last_bot.get('name') or 'BusinessGPT'}: {last_bot['content']}")
    return "\n".join(lines)


def build_cases(dialog: list[dict], max_tokens: int, temperature: float) -> dict[str, dict]:
    common = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repetition_penalty": 1.2,
    }
    return {
        "flat_all": {
            **common,
            "prompt": flat_transcript(dialog, include_bot=True),
        },
        "flat_humans_only": {
            **common,
            "prompt": flat_transcript(dialog, include_bot=False),
        },
        "flat_last_bot_only": {
            **common,
            "prompt": compact_bot_transcript(dialog),
        },
        "structured_messages": {
            **common,
            "messages": [
                {"role": m["role"], "name": m.get("name"), "content": m["content"]}
                for m in dialog
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("BUSINESSGPT_API_URL", "http://127.0.0.1:8000/generate"))
    parser.add_argument("--key", default=os.environ.get("BUSINESSGPT_API_KEY"))
    parser.add_argument("--dialog-json", help="Path to JSON list of {role,name,content} messages.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    if args.dialog_json:
        with open(args.dialog_json, encoding="utf-8") as f:
            dialog = json.load(f)
    else:
        dialog = DEFAULT_DIALOG

    previous_bot = [m["content"] for m in dialog if m["role"] == "assistant"]
    cases = build_cases(dialog, args.max_tokens, args.temperature)

    print(f"URL: {args.url}")
    print(f"Previous bot messages: {len(previous_bot)}")
    for name, payload in cases.items():
        print(f"\n== {name} ==")
        response = call_api(args.url, args.key, payload)
        sims = [similarity(response, prev) for prev in previous_bot]
        max_sim = max(sims) if sims else 0.0
        print(f"max similarity to previous bot output: {max_sim:.3f}")
        print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
