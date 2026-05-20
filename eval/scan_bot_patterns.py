"""Pre-training scan for bot-like patterns in raw chat data.

Finds clusters of near-identical messages (same text after normalizing digits,
URLs, mentions, whitespace). High-cardinality clusters are usually bot output
that leaked in under user names (slash commands, prediction bots, etc).

Output: ranked list of clusters with count + sender distribution + sample text.
Review the output, then add regexes to KNOWN_ARTIFACT_REGEXES in training.ipynb
standard_filter so they get dropped before training.

Run:
    python3 eval/scan_bot_patterns.py
    python3 eval/scan_bot_patterns.py --min-count 10 --top 30
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW_JSON = REPO / "result.json"

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_URL_RE = re.compile(r"https?://\S+|t\.me/\S+|@\w+")
_WS_RE = re.compile(r"\s+")


def flatten_text(text_field):
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        parts = []
        for part in text_field:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts)
    return ""


def normalize(text: str) -> str:
    """Replace digits → D, URLs/mentions → @, collapse whitespace, lowercase.

    Two messages that differ only by numbers, URLs, or mentions collapse to the
    same normalized form — so 'I am 5% gay' and 'I am 88% gay' both become
    'i am d% gay'.
    """
    t = _NUM_RE.sub("D", text)
    t = _URL_RE.sub("@", t)
    t = _WS_RE.sub(" ", t).strip().lower()
    return t


def scan(raw_path: Path, min_count: int, top: int, min_len: int, max_len: int):
    with raw_path.open(encoding="utf-8") as f:
        data = json.load(f)
    messages = data.get("messages", data)

    clusters: dict[str, list[dict]] = defaultdict(list)
    total = 0
    for msg in messages:
        if msg.get("type") != "message":
            continue
        text = flatten_text(msg.get("text", "")).strip()
        if not text:
            continue
        if not (min_len <= len(text) <= max_len):
            continue
        norm = normalize(text)
        clusters[norm].append({
            "from": msg.get("from", "?"),
            "text": text,
        })
        total += 1

    # Rank by cluster size; only show clusters >= min_count
    ranked = sorted(
        ((norm, msgs) for norm, msgs in clusters.items() if len(msgs) >= min_count),
        key=lambda x: -len(x[1]),
    )

    print(f"Scanned {total} non-empty messages")
    print(f"Found {len(clusters)} unique normalized forms")
    print(f"{sum(1 for _, m in ranked)} clusters with >= {min_count} occurrences")
    print(f"\nTop {top} clusters by size:")
    print("=" * 80)

    for i, (norm, msgs) in enumerate(ranked[:top], 1):
        senders = Counter(m["from"] for m in msgs)
        n_senders = len(senders)
        bot_likely = n_senders >= 3  # many users posting the same → likely bot
        flag = "  🤖" if bot_likely else "  "
        # Pick a sample with realistic length (median-ish)
        sample = sorted(msgs, key=lambda m: len(m["text"]))[len(msgs) // 2]["text"]
        sample_short = sample[:120].replace("\n", " | ")

        print(f"\n#{i:>3}{flag}  count={len(msgs):<5}  unique_senders={n_senders:<3}")
        print(f"     normalized: {norm[:120]!r}")
        print(f"     sample:     {sample_short!r}")
        if bot_likely:
            top_senders = ", ".join(f"{s}({c})" for s, c in senders.most_common(3))
            print(f"     senders:    {top_senders}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_JSON))
    ap.add_argument("--min-count", type=int, default=5,
                    help="Show only clusters with >= N occurrences")
    ap.add_argument("--top", type=int, default=40,
                    help="Print top N clusters")
    ap.add_argument("--min-len", type=int, default=8,
                    help="Min text length (skip 'ok', 'lol', etc)")
    ap.add_argument("--max-len", type=int, default=200,
                    help="Max text length (long unique msgs aren't bot output)")
    args = ap.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.is_file():
        print(f"Raw chat JSON not found at {raw_path}")
        return

    scan(raw_path, args.min_count, args.top, args.min_len, args.max_len)


if __name__ == "__main__":
    main()
