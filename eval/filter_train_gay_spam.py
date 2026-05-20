"""Drop train.jsonl examples where the assistant turn is "I am N% gay!" spam.

Origin: an external bot (not "BusinessGPT", which preprocess.ipynb already
filters by sender name) posted these as chat replies — they slipped through
because preprocess.ipynb only filters by sender, not content.

We keep user-context occurrences (so the model sees "this happened in the
chat") and only drop examples where the *assistant* response is this pattern,
since we don't want the trained model to reproduce it.

Run:
    python3 eval/filter_train_gay_spam.py
    python3 eval/filter_train_gay_spam.py --in val.jsonl --out val.jsonl
"""

import argparse
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GAY_SPAM_RE = re.compile(r"(?i)i\s*am\s+\d+(?:\.\d+)?%\s*gay|im\s+\d+(?:\.\d+)?%\s*gay")


def filter_file(in_path: Path, out_path: Path):
    kept, dropped = [], []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            assistant_msgs = [m for m in ex["messages"] if m["role"] == "assistant"]
            if any(GAY_SPAM_RE.search(m["content"]) for m in assistant_msgs):
                dropped.append(ex)
            else:
                kept.append(ex)

    if in_path == out_path:
        shutil.copy(in_path, in_path.with_suffix(in_path.suffix + ".bak"))
        print(f"Backup: {in_path}.bak")

    with out_path.open("w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Read {in_path}: {len(kept) + len(dropped)} examples")
    print(f"  dropped (assistant gay-spam): {len(dropped)}")
    print(f"  kept:                         {len(kept)}")
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="train.jsonl")
    ap.add_argument("--out", dest="out_path", default=None)
    args = ap.parse_args()

    in_path = (REPO / args.in_path).resolve()
    out_path = (REPO / (args.out_path or args.in_path)).resolve()
    filter_file(in_path, out_path)


if __name__ == "__main__":
    main()
