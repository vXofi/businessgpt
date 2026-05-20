"""Convert preference_pairs.jsonl → sft_augment.jsonl (chosen-only SFT examples).

Plan A for v14: re-use the labeled `chosen` responses as positive SFT examples.
Sidesteps DPO entirely — no ref model, no off-policy log-ratio, no length bias
in the loss (per-token CE is already length-normalized via gradient averaging).

Filters applied:
  - drop chosen < MIN_LEN chars (one-letter mumble — adds noise)
  - drop chosen > MAX_LEN chars (long-tail outliers, may not generalize)

Tier weighting: super-tier examples are emitted twice (replication) so they
appear 2x in training. Combined with AUGMENT_REPEAT=2 in training.ipynb,
super examples end up at 4x and normal at 2x relative weight vs raw chat data.

Output schema (one JSON per line, identical to train.jsonl format):
{
  "messages": [
    {"role": "system", "content": ...},
    {"role": "user",   "content": ...},
    {"role": "assistant", "content": chosen}
  ],
  "_meta": {"prompt_id": ..., "category": ..., "tier": ..., "source": ...}
}

Run from repo root:
    python3 eval/build_sft_augment.py
    python3 eval/build_sft_augment.py --upload-kaggle  # version businessgpt-eval dataset
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "eval"
PAIRS_PATH = EVAL_DIR / "preference_pairs.jsonl"
OUT_PATH = EVAL_DIR / "sft_augment.jsonl"

MIN_LEN = 5
MAX_LEN = 500
SUPER_REPEAT = 1  # was 2, dropped: 4× weight on one example caused "Зелёный" overfit in v14

_CJK_RE = __import__("re").compile(r"[一-鿿㐀-䶿]")
# "I am N% gay" — leak from the original (pre-abliterated) base model.
# v12+ suppressed it in outputs but the pattern is still in chosen data,
# and 9B+ capacity could re-learn it from noise.
_GAY_SPAM_RE = __import__("re").compile(r"(?i)i\s*am\s+\d+(?:\.\d+)?%\s*gay|im\s+\d+(?:\.\d+)?%\s*gay")


def build():
    if not PAIRS_PATH.is_file():
        print(f"Missing {PAIRS_PATH} — run eval/build_preference_pairs.py first")
        return None

    out = []
    stats = {
        "total": 0, "too_short": 0, "too_long": 0, "kept": 0, "emitted": 0,
        "dedup_skipped": 0,
        "by_category": Counter(), "by_tier": Counter(),
    }
    # (prompt_id, chosen_text) — same chosen winning multiple pairwise comparisons
    # for the same prompt is one positive example, not N. Without this, popular
    # winners get N× weight and overfit ("Зелёный диплом" в v14/v15).
    seen = set()

    with open(PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            stats["total"] += 1
            chosen_text = p["chosen"][-1]["content"]

            if len(chosen_text) < MIN_LEN:
                stats["too_short"] += 1
                continue
            if len(chosen_text) > MAX_LEN:
                stats["too_long"] += 1
                continue
            if _CJK_RE.search(chosen_text):
                stats.setdefault("cjk_filtered", 0)
                stats["cjk_filtered"] += 1
                continue
            if _GAY_SPAM_RE.search(chosen_text):
                stats.setdefault("gay_spam_filtered", 0)
                stats["gay_spam_filtered"] += 1
                continue

            dedup_key = (p["prompt_id"], chosen_text)
            if dedup_key in seen:
                stats["dedup_skipped"] += 1
                continue
            seen.add(dedup_key)

            ex = {
                "messages": p["prompt"] + p["chosen"],
                "_meta": {
                    "prompt_id": p["prompt_id"],
                    "category": p["category"],
                    "tier": p.get("tier", "normal"),
                    "source": p["source"],
                },
            }
            tier = p.get("tier", "normal")
            n_emit = SUPER_REPEAT if tier == "super" else 1

            for _ in range(n_emit):
                out.append(ex)
            stats["kept"] += 1
            stats["emitted"] += n_emit
            stats["by_category"][p["category"]] += n_emit
            stats["by_tier"][tier] += n_emit

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"=== Summary ===")
    print(f"Total pairs:      {stats['total']}")
    print(f"  too short:      {stats['too_short']}  (< {MIN_LEN} chars)")
    print(f"  too long:       {stats['too_long']}  (> {MAX_LEN} chars)")
    print(f"  cjk filtered:   {stats.get('cjk_filtered', 0)}")
    print(f"  gay-spam filt:  {stats.get('gay_spam_filtered', 0)}")
    print(f"  dedup skipped:  {stats['dedup_skipped']}  (same (prompt_id, chosen))")
    print(f"Unique kept:      {stats['kept']}")
    print(f"Emitted (with super-replication x{SUPER_REPEAT}): {stats['emitted']}")
    print(f"By category:      {dict(stats['by_category'])}")
    print(f"By tier:          {dict(stats['by_tier'])}")
    print(f"\nWrote {len(out)} examples to {OUT_PATH}")
    return stats


def upload_kaggle():
    metadata = EVAL_DIR / "dataset-metadata.json"
    if not metadata.is_file():
        print(f"No dataset-metadata.json in eval/ — initialize the Kaggle dataset first")
        return False

    print(f"\nVersioning Kaggle dataset...")
    cmd = ["kaggle", "datasets", "version", "-p", str(EVAL_DIR), "-m",
           "Update sft_augment + preference_pairs"]
    try:
        subprocess.check_call(cmd)
        print("Kaggle dataset updated.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Kaggle upload failed: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload-kaggle", action="store_true")
    args = ap.parse_args()

    stats = build()
    if stats is None:
        sys.exit(1)
    if args.upload_kaggle:
        upload_kaggle()


if __name__ == "__main__":
    main()
