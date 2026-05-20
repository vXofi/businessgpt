"""
Convert all eval/ratings_*.json into a single eval/preference_pairs.jsonl for DPO.

For each `ratings_<A>_vs_<B>.json` file:
  - parse A/B version names from the filename
  - join with `generations_<A>.json` and `generations_<B>.json`
  - for each non-tie rating, emit (prompt, chosen, rejected, prompt_id, category, tier, source)
  - dedupe by (prompt_id, chosen, rejected)

Output (one JSON per line):
{
  "prompt":   [{"role": "system", "content": ...}, {"role": "user", "content": ...}],
  "chosen":   [{"role": "assistant", "content": ...}],
  "rejected": [{"role": "assistant", "content": ...}],
  "prompt_id": "chat_0001",
  "category":  "chat",
  "tier":      "super" | "normal",
  "source":    "ratings_v11_vs_v12-local.json"
}

Run from repo root:
    python3 eval/build_preference_pairs.py
    python3 eval/build_preference_pairs.py --upload-kaggle  # version the businessgpt-eval dataset
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "eval"
OUT_PATH = EVAL_DIR / "preference_pairs.jsonl"

# Same prompt the model was trained with — must match for DPO to behave correctly.
SYSTEM_PROMPT = (
    "Ты BusinessGPT. Пиши как студент в мессенджере: коротко, дерзко, ахуевше, "
    "по-пидорски. Часто вставляй слова-паразиты: бля, нах, блять, ёпт, пиздец."
)

# `ratings_<A>_vs_<B>.json` — A/B can contain hyphens, dots, alphanumerics.
RATING_FNAME_RE = re.compile(r"^ratings_(.+)_vs_(.+)\.json$")


def _load_generations(version):
    path = EVAL_DIR / f"generations_{version}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing generations file for version '{version}': {path}")
    with open(path, encoding="utf-8") as f:
        return {g["id"]: g for g in json.load(f)}


def build():
    ratings_files = sorted(EVAL_DIR.glob("ratings_*.json"))
    if not ratings_files:
        print(f"No ratings_*.json files in {EVAL_DIR}")
        return None

    pairs = []
    seen = set()
    stats = {
        "total_ratings": 0, "ties": 0, "self_match": 0, "duplicates": 0,
        "missing_gen": 0, "kept": 0,
        "by_source": {}, "by_category": {}, "by_tier": {},
    }

    for rf in ratings_files:
        m = RATING_FNAME_RE.match(rf.name)
        if not m:
            print(f"  skip (bad filename): {rf.name}")
            continue
        ver_a, ver_b = m.group(1), m.group(2)
        try:
            gens_a = _load_generations(ver_a)
            gens_b = _load_generations(ver_b)
        except FileNotFoundError as e:
            print(f"  skip ({rf.name}): {e}")
            continue

        with open(rf, encoding="utf-8") as f:
            ratings = json.load(f)

        n_kept_this = 0
        for r in ratings:
            stats["total_ratings"] += 1
            if r["winner"] == "tie":
                stats["ties"] += 1
                continue

            pid = r["prompt_id"]
            ga, gb = gens_a.get(pid), gens_b.get(pid)
            if ga is None or gb is None:
                stats["missing_gen"] += 1
                continue

            if r["winner"] == "A":
                chosen, rejected = ga["response"], gb["response"]
            else:  # "B"
                chosen, rejected = gb["response"], ga["response"]

            if chosen.strip() == rejected.strip():
                stats["self_match"] += 1
                continue

            key = (pid, chosen, rejected)
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)

            tier = r.get("tier", "normal")
            pair = {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "\n".join(ga["context"])},
                ],
                "chosen":   [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "prompt_id": pid,
                "category":  r["category"],
                "tier":      tier,
                "source":    rf.name,
            }
            pairs.append(pair)
            stats["kept"] += 1
            n_kept_this += 1
            stats["by_category"][r["category"]] = stats["by_category"].get(r["category"], 0) + 1
            stats["by_tier"][tier] = stats["by_tier"].get(tier, 0) + 1

        stats["by_source"][rf.name] = n_kept_this
        print(f"  {rf.name}: kept {n_kept_this} / {len(ratings)} ratings")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n=== Summary ===")
    print(f"Total ratings:    {stats['total_ratings']}")
    print(f"  ties skipped:   {stats['ties']}")
    print(f"  self-match:     {stats['self_match']}")
    print(f"  duplicates:     {stats['duplicates']}")
    print(f"  missing gen:    {stats['missing_gen']}")
    print(f"Pairs kept:       {stats['kept']}")
    print(f"By category:      {stats['by_category']}")
    print(f"By tier:          {stats['by_tier']}")
    print(f"\nWrote {len(pairs)} pairs to {OUT_PATH}")
    return stats


def upload_kaggle():
    """Version-bump the businessgpt-eval Kaggle dataset so dpo.ipynb sees the new file.

    Requires:
      - kaggle CLI installed and authenticated (~/.kaggle/kaggle.json)
      - eval/dataset-metadata.json present (created on first dataset upload)
    """
    metadata = EVAL_DIR / "dataset-metadata.json"
    if not metadata.is_file():
        print(
            "No dataset-metadata.json in eval/ — initialize the Kaggle dataset first:\n"
            "  cd eval && kaggle datasets init -p .\n"
            "  # edit dataset-metadata.json: set 'id' to '<your-username>/businessgpt-eval'\n"
            "  kaggle datasets create -p .\n"
            "Then re-run with --upload-kaggle."
        )
        return False

    print(f"\nVersioning Kaggle dataset...")
    cmd = ["kaggle", "datasets", "version", "-p", str(EVAL_DIR), "-m",
           f"Update preference_pairs (and golden_prompts)"]
    try:
        subprocess.check_call(cmd)
        print("Kaggle dataset updated.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Kaggle upload failed: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload-kaggle", action="store_true",
                    help="After building, version the businessgpt-eval Kaggle dataset.")
    args = ap.parse_args()

    stats = build()
    if stats is None:
        sys.exit(1)
    if args.upload_kaggle:
        upload_kaggle()


if __name__ == "__main__":
    main()
