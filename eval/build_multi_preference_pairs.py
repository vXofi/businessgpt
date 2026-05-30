"""Build preference pairs from multi-candidate eval ratings.

Input:
  eval/generations_<version>_multi.json
  eval/ratings_<version>_multi.json

Each rated prompt has one best candidate. This emits one pair for every
non-best candidate: (best, other). The output format matches
preference_pairs.jsonl and can feed ORPO/RM notebooks directly.

Run:
  python3 eval/build_multi_preference_pairs.py --version v16
"""

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "eval"

SYSTEM_PROMPT = (
    "Ты BusinessGPT. Пиши как студент в мессенджере: коротко, дерзко, ахуевше, "
    "по-пидорски. Часто вставляй слова-паразиты: бля, нах, блять, ёпт, пиздец."
)

POISON_RE = re.compile(
    r"\bзел[её]н\w*|ч[её]\s+вы\s+гомики\s+молчите|а\s+ч[её]\s+вы\s+все\s+молчите|молчание\s+знак\s+согласия",
    re.IGNORECASE,
)


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_multi(version: str, *, out_path: Path | None = None) -> list[dict]:
    ratings_path = EVAL_DIR / f"ratings_{version}_multi.json"
    generations_path = EVAL_DIR / f"generations_{version}_multi.json"
    out_path = out_path or EVAL_DIR / f"preference_pairs_{version}_multi.jsonl"

    if not ratings_path.is_file():
        raise FileNotFoundError(f"missing ratings file: {ratings_path}")
    if not generations_path.is_file():
        raise FileNotFoundError(f"missing generations file: {generations_path}")

    ratings = _load_json(ratings_path)
    generations = {entry["id"]: entry for entry in _load_json(generations_path)}

    pairs = []
    seen = set()
    skipped = {
        "missing_generation": 0,
        "missing_candidate": 0,
        "identical": 0,
        "duplicate": 0,
        "skip_decision": 0,
    }

    for rating in ratings:
        pid = rating.get("prompt_id")
        if rating.get("decision") == "skip":
            skipped["skip_decision"] += 1
            continue

        entry = generations.get(pid)
        if entry is None:
            skipped["missing_generation"] += 1
            continue
        if any(POISON_RE.search(line) for line in entry.get("context", [])):
            skipped.setdefault("poison_context", 0)
            skipped["poison_context"] += 1
            continue

        best_idx = rating.get("best_candidate_idx")
        candidates = entry.get("candidates") or []
        best = next((c for c in candidates if c.get("idx") == best_idx), None)
        if best is None:
            skipped["missing_candidate"] += 1
            continue

        best_resp = (best.get("response") or "").strip()
        if not best_resp:
            skipped["missing_candidate"] += 1
            continue
        if POISON_RE.search(best_resp):
            skipped.setdefault("poison_chosen", 0)
            skipped["poison_chosen"] += 1
            continue

        for cand in candidates:
            if cand.get("idx") == best_idx:
                continue
            rejected_resp = (cand.get("response") or "").strip()
            if not rejected_resp:
                skipped["missing_candidate"] += 1
                continue
            if POISON_RE.search(rejected_resp):
                skipped.setdefault("poison_rejected", 0)
                skipped["poison_rejected"] += 1
                continue
            if rejected_resp == best_resp:
                skipped["identical"] += 1
                continue

            key = (pid, best_resp, rejected_resp)
            if key in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(key)

            pairs.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "\n".join(entry["context"])},
                ],
                "chosen": [{"role": "assistant", "content": best_resp}],
                "rejected": [{"role": "assistant", "content": rejected_resp}],
                "prompt_id": pid,
                "category": rating.get("category") or entry.get("category", "unknown"),
                "tier": rating.get("tier", "normal"),
                "source": ratings_path.name,
                "version": version,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Ratings: {len(ratings)}")
    print(f"Generations: {len(generations)}")
    print(f"Pairs: {len(pairs)}")
    print(f"Skipped: {skipped}")
    print(f"Wrote: {out_path}")
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="Version suffix, e.g. v16")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    build_multi(args.version, out_path=args.out)


if __name__ == "__main__":
    main()
