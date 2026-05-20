"""Re-rank generations_{V}_multi.json with the reward model.

Standalone CPU script — does NOT regenerate candidates, just scores existing
ones and writes generations_{V}_bestof.json.

Run after eval_only.ipynb finishes producing the multi.json:
    python3 eval/rank_with_rm.py --version v16
    python3 eval/rank_with_rm.py --version v16-orpo --rm-repo vXofi/businessgpt-reward-rubert

Output schema:
    [
      {
        "id", "category", "held_out", "context", "version": "{V}-bestof",
        "candidates": [{"idx", ..., "response", "rm_score"}],
        "best_idx", "best_response", "best_score"
      }, ...
    ]
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_DIR = REPO / "eval"
DEFAULT_RM_REPO = "vXofi/businessgpt-reward-rubert"


def serialize(prompt_context, response_text):
    """Match the chat template used at SFT/RM training time."""
    # Build messages: system + user (context as Имя: line OR multi-turn dict list)
    SYSTEM = (
        "Ты BusinessGPT. Пиши как студент в мессенджере: коротко, дерзко, ахуевше, "
        "по-пидорски. Часто вставляй слова-паразиты: бля, нах, блять, ёпт, пиздец."
    )
    if isinstance(prompt_context, list) and prompt_context and isinstance(prompt_context[0], str):
        ctx_text = "\n".join(prompt_context)
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": ctx_text},
        ]
    else:
        msgs = [{"role": "system", "content": SYSTEM}] + list(prompt_context)
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in msgs]
    parts.append(f"<|im_start|>assistant\n{response_text}<|im_end|>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="e.g. v16, v16-orpo")
    ap.add_argument("--rm-repo", default=DEFAULT_RM_REPO)
    ap.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--upload-to", default=None,
                    help="Optional HF repo to upload result (e.g. vXofi/businessgpt-v16-qwen3.5-9b)")
    args = ap.parse_args()

    multi_path = args.eval_dir / f"generations_{args.version}_multi.json"
    bestof_path = args.eval_dir / f"generations_{args.version}_bestof.json"

    if not multi_path.is_file():
        print(f"ERROR: {multi_path} not found. Run eval_only.ipynb for {args.version} first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading {multi_path}…")
    with multi_path.open(encoding="utf-8") as f:
        multi = json.load(f)
    print(f"  {len(multi)} prompts, ~{sum(len(e['candidates']) for e in multi)} candidates total")

    print(f"Loading RM from {args.rm_repo}…")
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(args.rm_repo)
    tok.truncation_side = "left"
    rm = AutoModelForSequenceClassification.from_pretrained(args.rm_repo)
    rm.eval()
    print(f"  loaded ({sum(p.numel() for p in rm.parameters()):,} params, CPU inference)")

    @torch.no_grad()
    def score(text):
        enc = tok(text, truncation=True, max_length=args.max_len, return_tensors="pt")
        return rm(**enc).logits.squeeze().item()

    bestof = []
    for i, entry in enumerate(multi):
        scored_candidates = []
        for cand in entry["candidates"]:
            text = serialize(entry["context"], cand["response"])
            cand_with_score = {**cand, "rm_score": score(text)}
            scored_candidates.append(cand_with_score)

        best = max(scored_candidates, key=lambda c: c["rm_score"])
        bestof.append({
            "id": entry["id"],
            "category": entry["category"],
            "held_out": entry.get("held_out", False),
            "context": entry["context"],
            "version": f"{args.version}-bestof",
            "candidates": scored_candidates,
            "best_idx": best["idx"],
            "best_response": best["response"],
            "best_score": best["rm_score"],
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(multi):
            print(f"  scored {i+1}/{len(multi)}")

    bestof_path.parent.mkdir(parents=True, exist_ok=True)
    with bestof_path.open("w", encoding="utf-8") as f:
        json.dump(bestof, f, ensure_ascii=False, indent=1)
    print(f"\nWrote {len(bestof)} entries to {bestof_path}")

    # Summary: which idx wins how often
    from collections import Counter
    idx_winners = Counter(e["best_idx"] for e in bestof)
    print(f"\nWinning candidate distribution (idx → count):")
    for idx in sorted(idx_winners):
        pct = idx_winners[idx] / len(bestof) * 100
        print(f"  idx={idx}: {idx_winners[idx]:>4}  ({pct:.1f}%)")
    if idx_winners.get(1, 0) / len(bestof) > 0.80:
        print("\n⚠️  RM picks the default-temp idx=1 candidate >80% of the time —")
        print("    re-ranking signal is weak. Either RM is undertrained or candidates")
        print("    are too similar. Consider N>4 or sbert-large RM.")

    if args.upload_to:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=bestof_path,
            path_in_repo=f"eval/{bestof_path.name}",
            repo_id=args.upload_to,
            commit_message=f"Upload {bestof_path.name} (RM={args.rm_repo})",
        )
        print(f"Uploaded to https://huggingface.co/{args.upload_to}/blob/main/eval/{bestof_path.name}")


if __name__ == "__main__":
    main()
