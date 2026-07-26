# Notebooks

Notebook workspace for training, eval generation, local review, and preference
experiments.

Launch Jupyter from the repo root when running these locally:

```bash
jupyter lab
```

Most notebook cells assume paths such as `eval/...`, `prompts/...`, and
`train.jsonl` are relative to the repository root.

## Active

| Notebook | Purpose |
| --- | --- |
| `training.ipynb` | v16 SFT training on Kaggle. |
| `eval_only.ipynb` | Golden/multi-candidate generation after training. |
| `local/businessgpt_bench.ipynb` | Local manual review, labeling UI, distillation review, best-of-N review. |
| `reward_model.ipynb` | Group-safe RuBERT reward model training, held-out ranking gate, and opt-in versioned publishing. |
| `model_eval.ipynb` | Thin Kaggle Save & Run wrapper for manifest-defined v15/v16/base generation. |
| `orpo.ipynb` | Parked ORPO research path; not production-ready. |

`model_eval.ipynb` writes only to Kaggle working storage. Download its
JSONL output and review it locally; do not add chat-derived eval data as an HF
model-repo artifact. Every profile is a separate self-contained Save & Run, and
the final cell fails if the JSONL is partial or contains generation errors.

## Archive

Legacy notebooks live in `archive/`. Keep them for reference, but do not use
them as the default path for new work.
