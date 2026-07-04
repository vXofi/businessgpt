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
| `reward_model.ipynb` | RuBERT reward model training. |
| `orpo.ipynb` | Parked ORPO research path; not production-ready. |

## Archive

Legacy notebooks live in `archive/`. Keep them for reference, but do not use
them as the default path for new work.

