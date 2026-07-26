# Local Notebooks

Notebooks meant to be run locally rather than as Kaggle training jobs.

Launch Jupyter from the repository root so local files resolve as expected:

```bash
jupyter lab
```

`businessgpt_bench.ipynb` is the manual review cockpit for:

- pairwise model comparisons;
- multi-candidate labeling;
- distillation experiment review;
- reward-model best-of-N comparison.

`model_review.ipynb` is the narrower manifest-driven blind review UI for
the out-of-time v16 baseline. It resumes from append-only rating JSONL and can
show Telegram-export media from a separately supplied private media root.
