# BusinessGPT

Training, evaluation, and export workspace for BusinessGPT: a Russian informal
group-chat chatbot tuned for short, slang-heavy Telegram-style replies.

This repo is not the production server repo. API, Docker, Caddy, and VM
deployment live in the sibling repo:

```text
../hugeballs-server
```

## Current State

- Current deployed baseline: v16 SFT, 9B Qwen3.5 abliterated line.
- Serving artifact: GGUF, with Q5_K_M as the practical production quant.
- The v16 text baseline is complete: fine-tuning clearly improves target-style
  adaptation over the base model, while v15 remains directionally preferred
  to v16 in a direct comparison.
- The deployed text configuration was rated usable on 88.4% of the absolute
  production audit. See `docs/EVALUATION_RESULTS.md` for the public-safe
  methodology, confidence intervals, and limitations.
- ORPO: attempted, parked, not shippable.
- Reward model: trained; next useful quality experiment is offline best-of-N
  reranking before any server integration.
- Runtime repetition: currently considered mostly mitigated by structured chat
  formatting, but should be monitored with real dialogs.

Use `ROADMAP.md` for the current backlog and `REPO_NOTES.md` for the compact
mental model.

## Main Files

| Path | Purpose |
| --- | --- |
| `ROADMAP.md` | Current status, active backlog, and next decisions. |
| `REPO_NOTES.md` | Repo mental model and important lessons. |
| `SCRIPT_GUIDE.md` | Commands for distillation, preference data, reward ranking, and export. |
| `PLAN.md` | Retrospective history of older model versions. |
| `docs/EVALUATION_RESULTS.md` | Public-safe v16 text evaluation results and limitations. |
| `docs/FAILURE_TRACKING.md` | Private failure-log tags and record shape. |
| `docs/TELEGRAM_EXPORTS.md` | Import Telegram Desktop HTML for private evaluation. |
| `notebooks/training.ipynb` | SFT training notebook. |
| `notebooks/eval_only.ipynb` | Candidate generation notebook. |
| `notebooks/local/businessgpt_bench.ipynb` | Manual eval, labeling, and best-of-N review UI. |
| `notebooks/reward_model.ipynb` | RuBERT reward model training. |
| `notebooks/model_eval.ipynb` | Kaggle Save & Run wrapper for HF baseline generation. |
| `notebooks/orpo.ipynb` | Parked ORPO research path. |
| `merge_and_push.py` | HF model/adapter merge, GGUF conversion, quantization, and push. |
| `eval/` | Data/eval utility scripts. |
| `prompts/` | Distillation prompt variants. |
| `scripts/legacy/` | Old one-off helpers kept out of the root. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.template .env
```

Fill `.env` with local tokens when needed:

```text
OPENROUTER_API_KEY=...
HF_TOKEN=...
```

Private data is not tracked. Pull it from Kaggle when needed:

```bash
kaggle datasets download alextech123/businessraw -p . --unzip
kaggle datasets download avxofi/businessgpt-eval -p eval/ --unzip
```

## Common Commands

Reward-model reranking:

```bash
python3 eval/rank_with_rm.py \
  --version v16 \
  --rm-repo vXofi/businessgpt-reward-rubert
```

GGUF export:

```bash
GGUF_QUANTS=Q5_K_M,Q4_K_M python3 merge_and_push.py
```

Script details live in `SCRIPT_GUIDE.md`.

Model baseline evaluation:

```bash
python -m eval.model_eval --help
```

## Privacy Rules

Do not commit chat-derived artifacts:

- `result.json`
- `train.jsonl`, `val.jsonl`, `val_examples.json`
- `eval/golden_prompts*.json`
- `eval/generations_*.json`
- `eval/ratings_*.json`
- `eval/preference_pairs*.jsonl`
- `eval/distilled_*.jsonl`
- model checkpoints, merged models, GGUFs, archives, tokens

These files are gitignored because they contain private chat fragments or
large model artifacts.

## Local Cleanup

Ignored local artifacts can become large. See `docs/LOCAL_ARTIFACTS.md` before
deleting anything.
