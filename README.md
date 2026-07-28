# BusinessGPT

Training, evaluation, and export workspace for BusinessGPT: a Russian informal
group-chat chatbot tuned for short, slang-heavy Telegram-style replies.

This repo is not the production server repo. API, Docker, Caddy, and VM
deployment live in the sibling repo:

```text
../hugeballs-server
```

## Project In One Minute

- **Problem:** adapt a general 9B language model to produce short,
  participant-style replies in Russian informal group chats under limited data
  and CPU-serving constraints.
- **Data and privacy:** training and evaluation use private chat-derived data.
  Raw conversations and generations are not published; evaluation uses later,
  session-aware holdouts and public aggregate reports.
- **Experiment:** train a PEFT LoRA with completion-only loss, export it to
  GGUF, and compare the base, v15, v16, and prompt variants through blinded
  human review with source-session cluster bootstrap intervals.
- **Result:** the adapted v16 artifact received 80.0% preference against the
  unadapted base artifact. On a separate audit, one reviewer judged 88.4% of
  deployed Q5 responses good or acceptable. v15 was directionally preferred
  to v16, so v16 is not claimed as a universal upgrade.
- **Limitations:** the base comparison is artifact-level, evaluation has one
  primary reviewer, and vision, long multi-turn stability, concurrency, and
  automated recovery remain unverified.
- **Decision:** keep v16 as the frozen deployed reference, retain v15 as a
  comparator, and evaluate a targeted v17 on a new session-disjoint holdout.

## Current State

- Current deployed baseline: v16 SFT, 9B Qwen3.5 abliterated line.
- Serving artifact: GGUF, with Q5_K_M as the practical production quant.
- The v16 text baseline is complete: the adapted inference artifact is clearly
  preferred to the unadapted base artifact in an artifact-level comparison,
  while v15 remains directionally preferred to v16.
- The deployed text configuration was judged good or acceptable on 88.4% of
  the absolute audit by one primary reviewer. See
  `docs/EVALUATION_RESULTS.md` for definitions, confidence intervals, and
  limitations.
- ORPO: attempted, parked, not shippable.
- Reward model: trained; next useful quality experiment is offline best-of-N
  reranking before any server integration.
- Runtime repetition: currently considered mostly mitigated by structured chat
  formatting, but should be monitored with real dialogs.

v16 remains the deployed baseline as a reference configuration, not as a claim
that it is the best checkpoint. The v15 advantage is directional, its
session-cluster interval includes parity, and it comes from one reviewer; v16
also retains a relative advantage on multiple-topic contexts. A production
rollback based on this evidence would exchange known behavior for an
insufficiently validated alternative. The measured failures instead define the
v17 hypothesis and independent promotion test.

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
| `docs/RESPONSIBLE_USE.md` | Intended use, content risk, and privacy boundary. |
| `docs/model_cards/` | Versioned source for the adapter and GGUF Hugging Face cards. |
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

## Responsible Use

BusinessGPT is a research chatbot for a closed, allowlisted informal group-chat
setting. Its persona intentionally permits profanity and can produce offensive,
biased, or otherwise unsafe text. It is not a general-purpose assistant,
factual authority, moderation system, or safety model, and its inference
endpoint is not intended for unrestricted public access.

The current public experiment manifest redacts exact operational prompts from
the main project surface and retains neutral descriptions and content hashes
for provenance. Prompt secrecy is not used as a security boundary. See
`docs/RESPONSIBLE_USE.md`.

## Local Cleanup

Ignored local artifacts can become large. See `docs/LOCAL_ARTIFACTS.md` before
deleting anything.
