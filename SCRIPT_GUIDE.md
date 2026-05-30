# BusinessGPT Script Guide

Operational notes for local scripts. Private/generated files are gitignored. Keep chat-derived generations/ratings/preference data on Kaggle or local disk; HF is for model artifacts unless a dataset is explicitly sanitized/public.

## Quick Map

| Script | Use |
|---|---|
| `eval/distill_responses.py` | OpenRouter distillation experiments and full train-context distillation |
| `eval/build_preference_pairs.py` | Convert pairwise ratings into `preference_pairs.jsonl` |
| `eval/build_sft_augment.py` | Convert preference pairs into chosen-only SFT augment |
| `eval/rank_with_rm.py` | Re-rank multi-candidate generations with the reward model |
| `eval/purge_hf_private_eval.py` | Delete accidentally uploaded private eval artifacts from HF model repos |
| `eval/scan_bot_patterns.py` | Find repeated bot-like patterns in raw Telegram export |
| `eval/filter_train_gay_spam.py` | Legacy cleanup for `I am N% gay` target spam |
| `merge_and_push.py` | Edit config, merge LoRA, convert/quantize GGUF, push to HF |
| `test_gemma_e2b_q4.py` | Old CPU RAM/speed sanity check for Gemma E2B |

## Distillation

### Two Modes

**Experiment mode** is local and cheap. It uses `eval/golden_prompts.json` or `eval/generations_*.json` to compare models/prompts.

**Full distillation mode** creates SFT augment data from `train.jsonl`. This is what feeds training.

For multi-turn `train.jsonl` rows, distillation samples the full context before the final assistant target. It drops the training system message, joins prior user/assistant turns into one chat-history prompt, and stores the final assistant turn as `original_assistant` metadata for review.

### Experiment Mode

Basic prompt/model comparison:

```bash
python3 eval/distill_responses.py experiment \
  --source eval/golden_prompts.json \
  --sample-size 20 \
  --models '~anthropic/claude-sonnet-latest,qwen/qwen3.6-max-preview,openai/gpt-chat-latest' \
  --prompt-file prompts/distill_nextmsg_v1.txt \
  --candidates 1 \
  --max-tokens 80 \
  --temperature 0.85 \
  --concurrency 1 \
  --out eval/distill_exp_nextmsg_v1.jsonl
```

DeepSeek reasoning sanity run:

```bash
python3 eval/distill_responses.py experiment \
  --source eval/golden_prompts.json \
  --sample-size 8 \
  --models deepseek/deepseek-v4-pro \
  --prompt-file prompts/distill_nextmsg_v4_gpt.txt \
  --candidates 1 \
  --max-tokens 1000 \
  --reasoning-effort low \
  --exclude-reasoning \
  --temperature 0.9 \
  --out eval/distill_debug_deepseek_reasoning_low_1000.jsonl
```

Low vs medium DeepSeek reasoning comparison:

```bash
python3 eval/distill_responses.py experiment \
  --source eval/golden_prompts.json \
  --sample-size 12 \
  --models deepseek/deepseek-v4-pro \
  --prompt-file prompts/distill_nextmsg_v4_gpt.txt \
  --max-tokens 1000 \
  --reasoning-effort low \
  --exclude-reasoning \
  --temperature 0.9 \
  --out eval/distill_debug_deepseek_low_12.jsonl

python3 eval/distill_responses.py experiment \
  --source eval/golden_prompts.json \
  --sample-size 12 \
  --models deepseek/deepseek-v4-pro \
  --prompt-file prompts/distill_nextmsg_v4_gpt.txt \
  --max-tokens 1600 \
  --reasoning-effort medium \
  --exclude-reasoning \
  --temperature 0.9 \
  --out eval/distill_debug_deepseek_medium_12.jsonl

python3 eval/distill_responses.py summary \
  eval/distill_debug_deepseek_low_12.jsonl \
  eval/distill_debug_deepseek_medium_12.jsonl
```

Paired low-vs-high answer review on the exact same prompts:

```bash
python3 eval/distill_responses.py experiment \
  --source eval/golden_prompts.json \
  --sample-size 12 \
  --models deepseek/deepseek-v4-pro \
  --reasoning-efforts low,high \
  --prompt-file prompts/distill_nextmsg_v4_gpt.txt \
  --max-tokens 2400 \
  --exclude-reasoning \
  --temperature 0.9 \
  --out eval/distill_debug_deepseek_low_vs_high_12.jsonl

python3 eval/distill_responses.py summary eval/distill_debug_deepseek_low_vs_high_12.jsonl
```

Then review it like a model comparison:

```python
distill_experiment_ui("eval/distill_debug_deepseek_low_vs_high_12.jsonl", session_size=12, seed=42)
```

Parameters:

| Parameter | Meaning |
|---|---|
| `--source` | Local prompt source. Defaults to `eval/golden_prompts.json`. Can be `eval/generations_v15.json`. Use `--source none` only if you intentionally want `train.jsonl`. |
| `--sample-size` | Number of source prompts sampled. Keep low for expensive models. |
| `--models` | Comma-separated OpenRouter IDs. Quote the whole string if any ID starts with `~`, otherwise zsh expands it. |
| `--prompt-file` | System prompt variant to test. If omitted, uses built-in default. |
| `--candidates` | Candidates per prompt per model. `2` helps judge variance; costs double. |
| `--max-tokens` | Completion budget. Use `80-160` for non-reasoning next-message style; reasoning models may need `800-1000` because hidden reasoning is billed as output. |
| `--reasoning-effort` | Optional OpenRouter reasoning effort. For DeepSeek V4 Pro, `low` with a large token budget worked better than `none` stylistically and avoided Qwen-level cost. |
| `--reasoning-efforts` | Comma-separated reasoning efforts for paired comparison, e.g. `low,high`. The script labels rows as `model [reasoning=low]`. |
| `--reasoning-max-tokens` | Optional explicit reasoning budget. Do not combine with `--reasoning-effort`. |
| `--exclude-reasoning` | Keep reasoning out of saved response text while still using it. Use this for distillation. |
| `--fallback-model` | Full distill only. Try this model if primary fails or hits a configured reject reason. |
| `--fallback-reasoning-effort` / `--fallback-max-tokens` | Fallback reasoning/token settings. Current practical fallback is same DeepSeek model with `low` reasoning. |
| `--fallback-on` | Comma-separated triggers. Defaults to `failed,empty,refusal,too_long,too_short`. `no_profanity` is accepted by default; use `--require-profanity` only for strict experiments. |
| `--provider-only` / `--provider-order` | Optional OpenRouter provider routing. Use only for debugging provider-specific empty responses. |
| `--temperature` / `--top-p` | Sampling params sent to OpenRouter. |
| `--out` | Raw experiment JSONL. Includes accepted, rejected, and failed outputs. |

Summarize experiment costs/statuses:

```bash
python3 eval/distill_responses.py summary eval/distill_debug_deepseek_reasoning_low_1000.jsonl
```

Review in `businessgpt_bench.ipynb`:

```python
distill_experiment_ui("eval/distill_exp_nextmsg_v1.jsonl", session_size=20, seed=42)
distill_model_win_table("eval/distill_exp_nextmsg_v1.jsonl")
distill_model_failures("eval/distill_exp_nextmsg_v1.jsonl", model="openai/gpt-chat-latest", limit=20)
```

### Full Distillation Mode

First create/export `train.jsonl` from `training.ipynb` on Kaggle. The notebook now writes it from `train_data`.

Run full distillation:

```bash
python3 eval/distill_responses.py \
  --train-jsonl train.jsonl \
  --sample-size 1000 \
  --model deepseek/deepseek-v4-pro \
  --prompt-file prompts/distill_nextmsg_v4_gpt.txt \
  --output eval/distilled_deepseek_v4pro_v16.jsonl \
  --max-tokens 2400 \
  --reasoning-effort high \
  --exclude-reasoning \
  --fallback-model deepseek/deepseek-v4-pro \
  --fallback-reasoning-effort low \
  --fallback-max-tokens 1000 \
  --fallback-on failed,empty,refusal,too_long,too_short \
  --temperature 0.9 \
  --concurrency 5
```

Fallback writes accepted fallback responses to the main output and stores `attempt=fallback` plus `fallback_history` in `_meta`. If both primary and fallback fail/reject, the final failed/rejected attempt is written to `.failures` or `.rejected`.

Promote existing soft rejects without API calls:

```bash
python3 eval/distill_responses.py promote-rejected \
  --rejected eval/distilled_deepseek_v4pro_v16.jsonl.rejected \
  --output eval/distilled_deepseek_v4pro_v16.jsonl \
  --reasons no_profanity
```

Observed local debug costs on 2026-05-24:

| Run | Empty rate | Finish | Cost estimate |
|---|---:|---|---:|
| DeepSeek V4 Pro, `reasoning none`, `max_tokens=120` | 0/6 | 5 stop, 1 length | ~$1.77 / 1000 |
| DeepSeek V4 Pro, `reasoning low`, `max_tokens=240` | 3/8 | 5 stop, 3 length | ~$2.31 / 1000 |
| DeepSeek V4 Pro, `reasoning low`, `max_tokens=1000` | 0/8 empty, 1 no-profanity reject | 8 stop | ~$2.72 / 1000 |

Use the `reasoning low + max_tokens=1000 + exclude_reasoning` shape unless a larger run shows provider drift.

Outputs:

| File | Meaning |
|---|---|
| `eval/distilled_*.jsonl` | Accepted SFT-ready examples. |
| `eval/distilled_*.jsonl.failures` | API errors or empty OpenRouter content. Re-queue these. |
| `eval/distilled_*.jsonl.rejected` | Post-filter rejects, such as no profanity, too long, CJK, refusal. Retry only after prompt/filter changes. |

Retry missed rows:

```bash
python3 eval/distill_responses.py \
  --retry-jsonl eval/distilled_deepseek_v4pro_v16.jsonl.failures \
  --output eval/distilled_deepseek_v4pro_v16.jsonl \
  --model deepseek/deepseek-v4-pro \
  --prompt-file prompts/distill_nextmsg_v4_gpt.txt \
  --max-tokens 1000 \
  --reasoning-effort low \
  --exclude-reasoning \
  --concurrency 3
```

Retry rejected rows after changing the prompt:

```bash
python3 eval/distill_responses.py \
  --retry-jsonl eval/distilled_deepseek_v4pro_v16.jsonl.rejected \
  --output eval/distilled_deepseek_v4pro_v16.jsonl \
  --model deepseek/deepseek-v4-pro \
  --prompt-file prompts/distill_nextmsg_v4_gpt.txt \
  --max-tokens 1000 \
  --reasoning-effort low \
  --exclude-reasoning
```

`--retry-jsonl` appends to the output and skips prompt IDs already accepted.

Manual review sample:

```bash
python3 eval/distill_responses.py \
  --output eval/distilled_deepseek_v4pro_v16.jsonl \
  --review > /tmp/distill_review.md
```

## Preference Data

### Build Preference Pairs

Input:

- `eval/ratings_<A>_vs_<B>.json`
- matching `eval/generations_<A>.json` and `eval/generations_<B>.json`

Run:

```bash
python3 eval/build_preference_pairs.py
```

Output:

- `eval/preference_pairs.jsonl`

Optional upload:

```bash
python3 eval/build_preference_pairs.py --upload-kaggle
```

### Build Multi-Candidate Preference Pairs

Use this after `eval_only.ipynb` creates `eval/generations_v16_multi.json` and you label it with:

```python
pairwise_ui_multi("v16", session_size=30)
```

Build fresh v16 ORPO/RM pairs:

```bash
python3 eval/build_multi_preference_pairs.py --version v16
```

Output:

- `eval/preference_pairs_v16_multi.jsonl`

Then add that file to the `businessgpt-eval` Kaggle dataset before running `orpo.ipynb` or `reward_model.ipynb`.

Privacy note: `eval_only.ipynb`, `training.ipynb`, `dpo.ipynb`, and `orpo.ipynb` default to `UPLOAD_EVAL_TO_HF = False`. Leave it off for private chat-derived prompts.

If private eval artifacts were already uploaded to a model repo, dry-run and then purge:

```bash
python3 eval/purge_hf_private_eval.py --repo vXofi/businessgpt-v16-qwen3.5-9b
python3 eval/purge_hf_private_eval.py --repo vXofi/businessgpt-v16-qwen3.5-9b --yes
```

### Build SFT Augment

Input:

- `eval/preference_pairs.jsonl`

Run:

```bash
python3 eval/build_sft_augment.py
```

Output:

- `eval/sft_augment.jsonl`

This emits chosen-only SFT examples and filters CJK/gay-spam/noisy length outliers.

For v16+, `training.ipynb` keeps this old augment disabled by default via
`USE_OLD_SFT_AUGMENT=False`. Treat it as an ablation input, not the default
training mix, because it is built from old model generations selected in
v11-v13 comparisons.

## Reward Model Re-Ranking

After `eval_only.ipynb` writes `eval/generations_v16_multi.json` and `reward_model.ipynb` pushes the RM:

```bash
python3 eval/rank_with_rm.py \
  --version v16 \
  --rm-repo vXofi/businessgpt-reward-rubert
```

Output:

- `eval/generations_v16_bestof.json`

Optional HF upload for sanitized/public generations only:

```bash
python3 eval/rank_with_rm.py \
  --version v16 \
  --upload-to vXofi/businessgpt-v16-qwen3.5-9b
```

## Data Hygiene

Scan raw chat for repeated bot-like artifacts:

```bash
python3 eval/scan_bot_patterns.py \
  --raw result.json \
  --min-count 5 \
  --top 40
```

If a repeated artifact is real contamination, add a regex to `_BOT_LEAK_PATTERNS` in `training.ipynb`.

Legacy gay-spam cleanup:

```bash
python3 eval/filter_train_gay_spam.py --in train.jsonl --out train.jsonl
```

Current training already filters this inline; use the script only for old exported jsonl files.

## Export

`merge_and_push.py` is not a CLI utility. It executes immediately.

Before running, edit the config block:

```python
SOURCE_REPO = "vXofi/businessgpt-v16-qwen3.5-9b"
BASE_MODEL_ID = "huihui-ai/Huihui-Qwen3.5-9B-abliterated"
HF_GGUF_REPO = "vXofi/businessgpt-v16-qwen3.5-9b-gguf"
MERGED_DIR = "merged_model_v16_9b"
GGUF_QUANTS = ["Q5_K_M", "Q4_K_M"]
SFT_REPO = None
```

For ORPO/preference adapter export, set:

```python
SOURCE_REPO = "vXofi/businessgpt-v16-orpo-qwen3.5-9b"
SFT_REPO = "vXofi/businessgpt-v16-qwen3.5-9b"
```

Then run:

```bash
python3 merge_and_push.py
```

It downloads HF artifacts, merges LoRA, builds/updates `llama.cpp`, converts to F16 GGUF, quantizes, and pushes to HF.
