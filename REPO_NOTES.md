# BusinessGPT Repo Notes

## Product Goal

BusinessGPT is a Russian informal group-chat chatbot trained to sound like a specific Telegram student chat: short, slang-heavy, profane, and context-aware. The repo is focused on model training, data curation, eval, preference learning, and GGUF export rather than serving or bot deployment.

## Current State

- Current roadmap target is v16.
- Production fallback is v14 SFT; v14-dpo is explicitly not shippable due gay-spam collapse.
- v15 is a 9B SFT checkpoint recovered from a partial Kaggle run and still needs eval.
- v16 plan is serial-gated:
  1. Distill extra SFT examples from OpenRouter.
  2. Train v16 SFT on Kaggle.
  3. Train ORPO adapter on preference pairs.
  4. Train RuBERT reward model.
  5. Use reward model for best-of-N candidate ranking.

## Repo Shape

- `SCRIPT_GUIDE.md`: command reference for local scripts, especially distillation experiments/full distill/retries.
- `training.ipynb`: main SFT notebook, currently v16-ready. It preprocesses raw Telegram data inline, mixes rap data, loads `eval/sft_augment.jsonl` and `eval/distilled_deepseek_v4pro_v16.jsonl`, trains LoRA on `huihui-ai/Huihui-Qwen3.5-9B-abliterated`.
- `eval_only.ipynb`: separate generation notebook for golden prompts because 9B eval does not fit inside the Kaggle training budget.
- `businessgpt_bench.ipynb`: local/manual evaluation UI, pairwise labeling, multi-candidate labeling, and best-of-N reward-model comparison.
- `orpo.ipynb`: v16 ORPO notebook replacing DPO; starts from SFT, merges it, then trains ORPO LoRA.
- `dpo.ipynb`: legacy/reference notebook. DPO has repeatedly failed and should not be the default path.
- `reward_model.ipynb`: trains `DeepPavlov/rubert-base-cased` pairwise reward model.
- `merge_and_push.py`: local HF download, LoRA merge, llama.cpp conversion, quantization, and HF push.
- `eval/*.py`: small data/eval utilities for distillation, preference pair construction, SFT augment construction, ranking, diagnostics, and bot-pattern scanning.

## Private Data And Artifacts

The repo intentionally excludes private/generated files:

- `result.json`, `train.jsonl`, `val.jsonl`, `val_examples.json`
- `eval/golden_prompts.json`
- `eval/generations_*.json`
- `eval/ratings_*.json`
- `eval/preference_pairs*.jsonl`
- `eval/sft_augment.jsonl`
- `eval/distilled_*.jsonl`

Docs say raw data comes from Kaggle dataset `alextech123/businessraw`, and derived eval artifacts come from `businessgpt-eval`.

## Important Data Lessons

- Bot contamination is a major risk. The pipeline filters BusinessGPT messages, bot mentions, replies to bot messages, and known bot-leak patterns.
- The `I am N% gay` pattern has caused repeated trouble and is filtered in training and augment scripts.
- Rap data is useful for vocabulary/style but can overgeneralize into fact/edge prompts. v13 narrowed rap triggers and reduced pure-trigger/full-verse examples.
- SFT augment from pairwise labels was adopted because DPO preference training had off-policy and length-bias problems.
- 9B capacity increases memorization risk on distinctive low-count phrases; docs mention "Зелёный диплом" as an observed leak.

## Evaluation Workflow

- Golden prompts cover chat, rap triggers, factual questions, and edge prompts.
- Eval generation produces both single/default generations and four-candidate multi generations.
- Manual pairwise ratings are stored as JSON and then converted into preference pairs.
- v16 best-of-N uses a RuBERT reward model to score candidate responses and compare RM-selected output against default candidate idx 1.

## Things To Watch

- The v16 cleanup pass fixed stale defaults in `training.ipynb`, `eval_only.ipynb`, `orpo.ipynb`, and `merge_and_push.py`.
- Distillation output is now consistently named `distilled_deepseek_v4pro_v16.jsonl` for the DeepSeek V4 Pro source model.
- `build_sft_augment.py` docs mention old super-tier replication, while `SUPER_REPEAT = 1`; code is probably intentional, comments may be stale.

## Open Questions

- Is v15 eval complete anywhere outside this repo, or should v16 start only after running `eval_only.ipynb` for v15?
- v16 distillation is the next concrete task; output filename is now `distilled_deepseek_v4pro_v16.jsonl`.
- Should ORPO use existing v14 multi preference pairs first, or should we regenerate and label v16 multi candidates before ORPO despite the extra manual labeling work?
- What deployment target should guide export defaults now: 9B Q5_K_M on 12 GB CPU RAM, or are you still considering Gemma E2B / smaller models?
- What is the hard boundary for persona content? The docs say uncensored/no safety filters, but production bot behavior may still need explicit rules for privacy leaks, real names, and training-data memorization.

## 2026-05-23 Familiarization Pass

### Mental Model

This is not an app repo yet; it is a training/evaluation lab for a private-style Telegram chatbot. The important loop is:

1. Raw/derived private data lives outside git (`result.json`, `train.jsonl`, eval artifacts).
2. `training.ipynb` builds SFT data inline from raw chat + rap + augment jsonl, then trains/pushes a LoRA.
3. `eval_only.ipynb` generates golden-prompt outputs separately because 9B eval exceeds the Kaggle training budget.
4. `businessgpt_bench.ipynb` is the manual labeling cockpit.
5. `eval/build_*` scripts turn labels into SFT augment or preference pairs.
6. `orpo.ipynb`, `reward_model.ipynb`, and `eval/rank_with_rm.py` are the planned v16 quality stack.
7. `merge_and_push.py` exports the selected HF adapter/model to GGUF for deployment.

### Verified Locally

- `python3 -m py_compile merge_and_push.py test_gemma_e2b_q4.py eval/*.py` passes.
- No tracked source changes were present at start; `REPO_NOTES.md` is untracked and being used as local project notes.
- There is no `README.md`; `ROADMAP.md` is the operational entry point and `PLAN.md` is the retrospective history.

### Current Drift / Cleanup Candidates

- Fixed: notebook vLLM install cells now use `--extra-index-url`.
- Fixed: `eval_only.ipynb` defaults `LORA_REPO` to `vXofi/businessgpt-v16-qwen3.5-9b`.
- Fixed: `training.ipynb` uses `distilled_deepseek_v4pro_v16.jsonl` and DeepSeek V4 Pro wording.
- Fixed: `training.ipynb` disabled-eval guidance points at v16.
- Fixed: ORPO notebook section headings use ORPO wording.
- Fixed: `merge_and_push.py` defaults to v16 export and uses preference-adapter wording.
- `businessgpt_bench.ipynb` still opens as "BusinessGPT v11 (2B) Test Notebook" and has many v12-v15 defaults/calls. It is useful, but it is not v16-default-safe.

### Risk Notes

- The persona is intentionally profane and private-chat-specific, so privacy/memorization risk is central, especially with real names and 9B capacity.
- Bot-leak contamination has been a recurring failure mode; `_BOT_LEAK_PATTERNS` in `training.ipynb` and `eval/scan_bot_patterns.py` are important maintenance points after any data refresh.
- Rap data is a style booster but can hijack generic prompts. v13 reduced this, and any future rap-ratio or trigger change should be evaluated against fact/edge prompts.
- Preference pairs are partly stale/off-policy (`preference_pairs_v14_multi.jsonl` for v16 ORPO/RM). This is acknowledged in `ROADMAP.md`, but it remains the highest methodological risk in v16.
