# BusinessGPT Roadmap

Forward-looking plan. Retrospective notes live in `PLAN.md`.

---

## 🚀 Resume from here (after PC migration)

```bash
# 1. Clone + deps
git clone https://github.com/vXofi/businessgpt
cd businessgpt
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Env vars (copy from .env.template, fill in)
cp .env.template .env
# Edit .env: set OPENROUTER_API_KEY (https://openrouter.ai/keys)
#                  HF_TOKEN (https://huggingface.co/settings/tokens, write scope)

# 3. Private data from Kaggle
#    Requires ~/.kaggle/kaggle.json (mode 600). Get from https://www.kaggle.com/settings/account
kaggle datasets download alextech123/businessraw -p . --unzip       # raw chat → result.json
kaggle datasets download avxofi/businessgpt-eval -p eval/ --unzip   # eval artifacts → eval/

# 4. Resume v16 work — Phase 1 first
source .env && export $(grep -v '^#' .env | xargs)   # load env vars
python3 eval/distill_responses.py                     # ~$1.30 OpenRouter spend, ~5 min
python3 eval/distill_responses.py --review > /tmp/review.md   # manual eyeball 30 samples
```

**Phase gates** (each must pass before moving on — details in v16 section below):

| Phase | Gate | Action |
|---|---|---|
| 1 distill | ≥25/30 review accepted | upload to Kaggle: `kaggle datasets version -p eval -m 'v16 distill'` |
| 2 SFT | eval_loss ∈ [2.8, 3.1] | run `training.ipynb` on Kaggle (LORA_REPO=v16) |
| 3 ORPO | rewards/chosen ↑, rejected ↓, no NaN | run `orpo.ipynb` on Kaggle |
| 4 RM | held-out acc ≥ 0.75 | run `reward_model.ipynb` on Kaggle |
| 5 best-of-N | ≥60% wins over 50 prompts | `python3 eval/rank_with_rm.py --version v16` then `pairwise_ui_bestof` in bench |

If any gate fails — see the Phase section below for the escalation path.

---

## Where we are (May 2026)

**v15** — Qwen3.5-9B-abliterated SFT pushed to `vXofi/businessgpt-v15-qwen3.5-9b`. Recovered from checkpoint-855 (1.5 / 2 epochs, eval_loss=3.10) after 12 h Kaggle timeout. Eval pending — run `eval_only.ipynb` for v15 generations + bench pairwise vs v14.

**v14-dpo** — DPO collapse #3: heavily spams `🏳️‍🌈 I am N% gay!` despite ~0 in training pairs (length bias × distribution drift). Documented; do not ship.

**Production fallback** — v14 SFT (1/631 = 0.16% gay-spam in outputs, below noise floor). Use until v15 is validated.

---

## v16 — three concurrent quality moves

Approved 2026-05-15. Goal: real quality jump, not regex band-aids. 14B base deferred to v17.

Serial execution: each phase gates the next.

### Phase 1: Distillation from Qwen3.5-397B-a17b via OpenRouter

- **Script**: `eval/distill_responses.py` (already written, syntax-validated)
- **Input**: 1000 contexts sampled from `train.jsonl` (filtered by `_is_quality_response`)
- **System prompt**: stricter persona than training-time SYSTEM_PROMPT — enforces 1-5 lines, no refusals, ≥1 profanity marker
- **Output record schema**: matches `sft_augment.jsonl` (training will load via existing path glob)
- **Post-filter**: length, CJK, gay-spam, refusal markers, profanity gate (regex `\b(бля|блять|нах|нахуй|хуй|пизд|ёпт|епт|сука|ебан|еба)\b`)
- **Cost**: ~$1.30 for 1000 distillations on Qwen3.5-397b-a17b. Default model `qwen/qwen3.5-397b-a17b`; escalate to `qwen/qwen3.6-max-preview` (~$3.40) if style holds < 60%
- **Run**: `OPENROUTER_API_KEY=... python3 eval/distill_responses.py`
- **Review**: `python3 eval/distill_responses.py --review > /tmp/review.md`, manual eyeball
- **Gate**: ≥25/30 samples accepted, ≥700 records survive filter

### Phase 2: v16 SFT

- `training.ipynb` cells already updated for v16: title, augment loader (sft_augment + distilled both load), SAVE_DIR, HF_REPO, streaming checkpoint repo
- Hyperparams unchanged from v15: 9B fp16, MAX_SEQ_LENGTH=1024, NUM_EPOCHS=1, batch=1, grad_accum=16, lr=5e-5, adamw_8bit, NEFTune=5, LoRA r=32 all-linear, DISTILL_REPEAT=1
- Upload distilled jsonl to Kaggle businessgpt-eval dataset before running
- Run on Kaggle T4×2; ~10-11 h
- Target: HF `vXofi/businessgpt-v16-qwen3.5-9b`
- Then `eval_only.ipynb` (LORA_REPO=v16) → `generations_v16{,_multi}.json`
- Gate: `eval_loss ∈ [2.8, 3.1]`, smoke test produces profanity-bearing Russian

### Phase 3: ORPO (replaces DPO)

- `orpo.ipynb` ready (clone of dpo.ipynb with ORPOTrainer)
- Uses `eval/preference_pairs_v14_multi.jsonl` (1450 pairs, in-distribution from v14 multi-candidate labeling)
- ORPO config: `beta=0.1`, `lr=1e-5`, `num_epochs=2`, no precompute_ref (= no Qwen3.5 hybrid fp16 NaN trigger)
- Pre-flight: `inspect.signature(ORPOConfig.__init__)` to verify param names (trl version drift)
- Stability callback ported from DPO (abort on NaN/runaway margin)
- Target: HF `vXofi/businessgpt-v16-orpo-qwen3.5-9b`
- Gate: rewards/chosen rising + rewards/rejected falling in logs; no NaN; smoke gen non-`!!!!`

### Phase 4: Reward model

- `reward_model.ipynb` ready
- Base: `DeepPavlov/rubert-base-cased` (180MB; fits production CPU RAM headroom alongside 9B Q5_K_M GGUF)
- Train on `preference_pairs_v14_multi.jsonl`: 1400 train+val, 50 held-out
- 4 epochs, lr=2e-5, batch=8, max_len=512, truncation_side=left
- Target: HF `vXofi/businessgpt-reward-rubert`
- **Strict gate**: held-out pairwise accuracy ≥ 0.75
  - 0.70-0.75 → escalate to `sbert_large_nlu_ru`, retry
  - <0.70 → labels too noisy, skip Phase 5, document failure

### Phase 5: Best-of-N inference

- `businessgpt_bench.ipynb`: `load_rm()`, `score_response()`, `chat_best_of_n()`, `pairwise_ui_bestof()` cells appended
- `eval/rank_with_rm.py`: standalone CPU script — re-ranks existing `generations_v16_multi.json` → `generations_v16_bestof.json`
- Pairwise UI compares bestof vs default (idx=1 of multi)
- Gate: ≥60% win rate over 50 labeled prompts

---

## v17 — 14B base (deferred from v16)

After v16 ships and we know whether the 9B + distill + ORPO + best-of-N stack moves the needle:

- Candidate: Qwen3.5-14B abliterated equivalent (check huihui-ai HF org for current variant; verify before pin)
- Constraint: 14B Q4_K_M ≈ 8.5 GB GGUF, fits 12 GB prod CPU RAM
- Training: QLoRA 4-bit base + fp16 LoRA on T4×2 (9B fp16 was already tight; 14B needs quantized base)
- Same data composition as v16 (chat + rap + sft_augment + distilled)
- Question to answer: is reasoning quality compute-bound at this dataset size, or does data become the bottleneck again?

---

## Open questions / unresolved

1. **Preference data staleness** — v14_multi pairs label v14 outputs. After v16, the policy has moved; "rejected" responses may no longer resemble v16 outputs. Right fix: regenerate v16_multi candidates after Phase 2, label 200 new pairs, run ORPO on `preference_pairs_v16_multi.jsonl`. Deferred (blocks Phase 3 on a labeling sprint). Acknowledged tradeoff.

2. **Single-example memorization on 9B** — "Зелёный диплом", "ни одной юбки" surface in v15 outputs from low-count training instances. The fundamental driver is 9B capacity vs ~10k example set. Regex blocklist (`_BOT_LEAK_PATTERNS` in `training.ipynb`) catches known patterns post-hoc. Structural levers if it stays a problem:
   - Drop `AUGMENT_REPEAT` 2 → 1 (less amplification of pairwise-chosen)
   - Higher `weight_decay` (0.01 → 0.05–0.1)
   - Lower LoRA rank (32 → 16) — less capacity for memorization but weaker style
   - tf-idf distinctiveness scan on train.jsonl assistant responses to flag rare-distinctive phrases for manual review

3. **Production inference glue** — `chat_best_of_n` is implemented in the bench notebook. Production runs GGUF on llama.cpp at 12 GB CPU RAM. Mirror the score-and-rank logic in deployment code (separate task, not in this repo).

4. **Style drift in distillation** — Qwen3.5-72B is RLHF'd; profanity gate is the mitigation. If post-filter rejection rate > 40%, swap distillation model (Mixtral 8x22B / Llama-3.3-70B / Qwen3.6-max) or relax the gate. Watch this on first Phase 1 run.

---

## Repo layout

```
businessgpt_retrain/
├── ROADMAP.md             # this file — upcoming work
├── PLAN.md                # retrospective notes (v1-v13)
├── training.ipynb         # SFT (v16-ready)
├── orpo.ipynb             # ORPO (v16-orpo)
├── dpo.ipynb              # legacy DPO (kept for reference)
├── reward_model.ipynb     # rubert reward model
├── eval_only.ipynb        # multi-candidate eval with HF checkpoint resume
├── businessgpt_bench.ipynb # local labeling UI + best-of-N
├── preprocess.ipynb       # Telegram JSON → train/val
├── merge_and_push.py      # LoRA → GGUF → HF
└── eval/
    ├── distill_responses.py      # Phase 1: distill from frontier model
    ├── build_sft_augment.py      # preference pairs → sft_augment.jsonl
    ├── build_preference_pairs.py # ratings → pairs
    ├── filter_train_gay_spam.py  # legacy: train.jsonl cleanup
    ├── scan_bot_patterns.py      # raw chat → bot-leak pattern discovery
    ├── rank_with_rm.py           # Phase 5: re-rank multi.json with RM
    ├── _seed_golden.py           # one-shot: build initial golden_prompts.json
    └── _diag_pairs.py            # one-shot: preference data diagnostics
```

Private data (chat JSON, train/val, generations, ratings, preference pairs) is on Kaggle dataset `alextech123/businessraw` (raw) and `<user>/businessgpt-eval` (derived). Model checkpoints are on HuggingFace under `vXofi/`.

---

## Migration checklist (when moving to a new PC)

1. `git clone <this repo>`
2. Pull private data from Kaggle: `kaggle datasets download alextech123/businessraw -p . --unzip`
3. Pull eval data from Kaggle: `kaggle datasets download <you>/businessgpt-eval -p eval/ --unzip`
4. `pip install -r requirements.txt` (TODO: create one if not present)
5. Set env: `OPENROUTER_API_KEY`, `HF_TOKEN` (for `huggingface_hub.login()`)
6. `.env.template` for reference — actual `.env` is gitignored

Models are downloaded on-demand from HF, not in the repo.
