# BusinessGPT Retrain -- Project Report

## Goal

Fine-tune a small LLM on a Telegram group chat dataset + Russian rap lyrics so the model learns to respond naturally in the group's conversational style with rich slang vocabulary. No safety filters, no censorship.

## Model

- **Base (v11-v12)**: `huihui-ai/Huihui-Qwen3.5-2B-abliterated` (1.9B params, hybrid linear+full attention, abliterated/uncensored)
- **Base (v8)**: `huihui-ai/Huihui-Qwen3.5-0.8B-abliterated` (0.8B params, VL architecture, abliterated/uncensored)
- **Base (v5-v7)**: `huihui-ai/Huihui-Qwen3-0.6B-abliterated-v2` (0.6B params, text-only, abliterated/uncensored)
- **Base (v1-v4)**: `Qwen/Qwen3-VL-2B-Instruct` (2.1B params, vision-language)
- **Method (v12)**: LoRA (r=32, alpha=64, all-linear, no DoRA) in fp16, completion-only loss, real names, **gay-spam filter**, **rap 15% with chat-style triggers**, NEFTune noise. T4x2.
- **Method (v11)**: LoRA (r=32, alpha=64, all-linear, no DoRA) in fp16, completion-only loss, real names, 5% rap with `[Artist]` prefix only. T4x2.
- **Method (v8)**: LoRA (DoRA, r=32, alpha=64) in fp32, completion-only loss, real user names, NEFTune noise
- **Method (v7)**: Full fine-tuning in fp16 (all 596M params trainable), NEFTune noise, early stopping. No LoRA.
- **Method (v6)**: Full fine-tuning in fp16 (all 596M params trainable), NEFTune noise. No LoRA.
- **Method (v5)**: LoRA via Unsloth (DoRA, r=32, alpha=64), fp16
- **System prompt (v12)**: `"Ты BusinessGPT. Пиши как студент в мессенджере: коротко, дерзко, ахуевше, по-пидорски. Часто вставляй слова-паразиты: бля, нах, блять, ёпт, пиздец."`
- **System prompt (≤v8)**: `"Ты BusinessGPT. Пиши как студент в мессенджере: коротко, дерзко, без цензуры."`

## Published Artifacts

| Version | Safetensors | GGUF | Notes |
|---------|------------|------|-------|
| v2 | `vXofi/businessgpt-qwen3vl-2b-v2` | `vXofi/businessgpt-qwen3vl-2b-v2-q4_gguf` | Chat only, no anonymous tokens |
| v3 | `vXofi/businessgpt-qwen3vl-2b-v3` | `vXofi/businessgpt-qwen3vl-2b-v3-q4_gguf` | Chat only, anonymous tokens, DoRA/NEFTune, val loss |
| v4 | `vXofi/businessgpt-qwen3vl-2b-v4` | `vXofi/businessgpt-qwen3vl-2b-v4-q4_gguf` | Chat + rap lyrics, Qwen3-VL-2B, QLoRA 4-bit |
| v5 | `vXofi/businessgpt-v5-qwen3-0.6b` | `vXofi/businessgpt-v5-qwen3-0.6b-q8_gguf` | 0.6B abliterated, date-split data, LoRA/DoRA fp16, Q8_0 GGUF |
| v6 | `vXofi/businessgpt-v6-qwen3-0.6b` | `vXofi/businessgpt-v6-qwen3-0.6b-gguf` (F16 + Q8_0) | Full fine-tuning, simplified next-msg format, no `<think>` blocks, 6 epochs |
| v7 | `vXofi/businessgpt-v7-qwen3-0.6b` | TBD | **Full fine-tuning**, 3 epochs, lower LR (2e-5), early stopping, stronger regularisation |
| v8 | `vXofi/businessgpt-v8-qwen3.5-0.8b` | TBD | **LoRA + completion-only loss**, Qwen3.5-0.8B, real names, 5% rap, response quality filtering |
| v9-v10 | (undocumented) | — | Iteration on Qwen3.5 architecture; not retained in this report |
| v11 | `vXofi/businessgpt-v11-qwen3.5-2b` | `vXofi/businessgpt-v11-qwen3.5-2b-gguf` (F16/Q8_0/Q4_K_M) | Qwen3.5-2B base, LoRA r=32 (no DoRA — fp16 stability), 2 ep, lr=5e-5. First "good enough" model per user; fp8 inference is dumber but acceptable. |
| v12 | `vXofi/businessgpt-v12-qwen3.5-2b` | TBD | v11 + **gay-spam filter** (7.45% → 0.16%) + **rap 5%→15% with chat-style triggers**. Trigger format over-generalised: model started rapping on fact/edge prompts (median length 16→305 on facts). Chat use-case unaffected. |
| v13 | `vXofi/businessgpt-v13-qwen3.5-2b` | TBD | v12 + **narrowed rap triggers**. Rap 15%→10%, format C (pure trigger → 8 lines) 33%→10%, dropped 5/13 generic trigger templates without explicit read/sing verb. Goal: keep rap-on-explicit-request, kill rap-on-anything-short. |

---

## Dataset

### Chat Data

- **Source**: Kaggle private dataset `alextech123/businessraw` (`result.json` Telegram export)
- **Chat**: "Бизнес" (private_supergroup), 2024-10-21 to 2025-12-16
- **Language**: Russian, very informal -- university students, heavy slang/profanity
- **Raw**: 15,889 messages, 11 senders

### Date-Based Split (v5)

The chat is split at the **first appearance of BusinessGPT** in the group:

- **Pre-bot data**: All messages before the bot was introduced. This is clean, uncontaminated conversational data. Used as the **primary training set**.
- **Post-bot data**: Messages after the bot was introduced. Enhanced filtering removes:
  - All existing bot filters (sender, @mentions, /generate, replies)
  - Additional regex scrubbing: `businessgpt`, `бизнесгпт`, `бизнес gpt`, `бизнес бот`, `гпт бот`
- **Mix ratio**: 15% of post-bot examples mixed into training data (`POST_BOT_MIX_RATIO = 0.15`, tunable 0.10-0.20)

### Rap Lyrics Data (added in v4)

- **Source**: `wadzim/modern-russian-rap` Kaggle dataset (mounted at `/kaggle/input/modern-russian-rap/`)
- **Columns**: `Artist`, `Genius Artist Name`, `Title`, `Lyric`
- **26 selected artists**: Pharaoh, Элджей, Тима Белорусских, Скриптонит, Пошлая Молли, ЛСП, Егор Крид, Макс Корж, Каспийский Груз, Гуф, АК-47, Yanix, Slava Marlow, Oxxxymiron, MORGENSHTERN, Lizer, Kizaru, GONE.Fludd, Дора, Мэйби Бэйби, INSTASAMKA, Платина, MACAN, FRIENDLY THUG 52 NGG, Boulevard Depo, SALUKI
- **IMPORTANT**: Lyrics have NO newlines -- stored as continuous text. `split_lines()` splits on sentence-ending punctuation followed by uppercase: `(?<=[.!?])\s*(?=[А-ЯA-Z«"])`
- Artist matching: case-insensitive on both `Artist` and `Genius Artist Name` columns
- Chunking: window=8 lines, stride=4. First 3 lines = user (prefixed `[Artist]`), rest = assistant. Fallback for 5-7 line songs.
- Capped at 20% of total training data (v6, was 30% in v4-v5)

---

## Preprocessing Pipeline

### Chat Pipeline (v12)

Inline in `training.ipynb` (not the separate `preprocess.ipynb`). Differences vs v5 doc below:

1. **Standard filter** additionally drops messages matching `r"I am \d+% gay"` (Telegram "gay-test" game bot output that polluted the chat as fake user turns; before v12 it was 169/5229 train targets and 801 contexts). Same regex check at record-extraction time so it can't appear as either context or target.
2. **Real names** instead of anonymous `<user_N>` tokens (since v8): `Name: message`. Provides semantic grounding.
3. **Multi-turn examples** with data augmentation: each session position generates 2 examples at window sizes (10, 15). Target user's prior messages become assistant turns.
4. **Response quality filter**: 3 ≤ len ≤ 300 chars, drop emoji-only, drop slash-commands, dedupe.
5. **Session split**: 1-hour gap, min 5 messages.
6. **Pre-bot only**: post-bot data dropped entirely (v12); was mixed at 15% in v5–v8.

### Chat Pipeline (v5 -- historical)

1. **Load JSON** via `kagglehub.dataset_download()`
2. **Extract & flatten**: Parse messages, flatten mixed text field
3. **Find bot introduction date**: First message from "BusinessGPT" sender → `bot_introduction_ts`
4. **Split by date**: `records_pre` (before bot) and `records_post` (after bot)
5. **Standard filter** (both halves):
   - Drop service messages, empty text, media-only
   - Drop `from == "BusinessGPT"`
   - Drop messages containing `@businessgpt_text_bot` or `/generate`
   - Drop messages replying to bot messages (`reply_to_message_id in bot_msg_ids`)
6. **Enhanced filter** (post-bot only): additionally drop messages matching bot name regex patterns
7. **Merge consecutive**: Same sender within 60s -> single turn (each half separately)
8. **Anonymous tokens**: Map real usernames to `<user_1>`, `<user_2>`, etc. (each half independently)
9. **Session split**: 2-hour gap, min 11 messages -> sessions
10. **Sliding window**: `window_size=15`, `min_context=3` (v6, was 10 in v5) -> examples. Each example: flat context lines joined by `\n` as single "user" message, target as "assistant"
11. **Mix**: Sample `POST_BOT_MIX_RATIO` (15%) of post-bot examples into pre-bot examples
12. **Target response**: Plain text, NO speaker token prefix (model responds as itself)
13. **Token filter**: max 2,048 tokens

### Rap Pipeline (v13 -- latest)

Same structure as v12 (3 formats A/B/C, 25 artists). Five differences vs v12:

1. **Format weights A=40% / B=50% / C=10%** (was 33/33/33). Format C was the dominant cause of v12 overgeneralisation. Reduced 33%→10%.
2. **8 trigger templates instead of 13.** Dropped: `{artist}?`, `{artist} бля`, `давай {artist}`, `хочу {artist}`, `а {artist} знаешь?`. Kept verb-anchored: `зачитай`, `кинь трек`, `спой`, `трек от {artist} пж`, `зачитать {artist}?`, `{artist} читни`, `а ну {artist} ебани`, `мне тут {artist} напомнили`.
3. **`TARGET_RATIO` 0.15 → 0.10.**
4. **`window` 8 → 5, stride 4 → 3.** A/B split 2 user lines + 3 assistant lines; C = full 5 lines. Hard cap on assistant output at 5 lines (was 8). Labelers reported 8 was «перебор» — chat-natural ceiling is ~5.
5. **`_join_lyric()` normalisation:** assistant content joins lines via `_join_lyric()`, which 50/50 picks `". "` (sentence-style) or `"\n"` (line-break). v12 always used `\n`, which made the model produce newline-heavy output even on short chat replies; balancing with punctuation breaks that lock-in.

Quantitative target: ~2.5% of training data is "short prompt → ≤5-line lyric" (was 5% × 8-line in v12) — and half of those use sentence-style joins, so newline-heavy patterns are ~1.25% of data.

### Rap Pipeline (v12)

1. Load CSV from `/kaggle/input/datasets/wadzim/modern-russian-rap/`
2. Filter to 25 selected artists (down from 26)
3. Split lyrics into lines (newline OR sentence-boundary regex)
4. Chunk with sliding window=8, stride=4
5. **Three formats per chunk** (chosen ~uniformly):
   - **A** (~33%) `[Artist]\n<3 lines>` → continuation (preserves original song-continuation skill)
   - **B** (~33%) `{name}: {trigger} {artist}\n<3 lines>` → continuation (bridges chat → rap)
   - **C** (~33%) `{name}: {trigger} {artist}` → full 8-line window (pure dialogue-initiated rap; the trigger that lets a chat prompt like `Некит Русанов: зачитай Элджей` start a verse)
6. 13 trigger templates × 9 real chat names → wide trigger surface
7. Cap at **15%** of total training examples (was 5% in v8/v11, 20% in v6)
8. **Outcome (measured via eval framework)**: model started rapping on fact/edge prompts. fact median length 16→305, edge 19→329; chat unaffected (18→21). Drove the v13 fix.

### Rap Pipeline (v6 -- historical)

1. Load CSV from `/kaggle/input/modern-russian-rap/`
2. Filter to 26 selected artists
3. Split lyrics into lines (sentence-boundary regex)
4. Chunk with sliding window, format as system/user/assistant
5. Cap at 20% ratio (v6)

### Combined

- Chat (pre + sampled post) + rap examples combined, shuffled (seed=42)
- Val split: 5% or min 50 examples
- Converted to HF Dataset with `text` field via manual `format_chat()` (avoids Qwen3's `<think>` injection)

### Training Data Format (v6)

Simple next-message prediction: flat context -> response (no role-swapping, no user token in response):

```json
{
  "messages": [
    {"role": "system", "content": "Ты BusinessGPT. Пиши как студент в мессенджере: коротко, дерзко, без цензуры."},
    {"role": "user", "content": "<user_1> кто сделал 15 практику?\n<user_2> я не делал\n<user_3> я тоже нет"},
    {"role": "assistant", "content": "каким медведем нахуй"}
  ]
}
```

Key differences from v5: all context messages are in a single flat `user` message (with `<user_X>` tokens), the assistant response is plain text (no user token prefix), always exactly 3 messages per example.

---

## Training (v15 -- latest)

**Model size jump 2B → 9B** (`huihui-ai/Huihui-Qwen3.5-9B-abliterated`). This is the only change vs v14.

**Why bigger model now**: pairwise of v14 vs v13 confirmed style is saturated — the 2B model knows the slang, the names, the rhythm. The remaining quality gap is in **reasoning**: connecting the style to a meaningful, contextually relevant reply. Reasoning depth is compute-bound (multi-layer abstraction), so it scales with parameters, not data. ~1700 labeled pairs already in hand, more data wouldn't help past this ceiling.

**Why 9B specifically**:
- Same Qwen3.5 family → drop-in pipeline (chat template, tokenizer, system prompt, hybrid linear_attn DPO concerns all already characterized)
- Q5_K_M GGUF ≈ 6.4 GB → fits new 12 GB CPU RAM prod constraint with comfortable KV-cache headroom
- ~2.25× compute capacity vs 4B sweet-spot alternative; same RAM footprint at Q5_K_M

**Hyperparameter changes vs v14** (memory-driven, not learning-driven):
- `MAX_SEQ_LENGTH` 4096 → 2048: 9B activations are bigger, T4×2 (30 GB total) needs the budget
- `GRAD_ACCUM` 8 → 16: keeps effective batch=16 while halving per-step memory pressure
- Everything else stays — same LR, optimizer, NEFTune, etc.

**Pipeline change**: inline eval at end of `training.ipynb` is **disabled** (`RUN_EVAL_AT_END=False`). 9B inference at 4 candidates × 631 prompts blows past Kaggle's 12 h budget. Eval lives in `eval_only.ipynb` which has its own fresh budget + HF-checkpointed resume.

### Training (v14)

Identical hyperparameter setup to v13. Two data-only changes:
1. **SFT augmentation** from labeled preference pairs. `eval/sft_augment.jsonl` (1144 unique chosen-only examples; super tier 2x replicated in the file) is loaded as additional training data with `AUGMENT_REPEAT=2`. Each unique chosen example appears 2x in training, super-tier 4x. Total ~2400 augment examples added to ~9,200 chat + ~480 rap.
2. **Eval cell generates 4 candidates per prompt** (varied temp 0.7/0.95/1.1/1.3) and saves both `generations_v14.json` (single, default-temp candidate for backward-compat pairwise) and `generations_v14_multi.json` (4 candidates, for plan C in-distribution DPO data collection).

**Why this approach** (replaces v13-dpo): three DPO attempts on v13 all collapsed (NaN in fp16, soft entropy collapse in fp32, identical pattern with tighter ref-anchor). Diagnostic on `preference_pairs.jsonl` revealed structural problems unfixable by hyperparams: 100% of pairs were OUT-OF-DISTRIBUTION for the v13 ref model (generated by v11/v12), and chosen was systematically 16% shorter than rejected on v12-vs-v13 pairs (length bias). Plan A (rejection-sampling SFT, this v14) sidesteps both issues — per-token CE has no length bias and no off-policy log-ratio. Plan C (DPO from in-distribution v14 self-pairs) becomes viable once v14 exists.

### Training (v13)

Identical hyperparameter setup to v12. Only the rap data pipeline changed — see "Rap Pipeline (v13)" below.

### Training (v12)

| Parameter | Value |
|---|---|
| Base model | `huihui-ai/Huihui-Qwen3.5-2B-abliterated` (Qwen3.5 2B, hybrid linear+full attention) |
| Method | LoRA (r=32, alpha=64, all-linear, dropout=0.05, **DoRA disabled** — magnitude norm unstable in fp16 on bf16-native base) |
| Loss | Completion-only via manual label masking on last `<\|im_start\|>assistant\n` prefix |
| Data format | Multi-turn `messages` (system / user / assistant pairs), full-conversation tokenization |
| Examples | ~9,200 chat (multi-turn, augmented) + ~1,630 rap (3-format mix) ≈ 10,800 total |
| Epochs | 2 |
| Effective batch size | 8 (1 × 8 grad_accum) |
| Learning rate | 5e-5 (cosine, conservative — v11 was first model to actually answer in-context, not aggressively overfitting) |
| Warmup steps | 50 |
| Optimizer | AdamW (torch) |
| Weight decay | 0.01 |
| Max seq length | 4,096 |
| NEFTune | alpha=5 |
| Chat template | Custom (no `<think>` injection) |
| Eval strategy | Every `STEPS_PER_EPOCH // 2` |
| Save strategy | Same as eval, `load_best_model_at_end=True`, `save_total_limit=2` |
| Hardware | Kaggle T4 ×2 |
| Precision | fp16 mixed precision |
| Inference quantization | Q8_0 GGUF for prod; fp8 also tested but noticeably dumber |

### Why these hyperparams stay conservative

v11 was the first version where the model "answers in context, doesn't spam strings from train". Cranking LR / DoRA / rank without an objective signal risks regressing that. v12 keeps the same training setup and changes only the **data** (gay-spam removed, rap reformatted). Quality changes from v11→v12 should be attributable to data, not hyperparams.

### Training (v8)

| Parameter | Value |
|---|---|
| Base model | `huihui-ai/Huihui-Qwen3.5-0.8B-abliterated` (Qwen3.5 0.8B, VL arch) |
| Method | LoRA (DoRA, r=32, alpha=64, all-linear targets, dropout=0.05) |
| Loss | **Completion-only** (loss computed only on assistant response tokens) |
| Data format | `prompt` / `completion` fields (not flat text) |
| User names | Real names from Telegram (`Name: message`), no anonymous `<user_N>` tokens |
| Response filtering | Min 3 chars, max 200 chars, no pure emoji, deduplicated |
| Rap ratio | 5% (down from 20%) |
| Epochs | 3 |
| Effective batch size | 8 (1 x 8 grad_accum) |
| Learning rate | 2e-4 (cosine schedule, higher for LoRA) |
| Warmup steps | 50 |
| Optimizer | AdamW (torch, fp32 compatible) |
| Weight decay | 0.01 |
| Max seq length | 2,048 |
| NEFTune | alpha=5 |
| Chat template | Custom override suppressing `<think>` block injection |
| Eval strategy | Every epoch |
| Early stopping | load_best_model_at_end=True, metric=eval_loss |
| Hardware | Kaggle T4 |
| Precision | fp32 (T4 does not support bf16; model is natively bf16) |

### Training (v7)

| Parameter | Value |
|---|---|
| Method | Full fine-tuning (all 596M params, no LoRA) |
| Epochs | 3 (+ early stopping) |
| Effective batch size | 8 (1 x 8 grad_accum) |
| Learning rate | 2e-5 (cosine schedule) |
| Warmup steps | 100 |
| Optimizer | AdamW 8-bit |
| Weight decay | 0.05 |
| Max seq length | 2,048 |
| NEFTune | alpha=10 |
| Eval strategy | Every epoch (steps = STEPS_PER_EPOCH) |
| Save strategy | Every epoch (steps = STEPS_PER_EPOCH), save_total_limit=2 |
| Early stopping | load_best_model_at_end=True, metric=eval_loss, greater_is_better=False |
| Hardware | Kaggle T4 |
| Precision | fp16 mixed precision |

### Training (v6)

| Parameter | Value |
|---|---|
| Method | Full fine-tuning (all 596M params, no LoRA) |
| Epochs | 6 |
| Effective batch size | 8 (1 x 8 grad_accum) |
| Learning rate | 5e-5 (cosine schedule) |
| Warmup steps | 100 |
| Optimizer | AdamW 8-bit |
| Weight decay | 0.01 |
| Max seq length | 2,048 |
| NEFTune | alpha=5 |
| Eval strategy | Every epoch |
| Hardware | Kaggle T4 |
| Precision | fp16 mixed precision |

### Training (v5)

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Effective batch size | 8 (1 x 8 grad_accum) |
| Learning rate | 2e-4 (cosine schedule) |
| Warmup steps | 50 |
| Optimizer | AdamW 8-bit |
| Weight decay | 0.01 |
| Max seq length | 2,048 |
| DoRA | Enabled |
| NEFTune | alpha=5 |
| Eval strategy | Every epoch |
| Hardware | Kaggle T4 |
| Precision | fp16 (no quantization during training) |

### v4 Loss Curve

| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 873 | 0.8730 | 0.8294 |
| 1746 | 0.5224 | 0.4981 |
| 2619 | 0.5118 | 0.4429 |
| 3492 | 0.3888 | 0.4353 |
| 4365 | 0.3303 | 0.4404 |

Val loss plateaus at epoch 3 (0.443) then rises -- overfitting starts at epoch 4. This motivated reducing to 3 epochs for v5.

### v3 Loss Curve

| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 609 | 0.6595 | 0.6257 |
| 1218 | 0.2133 | 0.2200 |
| 1827 | 0.1196 | 0.1442 |
| 2436 | 0.0912 | 0.1368 |

---

## Inference

### Sampling Parameters

| Parameter | Value |
|---|---|
| `do_sample` | `True` |
| `temperature` | 0.95 |
| `top_p` | 0.9 |
| `top_k` | 50 |
| `repetition_penalty` | 1.1 |
| `max_new_tokens` | 256 |

### Post-processing

```python
import re
text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
text = re.sub(r"^<user_\d+>\s*", "", text.strip())
text = re.sub(r"^<bot>\s*", "", text.strip())
```

Strips Qwen3 `<think>` blocks and any leaked speaker tokens from generation start.

---

## Evaluation Framework (NEW in v12)

`eval_loss` is a poor proxy for chat-model quality — a model can memorize training data and score low eval_loss while still feeling bad. The framework adds an objective signal via **pairwise blind A/B labeling** + automatic regression checks.

### Architecture

- **Generation lives on Kaggle**: at end of `training.ipynb`, model is already in memory → run `chat()` on the full golden pool → save `eval/generations_v<N>.json`. Pure batch, no UI, **background-mode safe** (Kaggle "Save & Run All"). Output also pushed to the model HF repo so the bench notebook can pull it without a Kaggle download.
- **Labeling lives locally**: `businessgpt_bench.ipynb` opens in Jupyter on Mac (or Colab), loads two `generations_v<N>.json` files, shows ipywidgets blind A/B UI, writes decisions to `eval/ratings_<A>_vs_<B>.json` incrementally (crash/restart safe). Already-rated prompts are skipped on re-open.

### Files

```
eval/
├── golden_prompts.json           # ~631 prompts (581 chat from val + 30 rap-trigger + 10 fact + 10 edge), 20 marked held_out
├── _seed_golden.py               # Idempotent regenerator
├── generations_v<N>.json         # Per-version outputs on the full pool
└── ratings_<A>_vs_<B>.json       # Pairwise decisions (A=v_a, B=v_b, "tie", or skipped)
```

### Categories in `golden_prompts.json`

- **chat** (~581) — bulk-imported from `val.jsonl`. Real Telegram contexts. Tests in-context response quality.
- **rap_trigger** (~30) — synthetic `{name}: {trigger} {artist}` across all 25 artists. Tests v12's new trigger format directly.
- **fact** (~10) — simple factual questions. Catches Chinese leakage and refusal regressions.
- **edge** (~10) — single-word, empty-context, provocations. Tests robustness.
- **`held_out: true`** flag on 20 random prompts — never to be used for DPO pair extraction (see Future Work). Reserves a clean regression-eval slice once DPO starts.

### Bench notebook functions (cells 19-23)

| Function | Output |
|---|---|
| `load_golden()` / `load_generations(version)` | Pool / version outputs (disk first, HF fallback with caching) |
| `run_eval(version)` | Local generation against currently-loaded backend (rare; usually Kaggle does this) |
| `guardrails_table(*versions)` | Table: `gay_spam_pct`, `chinese_pct`, `refusal_pct`, length stats, vocab entropy. Comparison columns side-by-side. |
| `pairwise_ui(a, b, session_size=30, category=None)` | ipywidgets UI: blind A/B side-by-side panels, **6 buttons** (1/2/★super1/★super2/tie/skip), progress counter, sampled from unrated subset, optional category filter. Super-tier marks exceptional responses for higher-weight DPO pairs. |
| `summarize_ratings(a, b)` | Win-rate + bootstrap 95% CI over 1000 resamples + per-category breakdown (Markdown table) + super-wins per category. |

### Workflow per training cycle

1. Train v<N> on Kaggle (background OK).
2. Eval cell at end of `training.ipynb` writes `eval/generations_v<N>.json`, pushes to HF.
3. Locally: `load_generations("v<N>")` (auto-downloads from HF if not on disk).
4. Run `guardrails_table("v<N-1>", "v<N>")` — verify no regression on auto checks.
5. Run `pairwise_ui("v<N-1>", "v<N>", session_size=30)` — label a session. Repeat across multiple sessions to expand coverage.
6. Run `summarize_ratings("v<N-1>", "v<N>")` — see win-rate + CI. If CI excludes 50% with v_new winning → ship.

### Validated by smoke test

Synthetic generations with 50% gay-spam and 10% Chinese in v_dirty, 0% in v_clean: guardrails reports the expected percentages exactly. Blind shuffle + winner unmapping, simulated at 70% v_new win-rate over 30 prompts, recovers ~67% empirically with bootstrap CI bracketing the truth.

### DPO (preference learning on top of SFT)

`ratings_<A>_vs_<B>.json` IS a DPO training set — every non-tie comparison is a `(prompt, chosen, rejected)` triple. The eval infra produces preference pairs as a **side effect** of normal version comparison.

- **Threshold**: ~500 pairs minimum, ~1000 ideal. With 30-prompt sessions, ~17 sessions across multiple version pairs.
- **Why DPO over PPO/RLHF**: no reward model needed; `trl.DPOTrainer` drops in on the existing SFT LoRA pipeline; fits on the same T4×2.
- **Risks**:
  - **Style collapse** — DPO often makes models "safer" / more generic. We want the opposite. Mitigated by `beta=0.3` (high) to stay close to SFT distribution.
  - **Prompt contamination** — prompts in DPO pairs leak when evaluating on the same pool. The `held_out: true` slice in `golden_prompts.json` is reserved for clean post-DPO eval.
- **Implementation**: see `eval/build_preference_pairs.py` (converter) and `dpo.ipynb` (Kaggle DPO trainer). Pipeline section below has the full flow.

---

## Iteration Pipeline (concrete commands)

End-to-end for one iteration cycle. Models live on HF; data lives Kaggle/local.

### A. SFT iteration (`training.ipynb` → `vXofi/businessgpt-v<N>-qwen3.5-2b`)

1. **Edit cells in `training.ipynb`:**
   - Cell `64987c61` (title): describe what changed in v<N>
   - Cell `184b2693`: bump `SAVE_DIR = "businessgpt_v<N>_2b_model"`
   - Cell `36ba67fd`: bump `HF_REPO = "vXofi/businessgpt-v<N>-qwen3.5-2b"` and commit message
   - Modify whichever data cell encodes the actual change (e.g. `23f7aac6` for rap pipeline, `e5b0da45` for filters, `e0921e1a` for SFT augment loading)
2. **Refresh `eval/sft_augment.jsonl`** (only if new ratings were added): `python3 eval/build_sft_augment.py --upload-kaggle` — versions the businessgpt-eval Kaggle dataset.
3. **Push to Kaggle**: Save Version → "Save & Run All (Background)". Attach datasets: `alextech123/businessraw`, `wadzim/modern-russian-rap`, `<you>/businessgpt-eval`.
4. **Wait** ~8-10h on T4×2.
5. **Outputs (automatic)**:
   - SFT adapter pushed to `vXofi/businessgpt-v<N>-qwen3.5-2b`
   - `eval/generations_v<N>.json` (single, default-temp) AND `eval/generations_v<N>_multi.json` (4 candidates, for plan C DPO data) written to `/kaggle/working/eval/` AND pushed to the same HF repo

### B. Eval (`businessgpt_bench.ipynb`, locally on Mac)

1. **Pull generations** — automatic on first call, cached afterward:
   ```python
   load_generations("v<N>")    # auto-fetches from vXofi/businessgpt-v<N>-qwen3.5-2b
   ```
2. **Automatic guardrails**:
   ```python
   guardrails_table("v<N-1>", "v<N>")
   ```
3. **Pairwise labeling** (30-min session):
   ```python
   pairwise_ui("v<N-1>", "v<N>", session_size=30)
   # 6 buttons: 1/2/★super1/★super2/tie/skip — writes eval/ratings_v<N-1>_vs_v<N>.json
   ```
4. **Win-rate readout**:
   ```python
   summarize_ratings("v<N-1>", "v<N>")   # bootstrap 95% CI + per-category + super-wins
   ```
5. **Optional GGUF for prod**: `python3 merge_and_push.py` (script auto-detects v<N> via `SOURCE_REPO` constant — bump it). With imatrix calibration on by default. Output: `vXofi/businessgpt-v<N>-qwen3.5-2b-gguf`.

### C. DPO iteration (`dpo.ipynb` → `vXofi/businessgpt-v<N>-dpo-qwen3.5-2b`)

Run when total preference pairs across all `ratings_*.json` exceeds ~500.

1. **Build preference pairs locally**:
   ```bash
   python3 eval/build_preference_pairs.py
   # writes eval/preference_pairs.jsonl, dedupe + ties skipped + tier metadata preserved
   ```
2. **Upload to Kaggle dataset** (one-time setup, then re-version):
   - First time: `cd eval && kaggle datasets init -p . && <edit metadata id>` then `kaggle datasets create -p .`
   - Subsequent: `python3 eval/build_preference_pairs.py --upload-kaggle`  (or `kaggle datasets version -p eval -m "..."`)
3. **Edit `dpo.ipynb` cell `dpo-config`**:
   - `BASE_VERSION = "v<N>"`
   - `DPO_VERSION  = "v<N>-dpo"`
   - `SUPER_WEIGHT = 2` (replicates ★ super pairs in train; 1 = no weighting)
4. **Push to Kaggle**: Save Version → "Save & Run All (Background)". Attach `<you>/businessgpt-eval` dataset.
5. **Wait** ~3-5h on T4×2 (faster than SFT — fewer steps with smaller dataset, smaller LoRA r=16).
6. **Outputs**:
   - DPO adapter pushed to `vXofi/businessgpt-v<N>-dpo-qwen3.5-2b`
   - `eval/generations_v<N>-dpo.json` pushed to same HF repo

### D. Eval the DPO model

Same as B but with `pairwise_ui("v<N>", "v<N>-dpo", session_size=30)`. Use the **held-out** subset of golden_prompts (where `held_out: true`) for clean signal — those prompts were never in DPO training.

For category-filtered labeling on a specific concern:
```python
pairwise_ui("v<N>", "v<N>-dpo", session_size=20, category="chat")
```

### File map for this pipeline

```
businessgpt_retrain/
├── training.ipynb               # A. SFT (Kaggle)
├── dpo.ipynb                    # C. DPO (Kaggle)
├── businessgpt_bench.ipynb      # B + D. Local eval & labeling
├── merge_and_push.py            # B optional. Local GGUF merge + push
└── eval/
    ├── golden_prompts.json      # Fixed eval pool (Kaggle dataset input)
    ├── preference_pairs.jsonl   # DPO training data (built locally, uploaded to Kaggle)
    ├── _seed_golden.py          # Regenerate golden_prompts from val
    ├── build_preference_pairs.py # Build preference_pairs from all ratings_*.json
    ├── generations_v<N>.json    # Per-version model outputs (cached from HF)
    ├── ratings_*.json           # Pairwise labels (local-only, not uploaded)
    └── dataset-metadata.json    # Kaggle dataset config (created via `kaggle datasets init`)
```

### One-time setup (per machine)

1. `pip install kaggle huggingface_hub`
2. Place `~/.kaggle/kaggle.json` with API token
3. `huggingface-cli login`
4. Initialize `eval/` as a Kaggle dataset:
   ```bash
   cd eval
   kaggle datasets init -p .
   # edit dataset-metadata.json: set "id" to "<your-username>/businessgpt-eval", title to anything
   kaggle datasets create -p .
   cd ..
   ```
5. Attach the resulting dataset to both `training.ipynb` and `dpo.ipynb` Kaggle pages.

---

## Known Issues

### Resolved

1. **No diversity**: `do_sample=True` required -- HuggingFace default is greedy
2. **Multi-turn runaway**: Fixed with anonymous tokens + alternating user/assistant roles
3. **Bot data poisoning**: Strict filtering of bot messages, mentions, replies to bot, `/generate` commands
4. **Speaker token leakage**: Model sometimes generates `<user_X>` prefix -- stripped via regex in post-processing
5. **Unsloth tokenizer quirk**: Returns processor, not raw tokenizer -- use `tokenizer.tokenizer` (VL models only; not needed for v5 text model)
6. **HF_HUB_ENABLE_HF_TRANSFER**: Missing in newer huggingface_hub -- monkeypatched in finetune.ipynb
7. **transformers version**: Must pin to 4.57.1 on Kaggle to avoid `additional_chat_templates` 404
8. **HF push from Kaggle**: Keeps failing -- workaround: download LoRA adapter, merge & push locally via `merge_and_push.py`. v6 uses `HfApi.upload_folder()` directly (no merge needed).
9. **Chinese text in output**: Qwen3-VL-2B had strong multilingual pretraining leaking through. Switched to 0.6B abliterated model in v5.
10. **4-bit quantization quality**: QLoRA 4-bit on 2B model produced near-random outputs in bench. v5 trains in full fp16, quantizes only at export (Q8_0).
11. **Post-bot data contamination**: v5 splits data at BusinessGPT introduction date, uses only clean pre-bot data as primary training set.
12. **v5 format mismatch**: Role-swapping training format didn't match flat bench/inference format, causing "random slop" outputs. Fixed in v6 with simplified next-message prediction format.
13. **Qwen3 `<think>` block injection**: `apply_chat_template` inserts empty `<think>\n\n</think>` blocks. v6 uses manual formatting to avoid this.

14. **Full-sequence loss**: v5-v7 trained on all tokens (system + context + response). Loss was dominated by context reproduction. v8 uses `completion_only_loss=True` to train only on response tokens.
15. **Anonymous tokens losing context**: `<user_N>` tokens carried no semantic information. v8 uses real Telegram names.
16. **Overfitting from full fine-tuning**: 596M trainable params on 7.5k examples caused memorization. v8 uses LoRA (~10-15M trainable params).
17. **GGUF export for Qwen3.5**: llama-cpp-python doesn't support Qwen3.5 yet, but the C++ project does. `merge_and_push.py` rewritten to auto-clone/build llama.cpp, install only `gguf` (not the full `requirements-convert_hf_to_gguf.txt`, which downgrades transformers below Qwen3.5 support).
18. **`torchvision::nms` op missing on local merge**: torch 2.6 + torchvision 0.25 binary mismatch. peft → transformers → image_transforms triggers torchvision import. Fix: `pip uninstall torchvision -y` (script doesn't use it).
19. **Gay-test bot spam in training data**: ~3% of v11 train targets were `🏳️‍🌈 I am X% gay!` from a Telegram game bot, model started spamming the pattern. Filtered in v12 via regex at record level.
20. **Rap not triggered in chat**: v8/v11 only saw `[Artist]\n<lines>` format, so the model never learned to start a verse from a chat prompt. v12 mixes in chat-style triggers (`{name}: зачитай {artist}`).
21. **DoRA in fp16 unstable on bf16-native base**: gradient/magnitude issues observed; v11/v12 keep `use_dora=False`. Earlier v3/v5 worked because Unsloth patched the magnitude norm.
22. **Synthetic chat-trigger over-generalisation** (v12 → v13): introducing `{name}: {trigger} {artist}` → 8-line lyric as 33% of rap data + generic trigger templates (`{artist}?`, `давай {artist}`) taught the model "any short `{name}: <text>` prompt = produce lyrics". Detected via guardrails: fact/edge median length jumped 16→305 / 19→329 in v12. Fixed in v13: format C 33%→10%, 13→8 verb-anchored triggers only, ratio 15%→10%.
23. **Eval framework retroactively justified data-only iteration**: v12 → v13 is the first transition driven by quantitative regression evidence (the per-category length table) rather than vibes. Validates the eval framework as a real signal source.

### Open

1. **No quantitative quality signal**: previous versions relied on eyeballing bench outputs. **Resolved in v12** via the eval framework (see below) — pairwise blind A/B + bootstrap CI + automatic guardrails.
2. **Qwen3.5 VL model loading edge case**: `AutoModelForCausalLM` works for v11/v12 but fallback to `AutoModel` retained for safety.
3. **fp8 inference quality**: v11 in fp8 is noticeably dumber than fp16/Q8 — the 2B size is at the edge of what survives heavy quantization. Future bigger-model bump (3-4B) might be more robust but needs different hardware.

---

## File Inventory

```
businessgpt_retrain/
├── PLAN.md                      # This report
├── training.ipynb               # Main training notebook v11/v12 (Kaggle GPU, Qwen3.5-2B)
├── finetune.ipynb               # Older training notebook (v5–v8, Qwen3-0.6B / 0.8B)
├── preprocess.ipynb             # Standalone preprocess (v5 era; v11+ embeds preprocess in training.ipynb)
├── businessgpt_bench.ipynb      # Benchmark + eval framework (Colab/local)
├── finetune_local.ipynb         # Local M3 Pro training version
├── merge_and_push.py            # Download v_N from HF, auto-detect LoRA vs full, merge, GGUF convert, push
├── eval/                        # Evaluation framework (NEW in v12)
│   ├── golden_prompts.json      # Fixed prompt pool (~631 prompts: 581 chat + 30 rap + 10 fact + 10 edge)
│   ├── _seed_golden.py          # One-shot script to regenerate golden_prompts from val.jsonl
│   ├── generations_v<N>.json    # Per-version model outputs on the pool (written by training.ipynb)
│   └── ratings_<A>_vs_<B>.json  # Pairwise blind A/B decisions (written by bench notebook UI)
├── train.jsonl, val.jsonl       # Old preprocess.ipynb output (still used to seed golden_prompts)
├── result.json                  # Cached Telegram export
├── lora_adapter/                # Older LoRA adapter copies
├── merged_model_v11/            # v11 merged safetensors + GGUF artifacts (local)
├── merged_model_v9/             # v9 merged (local)
├── llama.cpp/                   # Built locally for GGUF conversion
├── outputs_local/               # Local training checkpoints
└── .kaggle/kaggle.json          # Kaggle API credentials
```

### Notebook Cell Map (training.ipynb) — v12

- Cell `cbf9f0df`: pip install (vllm, transformers from git)
- Cell `7d4a3f55`: pip install (huggingface_hub, trl, kagglehub, datasets, accelerate, bitsandbytes, peft)
- Cell `294fd9e0`: HF login via Kaggle secrets
- Cell `33d86fb6`: Load `result.json` (Kaggle input or `kagglehub`)
- Cell `e5b0da45`: Extract / flatten / date-split + `standard_filter` (NEW: gay-spam regex) + `enhanced_filter`
- Cell `28d34ff8`: Merge consecutive (60s window)
- Cell `3a2d45d5`: Session split (1h gap, min 5 msgs, pre-bot only)
- Cell `855c997f`: `SYSTEM_PROMPT` + `create_multiturn_examples()` (window sizes 10, 15)
- Cell `23f7aac6`: Rap loading + `chunk_lyrics()` (3-format mix: A/B/C, ratio 15%)
- Cell `1276f8b7`: Model loading (Qwen3.5-2B-abliterated, fp16, multi-GPU)
- Cell `5b609885`: LoRA config (r=32, alpha=64, all-linear, no DoRA)
- Cell `b183634d`: Token filter, dedupe, train/val split
- Cell `e80175de`: `tokenize_with_labels()` — completion-only via manual masking on last `<\|im_start\|>assistant\n`
- Cell `ceede2a2`: Export `val_examples.json` for bench
- Cell `515930a1`: Trainer config + dry-run loss check
- Cell `32ff76d7`: `trainer.train()`
- Cell `184b2693`: Save (`SAVE_DIR = businessgpt_v12_2b_model`)
- Cell `416b5b95`–`3e3cd615`: Smoke-test inference with `chat()`
- Cell `36ba67fd`: HF push to `vXofi/businessgpt-v12-qwen3.5-2b`
- Cell `bad16d15` / `911feeac`: **Eval generation** — load `golden_prompts.json`, run `chat()` on full pool, save `eval/generations_v12.json`, push to HF (NEW in v12)

### Notebook Cell Map (finetune.ipynb) — v7

- Cell 2: pip install (huggingface_hub, unsloth, kagglehub, datasets)
- Cell 7: Extraction, filtering, date-split (pre/post bot)
- Cell 8: Merge consecutive + anonymous tokens (both halves)
- Cell 9: Session split (both halves)
- Cell 10: `SYSTEM_PROMPT` + `create_examples()` (flat context → next message), post-bot mixing
- Cell 11: Rap lyrics loading, artist filtering, `split_lines()`, `chunk_lyrics()`, capped at 20%
- Cell 13: Model loading (`FastLanguageModel`, Qwen3-0.6B abliterated, fp16)
- Cell 15: Full fine-tuning setup (enable gradients on all params, no LoRA)
- Cell 16: `format_chat()` (manual formatting, no `<think>` injection) + `count_tokens()` + combine (chat + rap), train/val split
- Cell 17: `format_to_text()` via `format_chat()` → HF Dataset
- Cell 18: Export `val_examples.json` for bench notebook
- Cell 20: SFTTrainer config (3 epochs, lr=2e-5, full fine-tuning, early stopping, save per epoch)
- Cell 23: Save model (`model.save_pretrained()`)
- Cell 33: Push to hub via `HfApi.upload_folder()`

### Notebook Cell Map (businessgpt_bench.ipynb) — v12

- Cell 1: pip install (transformers, ipywidgets — needed for eval UI)
- Cell 2: Backend selection (`transformers` or `gguf`), `MODEL_ID` (v11/v12)
- Cell 3: Model loading via `AutoModelForCausalLM` (or `Llama` for GGUF)
- Cell 4: `chat()` with `_format_chat()` (manual, no `<think>`), `_strip_tokens()`
- Cells 6-7: 10 benchmark scenarios + diversity check
- Cell 10: Validation examples from training data (`val_examples.json`)
- Cells 12-14: Parameter Grid Comparison (10 configs × 3 scenarios × 3 gens) + stats
- Cell 16: Interactive chat (`/clear`, `/context`, `/exit`)
- Cell 18 (header): **## Evaluation Framework**
- Cell 19: `load_golden()`, `load_generations(version)` — disk first, HF fallback with caching (NEW)
- Cell 20: `run_eval(version)` — local variant of the Kaggle eval cell (NEW)
- Cell 21: `guardrails(version)`, `guardrails_table(*versions)` — gay/CJK/refusal/length/entropy regex checks (NEW)
- Cell 22: `pairwise_ui(a, b, session_size=30, category=None)` — ipywidgets blind A/B with crash-safe writes (NEW)
- Cell 23: `summarize_ratings(a, b)` — win-rate + bootstrap 95% CI + per-category breakdown (NEW)

---

## Training History

### v1 (discarded)
- 3 epochs, lr=2e-4, no DoRA/NEFTune, no anonymous tokens
- Severe overfitting (train 0.05, val 3.14). Outputs too vanilla, leaked sender names.

### v2
- 4 epochs, lr=1.5e-4, dropout=0.05, stripped sender names entirely
- Better style but occasional Chinese text, leaked bot commands

### v3
- Anonymous speaker tokens (`<user_N>`), DoRA, NEFTune, lora_dropout=0
- Strict bot filtering (messages, mentions, replies)
- Validation loss re-introduced (5% split)
- Train: 0.0912, Val: 0.1368 -- healthy

### v4
- Added Russian rap lyrics dataset (26 artists, ~30% of training data)
- New unified system prompt
- Sentence-boundary splitting for no-newline lyrics
- Val loss plateau at epoch 3, overfitting epochs 4-5
- Bench: Chinese text leaking, random word hallucinations, some coherent outputs at low temp

### v13 (latest)
- **Same hyperparams as v11/v12** — only rap data pipeline changed.
- **Trigger over-generalisation fix** (from v12 guardrails: model rapping on fact/edge prompts):
  - Format C weight 33% → 10% (A=40 / B=50 / C=10)
  - 13 → 8 trigger templates, kept only verb-anchored
  - `TARGET_RATIO` 0.15 → 0.10
- **Length cap fix** (from manual labeling feedback):
  - `window` 8 → 5 in `chunk_lyrics()`. A/B split 2/3, C=full 5 lines. Caps assistant output at 5 lines (was 8) — labelers consistently flagged 8-line responses as «перебор».
- **Newline normalization** (from manual labeling feedback):
  - 50% chance to join lyric lines with `". "` instead of `"\n"`. v12 used `"\n"` everywhere in rap data, which leaked into chat replies (model started using newlines where punctuation belongs).
- **Eval-framework addition**: `pairwise_ui` got `★ супер 1/2` buttons that record `tier: "super"` on exceptional responses. `summarize_ratings` reports super-wins per category. These super pairs become higher-weight training data when DPO kicks in.
- **Expected outcomes** (to verify with `guardrails_table` after v13 trains):
  - chat category unchanged
  - fact/edge median length back near v11 (~16-19 chars)
  - rap_trigger stays at ~280+ when explicitly asked (verb-anchored format C)
  - max response length ≤ 5 lines instead of 8

### v12
- **Same training setup as v11** — model architecture, LoRA config, hyperparams unchanged. Only data changed.
- **Gay-spam filter** worked exactly as designed: guardrails confirmed **7.45% → 0.16%** on the v11/v12 eval pool (47/631 → 1/631).
- **Rap ratio 5% → 15%** + **chat-style triggers**: `chunk_lyrics()` emits 3 formats per window (A=33% `[Artist]` continuation, B=33% chat-trigger+hint, C=33% trigger-only). 13 templates × 9 names → wide trigger surface.
- **Measured outcome (per-category response length, v11 vs v12):**

  | category | v11 (med) | v12 (med) | verdict |
  |---|---|---|---|
  | chat | 18 | 21 | unchanged ✓ |
  | rap_trigger | 23 | **287** | works as intended ✓ |
  | fact | 16 | **305** | **regression** — model raps on factual questions |
  | edge | 19 | **329** | **regression** — model raps on single-word prompts |

  Format C plus generic trigger templates (`{artist}?`, `давай {artist}`, etc.) collapsed into "any short `{name}: <text>` → 8 lines of lyrics". Driven the v13 narrowing.

- **Rationale for staying data-only**: v11 was the first model that answered in-context. Cranking hyperparams without objective signal risks regressing that. Both v12 and v13 only touch data so quality deltas are attributable.

### v11
- **First model to actually work**: answers in-context, doesn't spam memorized training strings.
- **Base model**: `huihui-ai/Huihui-Qwen3.5-2B-abliterated` (1.9B params, hybrid linear+full attention, native bf16).
- **LoRA without DoRA**: r=32, alpha=64, all-linear, dropout=0.05. DoRA disabled because magnitude norm is unstable in fp16 on a bf16-native base (Unsloth used to patch this for earlier versions).
- **Conservative training**: 2 epochs, lr=5e-5 (lower than v8's 2e-4), max_seq=4096.
- **Multi-GPU**: T4×2 with `device_map="auto"`. Model is split across GPUs.
- **fp8 quality dip**: Q8_0 GGUF stays good, fp8 is noticeably dumber — at the edge of what 2B can survive under heavy quantization.
- **Open issues that motivated v12**: gay-spam (~3% of targets), rap never triggered from chat prompts.

### v8 (formerly latest)
- **New base model**: `huihui-ai/Huihui-Qwen3.5-0.8B-abliterated` (Qwen3.5 0.8B, hybrid linear+full attention)
- **Completion-only loss**: Loss computed ONLY on assistant response tokens (was full-sequence in v5-v7). This is the critical fix -- previous versions trained on context reproduction, not response generation.
- **Real user names**: Replaced anonymous `<user_N>` tokens with actual Telegram names (`Name: message` format). Provides semantic grounding.
- **LoRA instead of full fine-tune**: DoRA r=32, all-linear targets. Reduces overfitting (v7 had train 0.19 vs val 0.59 gap).
- **Response quality filtering**: Min 3 chars, max 200 chars, no pure emoji, deduplicated identical responses.
- **Rap ratio reduced**: 20% → 5%. Chat-focused training.
- **FP32 on T4**: Qwen3.5 is natively bf16; T4 doesn't support bf16, so cast to fp32.
- **`<think>` suppression**: Custom chat template (no `<think>` injection) + `suppress_tokens` in generation.
- **Rationale**: v7 quality was poor because (a) full-sequence loss trained the model to reproduce context, not respond to it, (b) 596M trainable params on 7.5k examples caused severe memorization, (c) anonymous tokens provided no semantic information.

### v7
- **Hyperparameter fixes**: Val loss was flat at epoch 2-3 in v6, so reduced to 3 epochs with early stopping
- **Early stopping**: `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `greater_is_better=False`; checkpoints saved every epoch (`save_strategy="steps"`, `save_steps=STEPS_PER_EPOCH`), `save_total_limit=2`
- **Lower LR**: 5e-5 → 2e-5 for full fine-tuning (less aggressive with all 596M params)
- **Stronger regularisation**: weight_decay 0.01 → 0.05, NEFTune alpha 5 → 10
- **Rationale**: v6 ran 3 epochs past the optimal point; val loss curve and v4 history both confirm epoch 2-3 is the sweet spot. Lower LR + higher weight decay/NEFTune to resist memorisation on 7.5k examples.

### v6
- **Full fine-tuning**: All 596M params trainable (no LoRA/DoRA), eliminates Unsloth merge bugs
- **Simplified data format**: Flat context (one "user" message) → plain text response (one "assistant" message). No role-swapping gimmick. Matches inference format exactly.
- **Manual chat formatting**: Bypasses Qwen3's `apply_chat_template` to avoid `<think>` block injection
- **Training params**: 6 epochs, lr=5e-5 (cosine), AdamW, NEFTune alpha=5
- **Songs at 20%**: Reduced from 30% to give more weight to chat data
- **Rationale**: v5 outputs were "random slop" due to format mismatch between training (multi-turn role-swapped) and inference (flat context). Full fine-tuning with lower LR should produce more coherent outputs than LoRA.

### v5
- **New base model**: `huihui-ai/Huihui-Qwen3-0.6B-abliterated-v2` (0.6B, abliterated)
- **No quantization during training**: Full fp16 (model is small enough for T4)
- **Export**: Both full 16-bit safetensors and Q8_0 GGUF
- **Date-split data**: Only clean pre-bot chat data as primary set, 15% post-bot mix with enhanced scrubbing
- **Training params**: 3 epochs (based on v4 overfitting analysis), lr=2e-4, LoRA r=32
- **Rationale**: 0.6B abliterated model should reduce Chinese text leakage and remove safety refusals. Full fp16 training eliminates 4-bit quantization artifacts. Date split removes bot-contaminated training data.
