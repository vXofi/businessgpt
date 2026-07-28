# BusinessGPT Roadmap

Forward-looking plan. Retrospective notes live in `PLAN.md`; detailed command
reference lives in `SCRIPT_GUIDE.md`.

## Current Snapshot

As of July 2026, the active baseline is v16 SFT:

- v16 SFT was trained, exported to GGUF, and deployed through the separate
  `../hugeballs-server` repo.
- The production shape is a 9B Qwen3.5 abliterated LoRA merged/exported to GGUF,
  with Q5_K_M as the practical serving quant.
- The v16 text baseline is frozen and documented in
  `docs/EVALUATION_RESULTS.md`. Fine-tuning clearly adapts the base model, the
  current production configuration is usable on 88.4% of the absolute audit,
  and v15 remains directionally preferred to v16.
- The production system prompt is directionally preferred to the legacy prompt,
  but the measured interval includes parity. It is retained without claiming a
  conclusive win.
- API/deployment code no longer lives here. This repo owns training, eval,
  distillation, preference data, reward model tooling, and GGUF export.
- The recent runtime repetition issue appears mostly fixed by moving the bot
  integration toward structured chat formatting instead of feeding a flat
  bot-heavy transcript. Keep monitoring, but do not treat it as an active
  training blocker unless it recurs.
- ORPO was attempted and should not be shipped right now; validation/test output
  was not good enough.
- The RuBERT reward model trained successfully and is the next available quality
  lever.

## Active Backlog

### 1. Define And Evaluate v17

Use the frozen v16 results to target the measured failure modes:

- improve last-relevant-turn selection;
- suppress unrelated continuation and assistant-like formatted insertions;
- improve wording coherence;
- preserve concise conversational behavior and v16's multiple-topic strength.

Do not train on the exact baseline prompts. Build a fresh targeted training
augment and reserve a newer, session-disjoint temporal holdout for the promotion
decision. Keep v16 unless v17 improves the target failures without a material
aggregate or category regression.

### 2. Add A Multi-Turn Chat Benchmark

The completed baseline evaluates fixed contexts. Add controlled 8-15 turn
conversations to measure:

- repetition of prior bot answers;
- return to an old thread;
- loss of participant identity;
- coherence and wording degradation;
- response to topic changes.

Keep runtime input role-aware. If repetition returns in production, validate
the captured request with `eval/validate_runtime_context.py` before changing
training data.

### 3. Evaluate Reward-Model Best-Of-N

Use existing v16 multi-candidate generations if available; otherwise generate a
small fresh set first.

```bash
python3 eval/rank_with_rm.py \
  --version v16 \
  --rm-repo vXofi/businessgpt-reward-rubert
```

Then compare RM-selected outputs against default outputs in
`notebooks/local/businessgpt_bench.ipynb`.

Gate for moving RM into production:

- RM best-of-N wins clearly on real failure-style prompts;
- no obvious preference for bland/long/safe answers;
- no increased private-phrase memorization.

If it passes, integrate best-of-N in `../hugeballs-server`, not this repo.

### 4. Complete ML Systems Validation

Run this in `../hugeballs-server`:

- latency, throughput, memory, and timeout rate at concurrency 1, 2, and 4;
- API-to-inference timeout and overload behavior;
- health, readiness, restart, and recovery after terminating the inference
  process;
- an end-to-end Telegram-to-model contract test with structured messages.

The existing single-request measurements are descriptive and must not be
presented as service-level or concurrency evidence.

### 5. Keep Optional Studies Separate

Vision quality, Q4 versus Q5 quality, and a full repetition-penalty sweep remain
useful follow-up studies. They do not block the completed text baseline and
should be run only when their result supports a concrete product decision.

### 6. Keep ORPO Parked

ORPO remains a research branch, not a production path. Revisit only after we
have enough fresh v16 preference pairs and a narrow reason to believe ORPO will
solve something RM reranking cannot.

## Done / Parked

| Item | Status | Notes |
| --- | --- | --- |
| API repo split | Done | Deployment lives in `../hugeballs-server`. |
| v16 distillation | Done enough for v16 SFT | DeepSeek V4 Pro prompt/flow became the accepted source path. |
| v16 SFT | Done | Current deployed baseline. |
| GGUF export | Done | `merge_and_push.py` defaults to v16 9B Q5/Q4 exports. |
| Runtime formatting fix | Probably fixed | Repetition reportedly happened only once after the change. |
| v16 text baseline | Done | Public-safe aggregate report is in `docs/EVALUATION_RESULTS.md`. |
| ORPO | Parked | Attempted, but output was not acceptable. |
| Reward model | Done | Use it for offline best-of-N validation next. |

## Known Risks

- Private generations/eval artifacts must not be pushed to HF model repos.
- 9B capacity can memorize distinctive low-count phrases. Known example family:
  "Зелёный диплом".
- The `I am N% gay` artifact remains a known regression family and is filtered.
- Rap data improves style but can hijack generic prompts if overrepresented.
- One-on-one bot-heavy runtime chats are distribution-shifted from the original
  group-chat training data.

## Setup

```bash
git clone https://github.com/vXofi/businessgpt
cd businessgpt
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.template .env
```

Set at least:

```text
OPENROUTER_API_KEY=...
HF_TOKEN=...
```

Private data sources:

```bash
kaggle datasets download alextech123/businessraw -p . --unzip
kaggle datasets download avxofi/businessgpt-eval -p eval/ --unzip
```

## Current Command Pointers

Distillation and data conversion commands are in `SCRIPT_GUIDE.md`.

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

## Repo Layout

```text
businessgpt/
├── SCRIPT_GUIDE.md          # local script command reference
├── ROADMAP.md               # current project state and backlog
├── REPO_NOTES.md            # compact repo mental model
├── PLAN.md                  # retrospective notes through older versions
├── merge_and_push.py        # HF LoRA/full model -> GGUF export
├── notebooks/
│   ├── training.ipynb       # SFT training notebook
│   ├── eval_only.ipynb      # candidate generation notebook
│   ├── reward_model.ipynb   # RuBERT reward model training
│   ├── orpo.ipynb           # parked ORPO research notebook
│   ├── local/               # local review/labeling notebooks
│   └── archive/             # legacy notebooks
└── eval/
    ├── distill_responses.py
    ├── build_sft_augment.py
    ├── build_preference_pairs.py
    ├── build_multi_preference_pairs.py
    ├── rank_with_rm.py
    ├── validate_runtime_context.py
    ├── purge_hf_private_eval.py
    └── scan_bot_patterns.py
```

Server/API/deployment repo:

```text
../hugeballs-server
```
