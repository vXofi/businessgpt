# BusinessGPT Roadmap

Forward-looking plan. Retrospective notes live in `PLAN.md`; detailed command
reference lives in `SCRIPT_GUIDE.md`.

## Current Snapshot

As of July 2026, the active baseline is v16 SFT:

- v16 SFT was trained, exported to GGUF, and deployed through the separate
  `../hugeballs-server` repo.
- The production shape is a 9B Qwen3.5 abliterated LoRA merged/exported to GGUF,
  with Q5_K_M as the practical serving quant.
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

### 1. Verify Runtime Input Format

Confirm the Telegram bot sends either the structured `messages` request or an
equivalent role-aware format to the server:

```json
{
  "messages": [
    {"role": "user", "name": "xofi", "content": "text"},
    {"role": "assistant", "content": "previous bot answer"},
    {"role": "user", "name": "xofi", "content": "next text"}
  ],
  "max_tokens": 128,
  "temperature": 0.9,
  "repetition_penalty": 1.2
}
```

If repetition returns, use `eval/validate_runtime_context.py` against captured
dialogs before changing training data.

### 2. Build A Small Failure Set

Collect real deployed failures before spending Kaggle time:

- repetition / echoing old bot messages;
- dead short answers;
- observer/reviewer behavior instead of chat participant behavior;
- memorized private phrases;
- unwanted artifact phrases such as the old gay-spam family;
- any new one-off pattern that appears more than once.

Store private examples outside git, preferably in the `businessgpt-eval` Kaggle
dataset. Use `docs/FAILURE_TRACKING.md` for tags and the private record shape.

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

### 4. Decide Whether To Retrain

Do not start another Kaggle training run just because a few bugs exist. Retrain
only if the failure set shows a pattern that inference formatting and RM
reranking do not fix.

Likely retrain inputs:

- fresh chosen-only SFT augment from reviewed v16 outputs;
- targeted distillation for failure categories;
- stricter filtering for memorized/private phrases;
- no stale old manual generations unless they pass current review.

### 5. Keep ORPO Parked

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
