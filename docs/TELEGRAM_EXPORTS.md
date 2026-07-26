# Telegram HTML Imports

Telegram Desktop can export a chat as `messages.html`, `messages2.html`, and
later numbered pages instead of `result.json`. The model-evaluation importer
normalizes this format without external HTML-parser dependencies.

Keep the source export, normalized JSON, anchor configuration, and reports
private. The recommended destination is `eval_runs/imports/`, which is ignored
by Git.

## Import

```bash
python -m eval.model_eval import-telegram \
  --export /private/telegram-export \
  --output eval_runs/imports/chat.json \
  --anchors eval_runs/imports/chat.anchors.json \
  --anchor-config eval_runs/imports/chat.config.json
```

The importer:

- discovers and naturally orders all `messages*.html` pages;
- carries joined-message senders across page boundaries;
- preserves message IDs, timestamps, replies, rich text, polls, and local media
  paths;
- distinguishes image attachments from video, audio, stickers, and files;
- inserts a placeholder when Telegram omitted the attachment;
- redacts common API keys, bearer tokens, HF tokens, and long hexadecimal
  secrets before writing output;
- fingerprints the source HTML pages for reproducibility.

The normalized JSON uses a Telegram-like `messages` list and can be passed back
to `python -m eval.model_eval build`. The builder can also read the HTML export
directory directly.

## Private Anchors

Release and deployment history is often less clean than model version numbers.
Use an ignored JSON config to record known phrases and bot aliases:

```json
{
  "versions": ["v1", "v2"],
  "expected_version_sequence": ["v1", "v2"],
  "skipped_versions": [],
  "version_boundaries": {
    "v2_confirmed": "a phrase that confirms v2 was active"
  },
  "custom_events": {
    "rollback": "a phrase that confirms rollback"
  },
  "old_bot_handles": ["old_bot", "old_bot_alias"],
  "new_bot_handle": "new_bot"
}
```

The anchor report separates:

- ranked version mentions and release candidates;
- exact configured boundary phrases;
- custom deployment events such as rollback or endpoint changes;
- first/last old and new handle mentions;
- legacy-handle mentions after the first new-handle event.

These are review aids. A release announcement does not automatically become a
dataset cutoff, especially if inference later rolled back.

## Build Evaluation Data

From normalized JSON:

```bash
python -m eval.model_eval build \
  --export eval_runs/imports/chat.json \
  --media-root /private/telegram-export \
  --after 2026-01-01T00:00:00+03:00 \
  --output eval_runs/datasets/temporal_eval.json \
  --audit eval_runs/results/temporal_eval.audit.json
```

From the original HTML directory:

```bash
python -m eval.model_eval build \
  --export /private/telegram-export \
  --media-root /private/telegram-export \
  --after 2026-01-01T00:00:00+03:00
```

Use the normalized form when the anchor report is part of the provenance you
want to retain locally. Use direct HTML input for a quick rebuild.
