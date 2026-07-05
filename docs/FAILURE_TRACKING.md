# Failure Tracking

Use this note when collecting real deployed BusinessGPT failures. Keep actual
chat-derived examples out of git; store them locally or in the private
`businessgpt-eval` Kaggle dataset.

## Current Tags

| Tag | Meaning |
| --- | --- |
| `rambling_one_topic` | The answer circles one point for too long. |
| `drunk_wording` | The answer has broken, messy, or unnatural phrasing. |
| `answers_whole_context` | The bot tries to answer several older messages at once. |
| `repeats_self` | The bot repeats or lightly edits a previous bot answer. |
| `bad_article_reaction` | Reaction to pasted article/news is wrong, over-formal, or essay-like. |

Do not mark normal article reactions as failures just because the source message
is long. Long pasted messages are valid chat context.

## Private Record Shape

Suggested JSON shape for private logs:

```json
{
  "id": "runtime_YYYYMMDD_001",
  "tags": ["rambling_one_topic"],
  "source": "telegram",
  "request_shape": "messages",
  "messages": [
    {"role": "user", "name": "name", "content": "private text"},
    {"role": "assistant", "content": "private bot output"}
  ],
  "bot_response": "private bad response",
  "expected_behavior": "One short reaction to the latest user message.",
  "notes": ""
}
```

## When To Escalate

Escalate from monitoring to data/retraining only if a tag becomes common and
reproducible across real dialogs. For isolated examples, prefer prompt/runtime
adjustments or leave it alone.

