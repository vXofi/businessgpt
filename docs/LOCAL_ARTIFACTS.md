# Local Artifacts

This repo intentionally accumulates local generated files while training,
exporting, and evaluating models. They are ignored by git, but can consume a lot
of disk.

Observed examples from this workspace:

| Path | Why it exists | Current size seen |
| --- | --- | ---: |
| `merged_model_v16_9b/` | Local merged HF model and GGUF export outputs. | ~45 GB |
| `llama.cpp/` | Cloned/built by `merge_and_push.py`. | ~241 MB |
| `businessgpt-api.tar.gz` | Old API image archive from before deployment split. | ~295 MB |
| `eval/` private JSON files | Local/Kaggle eval artifacts and ratings. | ~33 MB |
| `train.jsonl` | Exported training data. | ~12 MB |
| `__pycache__/` | Python bytecode cache. | small |

Safe cleanup candidates when not actively exporting/evaluating:

```bash
rm -rf __pycache__
rm -rf llama.cpp
rm -rf merged_model_v16_9b
rm -f businessgpt-api.tar.gz
```

Only delete `eval/*.json`, `eval/*.jsonl`, `train.jsonl`, or
`val_examples.json` after confirming those artifacts are backed up in Kaggle or
can be regenerated.

Deployment artifacts now belong in `../hugeballs-server`; avoid keeping new API
image archives in this training repo.

