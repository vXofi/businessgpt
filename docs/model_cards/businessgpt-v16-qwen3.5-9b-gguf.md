---
language:
- ru
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
base_model: vXofi/businessgpt-v16-qwen3.5-9b
tags:
- gguf
- llama.cpp
- conversational
- qwen3.5
---

# BusinessGPT v16 Qwen3.5 9B GGUF

This repository contains merged GGUF exports of the
[BusinessGPT v16 LoRA](https://huggingface.co/vXofi/businessgpt-v16-qwen3.5-9b)
and its Qwen3.5 9B abliterated base for local or CPU inference with
`llama.cpp`.

## Files And Evaluation Status

- `Q5_K_M` is the deployed and evaluated production quant.
- `Q4_K_M` is provided as a smaller alternative but has not completed a
  controlled quality comparison against Q5.

The Q5 configuration was evaluated on 43 private out-of-time text contexts.
One primary reviewer judged 51.2% of responses `good` and 37.2%
`acceptable`, for 88.4% `reviewer-usable`. This is an ordinal single-reviewer
result, not a production acceptance rate.

Additional controlled results:

| Comparison | Result | 95% session-cluster CI |
| --- | ---: | ---: |
| v16 artifact vs unadapted base artifact, 40 contexts | v16 preference 80.0% | 67.9%-88.9% |
| v15 vs v16, 64 contexts | v15 preference 61.7% | 49.3%-73.3% |
| production vs legacy prompt, 43 contexts | production preference 55.8% | 45.0%-65.3% |

See the [public evaluation report](https://github.com/vXofi/businessgpt/blob/main/docs/EVALUATION_RESULTS.md)
for methodology, label definitions, experiment caveats, and limitations.

## Intended Use

The model is intended for research and private, allowlisted Russian informal
group-chat deployments. It is not a general assistant, factual authority,
moderation model, or safety-critical system.

## Risks

The persona permits strong profanity and provocative humor, and the base model
is abliterated. Outputs can be offensive, discriminatory, sexual, biased,
factually incorrect, or otherwise unsafe. Do not expose the model through an
unrestricted public endpoint.

Raw training conversations and evaluation artifacts are private. Memorization
risk is reduced through filtering and access-controlled deployment, not
eliminated. Vision quality, Q4/Q5 equivalence, long multi-turn stability, and
inter-rater agreement have not been established.

Read the complete
[responsible-use statement](https://github.com/vXofi/businessgpt/blob/main/docs/RESPONSIBLE_USE.md)
before deployment.

## Links

- [Adapter model card](https://huggingface.co/vXofi/businessgpt-v16-qwen3.5-9b)
- [Source and evaluation code](https://github.com/vXofi/businessgpt)
