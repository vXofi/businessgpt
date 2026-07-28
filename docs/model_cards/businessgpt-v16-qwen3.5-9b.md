---
language:
- ru
license: apache-2.0
library_name: peft
pipeline_tag: text-generation
base_model: huihui-ai/Huihui-Qwen3.5-9B-abliterated
tags:
- peft
- lora
- conversational
- qwen3.5
---

# BusinessGPT v16 Qwen3.5 9B LoRA

BusinessGPT v16 is a Russian informal group-chat adaptation of
[`huihui-ai/Huihui-Qwen3.5-9B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-9B-abliterated).
This repository contains the PEFT LoRA adapter, not standalone model weights.

## Objective And Intended Use

The model explores small-data domain adaptation for short, slang-heavy
Telegram-style conversation. It is intended for research and private,
allowlisted informal-chat deployments with human oversight.

It is not intended as a general assistant, factual authority, moderation model,
or safety-critical system.

## Training Summary

- base: Qwen3.5 9B abliterated;
- method: PEFT LoRA, rank 32, alpha 64, dropout 0.05, all-linear targets;
- objective: completion-only causal language modeling;
- schedule: one epoch, effective batch size 16, learning rate `5e-5`;
- data: private chat-derived conversations, a capped rap-style subset, and a
  reviewed/distilled v16 augmentation;
- stale earlier model-output augmentation disabled for v16.

The raw training corpus and generated augmentation are private and are not
included in this repository.

## Evaluation

| Comparison | Result | 95% session-cluster CI |
| --- | ---: | ---: |
| v16 artifact vs unadapted base artifact, 40 contexts | v16 preference 80.0% | 67.9%-88.9% |
| v15 vs v16, 64 contexts | v15 preference 61.7% | 49.3%-73.3% |
| production vs legacy prompt, 43 contexts | production preference 55.8% | 45.0%-65.3% |

On a separate 43-context absolute audit, one primary reviewer judged 51.2% of
responses `good` and another 37.2% `acceptable`: 88.4% were therefore
`reviewer-usable`. This is an ordinal single-reviewer result, not a production
acceptance rate.

The base comparison is artifact-level rather than a template-identical
weight-only ablation: base and v16 use their intended native no-thinking
templates. v15 is directionally preferred to v16, so v16 is not presented as
an across-the-board quality upgrade.

Full methodology, definitions, confidence intervals, and limitations:
[public evaluation report](https://github.com/vXofi/businessgpt/blob/main/docs/EVALUATION_RESULTS.md).

## Privacy

Raw conversations, evaluation contexts, model responses, and human ratings are
not public. Public reports contain aggregate statistics only. The current
public experiment manifest represents operational prompts with neutral
descriptions and SHA-256 hashes; prompt secrecy is not a security boundary.

## Risks And Limitations

- The target persona permits strong profanity and provocative humor.
- The abliterated base has reduced refusal behavior.
- Outputs can be offensive, discriminatory, sexual, biased, factually wrong,
  or otherwise unsafe.
- Evaluation used one primary reviewer; inter-rater agreement is unavailable.
- v15 was directionally preferred to v16.
- Vision quality, long multi-turn stability, Q4/Q5 quality, and reward-model
  reranking have not been formally evaluated.
- Memorization risk is reduced through filtering and private evaluation, not
  eliminated.

Do not expose this model through an unrestricted public endpoint. Add
authentication, allowlisting, rate limits, and content controls appropriate to
the deployment.

## Links

- [Source and training/evaluation code](https://github.com/vXofi/businessgpt)
- [Responsible-use statement](https://github.com/vXofi/businessgpt/blob/main/docs/RESPONSIBLE_USE.md)
- [GGUF repository](https://huggingface.co/vXofi/businessgpt-v16-qwen3.5-9b-gguf)
