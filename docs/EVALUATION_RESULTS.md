# BusinessGPT Text Evaluation

Status: v16 text-only baseline completed in July 2026.

This report contains aggregate, public-safe results. Evaluation prompts,
responses, ratings, and source chat exports remain private.

## Scope

The evaluation answers four questions:

1. Is the adapted v16 inference artifact preferred to the unadapted base
   artifact in their intended no-thinking modes?
2. Did v16 improve over the previous v15 checkpoint?
3. Does the current production system prompt improve output?
4. What is the absolute quality of the deployed v16 Q5 configuration?

The primary out-of-time set contains 64 text contexts from 30 source sessions.
A later post-prompt set contains 43 contexts from 20 sessions. Contexts were
sampled after the training-data cutoff and were kept out of model training.

Matched comparisons use identical source contexts and deterministic seeds.
The v15/v16 and prompt comparisons also match their sampling configuration.
The base comparison is an artifact-level adaptation comparison, not a
template-identical weight-only ablation: the base uses its native tokenizer
and official `enable_thinking=False` template, while v16 uses its trained
artifact template. Both are evaluated in their intended no-thinking inference
modes with the same 256-token sampling profile.

Sides were blinded during review. Confidence intervals use source-session
cluster bootstrap resampling where applicable, so repeated contexts from one
conversation are not treated as fully independent. The current public manifest
redacts exact operational prompts from the main project surface and records
their descriptions and SHA-256 hashes instead.

The study used one primary human reviewer. Preference scores count a win as
1, a tie as 0.5, and a loss as 0.

## Pairwise Results

| Comparison | Rated | Result | Session-cluster 95% CI | Interpretation |
| --- | ---: | ---: | ---: | --- |
| v16 artifact vs unadapted base artifact | 40 | v16 preference 80.0% | 67.9%-88.9% | Clear artifact-level preference |
| v15 vs v16 | 64 | v15 preference 61.7% | 49.3%-73.3% | Directional v15 advantage; interval includes parity |
| Production vs legacy prompt | 43 | production preference 55.8% | 45.0%-65.3% | Directional prompt gain; interval includes parity |

In the base comparison, v16 won 28 prompts, the base won 4, 7 were ties where
both were bad, and 1 was a tie where both were good. Among non-ties, v16 won
87.5%. Median completion length fell from 54.5 tokens for the base to 11 for
v16. The adapted artifact is clearly preferred in this deployment-oriented
comparison, but the design does not isolate model weights from template
effects.

The v15 comparison is an important negative result. v15 won 32 prompts and v16
won 17, with 15 ties. v16 often expanded a conversational opening into an
explanation, invented continuation, formatted insertion, or answer to the
whole context. Its relative strength was handling contexts with multiple
topics. The result does not support claiming that v16 is an across-the-board
upgrade over v15.

The production prompt won 22 prompts and the legacy prompt won 17, with 4
ties. Both had a 13-token median response. The directional gain therefore is
not explained by globally shorter output. The production prompt is retained,
but the measured evidence does not establish a conclusive win.

## Absolute Production Quality

The deployed v16 Q5 configuration was rated on all 43 post-prompt contexts.

The review UI offered ordinal `good`, `acceptable`, and `bad` choices, but a
separate formal rubric was not pre-registered. For interpretation:

- `good` means the reviewer judged the response relevant, coherent, and
  successful in the target chat style;
- `acceptable` means it remained a valid contextual reply despite noticeable
  style, specificity, or wording weaknesses;
- `bad` means it was not a satisfactory reply because of irrelevance,
  incoherence, wrong-thread selection, or a comparable failure.

For compact reporting, `good + acceptable` is called `reviewer-usable`. It is
an ordinal judgment by one reviewer, not a production acceptance rate.

| Rating | Count | Rate |
| --- | ---: | ---: |
| Good | 22 | 51.2% |
| Acceptable | 16 | 37.2% |
| Reviewer-usable: good or acceptable | 38 | 88.4% |
| Bad | 5 | 11.6% |

The session-cluster 95% confidence interval is 39.5%-60.5% for good and
78.1%-97.6% for reviewer-usable. The Wilson interval for bad is 5.1%-24.5%.

| Context category | Examples | Good | Reviewer-usable |
| --- | ---: | ---: | ---: |
| Normal chat | 15 | 66.7% | 93.3% |
| Reply-heavy | 12 | 25.0% | 83.3% |
| Fluency stress | 8 | 50.0% | 100.0% |
| Long post | 3 | 66.7% | 66.7% |
| Multiple topics | 3 | 33.3% | 66.7% |
| Factual edge case | 2 | 100.0% | 100.0% |

The small category counts are diagnostic slices, not precise population
estimates. The weakest larger slice is reply-heavy context.

Response length did not separate quality: median completion lengths were 21
tokens for good, 23 for acceptable, and 18 for bad. This argues against a
universal output-length cap as the next intervention.

Post-hoc inspection grouped the five bad outputs into two unrelated answers,
one selection of an old thread after a long post, one incoherent wording
failure, and one weakly grounded answer. These groups were assigned after
review and are treated as diagnostic hypotheses rather than pre-registered
labels.

## Generation And Runtime

The absolute audit completed 43 of 43 generations with no empty responses,
length-truncated responses, known name-prefix artifacts, repeated four-grams,
or high-similarity copies of prior bot messages.

| Metric | Result |
| --- | ---: |
| Median completion length | 22 tokens |
| 95th-percentile completion length | 76 tokens |
| Median wall latency | 18.3 s |
| 95th-percentile wall latency | 55.0 s |
| Maximum wall latency | 134.7 s |

During the initial run, an inference slot degraded after eight successful
requests. Subsequent deterministic requests reached the 300-second backend
timeout. Restarting only the model API container restored the same prompt IDs
and seeds to successful 14-47 second generations. Failed rows were retried and
were not counted as model-quality failures. This remains an ML systems
recovery issue; the current evidence is not a concurrency or service-level
benchmark.

## Conclusions

- The adapted v16 inference artifact was clearly preferred to the unadapted
  base artifact in their intended no-thinking configurations.
- v16 is not demonstrated to be better than v15 overall.
- The production prompt is directionally preferred but does not resolve
  wrong-thread selection and unrelated continuations.
- The deployed text configuration was judged good or acceptable on 88.4% of
  the audited contexts by the primary reviewer.
- The next training hypothesis is to improve last-relevant-turn selection and
  suppress unrelated continuation while preserving concise conversational
  behavior and v16's relative strength on multiple-topic contexts.

Any v17 promotion decision must use a newer or separately reserved holdout.
The exact baseline prompts must not become training examples.

## Limitations And Deferred Work

- One primary reviewer; inter-rater agreement was not measured.
- No controlled long multi-turn conversation benchmark.
- No vision-quality audit.
- No Q4 versus Q5 quality comparison.
- No completed reward-model best-of-N evaluation.
- No full repetition-penalty sweep.
- No load, concurrency, or automated recovery benchmark.

These are separate follow-up studies. They are not included in the completed
v16 text-baseline claims above.
