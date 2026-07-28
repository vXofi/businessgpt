# Responsible Use

BusinessGPT is an experimental Russian informal group-chat chatbot. It was
designed for a closed, allowlisted Telegram deployment and for research into
small-data domain adaptation, human evaluation, and CPU inference.

## Intended Use

- private, access-controlled informal chat experiments;
- research on persona adaptation and conversational response selection;
- offline evaluation of LoRA, prompt, quantization, and reranking variants;
- local or private-server inference with human oversight.

## Not Intended For

- unrestricted public chatbot access;
- factual, medical, legal, financial, or safety-critical advice;
- content moderation or classification;
- use by minors;
- automated decisions about people;
- impersonation, harassment, or targeted abuse.

## Content Risk

The target persona intentionally permits strong profanity and provocative
humor. The base model is an abliterated variant with reduced refusal behavior.
Outputs can therefore contain offensive, discriminatory, sexual, biased,
factually incorrect, or otherwise unsafe material. Training for a particular
chat style does not constitute safety alignment.

The production deployment is kept behind authentication and allowlisting.
Consumers integrating the public model artifacts must add controls appropriate
to their users and jurisdiction.

## Privacy Boundary

Training and evaluation use private chat-derived data. Raw exports, prompts,
responses, ratings, and generated training artifacts are excluded from git and
are not published with the model. Public reports contain only reviewed
aggregate statistics.

The current public experiment manifest stores neutral prompt descriptions and
SHA-256 hashes so completed generations can retain provenance without keeping
explicit persona instructions in the main project surface. Historical git
revisions may retain earlier prompt text; prompt secrecy is therefore not a
security control. Deployment safety relies on authentication, allowlisting,
and operational controls instead.

No claim is made that memorization risk has been eliminated. Distinctive phrase
filters, held-out temporal evaluation, private-artifact scanning, and
access-controlled deployment reduce risk but do not provide a formal privacy
guarantee.

## Evaluation Boundary

The completed public evaluation covers text behavior only and uses one primary
reviewer. It does not establish vision quality, inter-rater agreement,
concurrency capacity, a production service-level objective, or general safety.
See [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) for measured results and
limitations.
