# Feasibility: LLM Token Clouds and Clustering

## Verdict

This repo can achieve LLM clustering if clustering means comparing observable
model behavior from generated text traces.

It cannot directly cluster model internals without access to model weights,
activations, architecture metadata, or each model's native tokenizer.

## Achievable Now

The practical path is:

1. Run the same prompt suite against multiple local models.
2. Collect one or more responses per prompt.
3. Tokenize the generated responses.
4. Build one sparse token-count fingerprint per model.
5. Compare fingerprints with vector similarity.
6. Cluster models whose fingerprints are similar.
7. Render top tokens as token-cloud data.

This is now represented by `model_fingerprint.py`.

## What The Clusters Mean

The clusters describe behavioral similarity under the selected prompt suite.
For example, two models may cluster together because they:

- produce similar vocabulary,
- repeat similar phrases,
- use similar response lengths,
- share stylistic habits,
- respond similarly to the tested task types.

These clusters are useful for auditing and comparing models, but they should not
be treated as proof that the models have similar training data or internal
representations.

## Important Limitations

- The current tokenizer defaults to `tiktoken`'s `cl100k_base`, which is not the
  native tokenizer for many Ollama-hosted models.
- Token-frequency clustering is sensitive to the prompt suite.
- Small sample sizes can produce unstable clusters.
- Token counts capture surface behavior better than reasoning quality.
- Semantic clustering will need embeddings or evaluator scores in a later stage.

## Recommended Direction

The repo should evolve into an LLM audit lab:

- token fingerprints,
- response statistics,
- prompt-suite management,
- pairwise model comparison,
- behavioral clustering,
- token-cloud exports for visualization,
- optional semantic embeddings for deeper clustering.

The token-cloud idea is viable as a visualization and fingerprinting layer. The
engineering goal should be behavioral auditability rather than claiming direct
visibility into model internals.
