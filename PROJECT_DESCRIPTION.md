# Project Description

The **LLM-Analyzer** project contains utilities for working with locally hosted
large language models via the [Ollama](https://ollama.com/) API. It explores how
text maps to token ids and provides building blocks for analysing token usage,
building token-cloud data, and clustering models by observable response
behavior.

Current components include:

* a minimal HTTP client for the Ollama API,
* tools for inspecting token indices and vocabulary size,
* helpers to build token frequency maps for future word-cloud visualisations,
* model fingerprinting from shared prompt suites,
* cosine-similarity clustering for generated response traces.

The repository is intended as a starting point for experimenting with open
source models and auditing their behavioral token landscape. The clustering
layer compares generated outputs; it does not inspect model weights or claim
direct visibility into model internals.
