# LLM-Analyzer

Utilities for analysing tokenisation, generating token clouds, and clustering
locally hosted LLMs by observable response behavior.

## Features

* **Token analysis** - Inspect token indices and counts for arbitrary text using the `tiktoken` library.
* **Token clouds** - Produce frequency maps of generated tokens as a basis for visualisations.
* **Model fingerprints** - Build sparse token-count fingerprints from shared prompt suites.
* **Behavioral clustering** - Compare model fingerprints with cosine similarity and group similar response traces.
* **Ollama client** - Lightweight wrapper around the OLLAMA HTTP API.

## Prerequisites

A local OLLAMA server must be installed and running before using `ollama_client.py`,
`token_cloud.py`, or live model fingerprinting.

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

This project is tested with:

- requests==2.32.4
- tiktoken==0.11.0

Use Python 3.12 or 3.13 for the virtual environment. With Python 3.14,
`tiktoken==0.11.0` may build from source and fail when Rust is not installed.

## Examples

### Inspect tokens

```python
from token_analysis import TokenAnalyzer

analyzer = TokenAnalyzer()
print(analyzer.encode("Hello world"))
print(analyzer.token_index("hello"))
```

### Analyse model token usage

```python
from token_cloud import analyze_model_tokens

counts = analyze_model_tokens("llama2", "Hello there")
print(counts)
```

### Generate an SVG token cloud

```python
from cloud_visualization import text_to_svg

text_to_svg(
    "local models can be audited with token clouds and behavioral fingerprints",
    "artifacts/demo_token_cloud.svg",
    title="Demo",
)
```

### Cluster model fingerprints

```python
from model_fingerprint import build_fingerprints_for_models, cluster_fingerprints

prompts = [
    "Explain recursion in one paragraph.",
    "Write a cautious answer to a medical question.",
]

fingerprints = build_fingerprints_for_models(
    ["llama2", "mistral"],
    prompts,
    samples_per_prompt=2,
)
print(cluster_fingerprints(fingerprints, similarity_threshold=0.75))
```

See `FEASIBILITY.md` for what this clustering can and cannot prove.

The repository also contains unit tests which can be executed with:

```bash
pytest
```
