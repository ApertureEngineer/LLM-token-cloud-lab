# LLM-Analyzer

Utilities for analysing tokenisation and generating token clouds from locally hosted LLMs.

## Features

* **Token analysis** – Inspect the token indices and counts for arbitrary text using the `tiktoken` library.
* **Token clouds** – Produce frequency maps of generated tokens as a basis for visualisations.
* **Ollama client** – Lightweight wrapper around the OLLAMA HTTP API.

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

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

The repository also contains unit tests which can be executed with:

```bash
pytest
```
