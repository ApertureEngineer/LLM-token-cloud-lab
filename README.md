# LLM-Analyzer

Utilities for experimenting with locally hosted LLMs via the [OLLAMA](https://github.com/jmorganca/ollama) API and exploring tokenisation behaviour.

## Features

* **Ollama client** – Lightweight wrapper around the OLLAMA HTTP API.
* **Model conversations** – Script for making two models talk to each other.
* **Token analysis** – Inspect the token indices and counts for arbitrary text using the `tiktoken` library.

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Examples

### Make two models converse

```bash
python conversation.py "Hello there" --model-a llama2 --model-b mistral --turns 4
```

### Inspect tokens

```python
from token_analysis import TokenAnalyzer

analyzer = TokenAnalyzer()
print(analyzer.encode("Hello world"))
print(analyzer.token_index("hello"))
```

The repository also contains unit tests which can be executed with:

```bash
pytest
```
