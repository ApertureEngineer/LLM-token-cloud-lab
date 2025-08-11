"""Utilities for token frequency analysis of model responses."""
from __future__ import annotations

from collections import Counter
from typing import Dict

from token_analysis import TokenAnalyzer
from ollama_client import OllamaClient


def analyze_text_tokens(text: str, analyzer: TokenAnalyzer | None = None) -> Dict[int, int]:
    """Return a frequency map of token ids for ``text``.

    Parameters
    ----------
    text:
        The text to analyse.
    analyzer:
        Optional :class:`TokenAnalyzer` instance. If omitted a default
        analyzer is created.
    """
    analyzer = analyzer or TokenAnalyzer()
    tokens = analyzer.encode(text)
    return Counter(tokens)


def analyze_model_tokens(
    model: str,
    prompt: str,
    client: OllamaClient | None = None,
    analyzer: TokenAnalyzer | None = None,
) -> Dict[int, int]:
    """Generate a model response and return token frequency counts.

    This function queries a model via :class:`OllamaClient` and analyses the
    token ids of the returned text. The resulting frequency map can later be
    used to build visualisations such as token clouds.

    Parameters
    ----------
    model:
        Name of the model to query.
    prompt:
        Text prompt provided to the model.
    client:
        Optional OLLAMA client. If omitted a new one is created.
    analyzer:
        Optional token analyzer. If omitted a default one is used.
    """
    client = client or OllamaClient()
    analyzer = analyzer or TokenAnalyzer()
    response = client.generate(model, prompt, stream=False)
    return analyze_text_tokens(response, analyzer)
