"""Client utilities for interacting with a local OLLAMA server."""
from __future__ import annotations

from typing import List, Dict, Optional

try:  # pragma: no cover - import is trivial
    import requests  # type: ignore
except Exception:  # pragma: no cover - handled in generate
    requests = None  # type: ignore


class OllamaClient:
    """A small wrapper around the OLLAMA HTTP API.

    Parameters
    ----------
    base_url:
        URL where the OLLAMA server is accessible. Defaults to the standard
        local installation URL.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        options: Optional[Dict] = None,
        timeout: int = 60,
    ) -> str:
        """Generate a completion from a model using the /api/generate endpoint.

        Parameters
        ----------
        model:
            The name of the model to query.
        prompt:
            Prompt text to send to the model.
        stream:
            Whether to use streaming responses. Streaming is disabled by default
            because this client collects the full response before returning.
        options:
            Additional options passed directly to the API.
        timeout:
            Timeout for the HTTP request in seconds.
        """

        url = f"{self.base_url}/api/generate"
        payload: Dict = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        if options:
            payload["options"] = options

        if requests is None:  # pragma: no cover - network library missing
            raise ImportError("The 'requests' package is required to call the OLLAMA API")

        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
