import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ollama_client import OllamaClient


def test_generate_posts_to_correct_endpoint() -> None:
    client = OllamaClient(base_url="http://localhost:11434")
    dummy_requests = SimpleNamespace(post=MagicMock())
    dummy_response = dummy_requests.post.return_value
    dummy_response.json.return_value = {"response": "hi"}
    dummy_response.raise_for_status.return_value = None

    with patch("ollama_client.requests", dummy_requests):
        result = client.generate("llama2", "Hello", stream=False)

    dummy_requests.post.assert_called_once()
    args, kwargs = dummy_requests.post.call_args
    assert args[0] == "http://localhost:11434/api/generate"
    assert kwargs["json"]["model"] == "llama2"
    assert kwargs["json"]["prompt"] == "Hello"
    assert result == "hi"
