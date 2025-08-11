import sys
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parents[1]))

from token_analysis import TokenAnalyzer
from token_cloud import analyze_text_tokens, analyze_model_tokens


class DummyClient:
    def generate(self, model: str, prompt: str, stream: bool = False) -> str:
        assert model == "dummy"
        assert prompt == "hi"
        return "hello world"


def test_analyze_text_tokens() -> None:
    analyzer = TokenAnalyzer()
    text = "hello world hello"
    counts = analyze_text_tokens(text, analyzer)
    expected = Counter(analyzer.encode(text))
    assert counts == expected


def test_analyze_model_tokens() -> None:
    analyzer = TokenAnalyzer()
    counts = analyze_model_tokens("dummy", "hi", client=DummyClient(), analyzer=analyzer)
    expected = Counter(analyzer.encode("hello world"))
    assert counts == expected
