import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from token_analysis import TokenAnalyzer


def test_roundtrip() -> None:
    analyzer = TokenAnalyzer()
    text = "Hello world"
    tokens = analyzer.encode(text)
    assert analyzer.decode(tokens) == text
    assert analyzer.token_count(text) == len(tokens)


def test_token_index_single_token() -> None:
    analyzer = TokenAnalyzer()
    token_id = analyzer.token_index("hello")
    assert isinstance(token_id, int)
    assert token_id >= 0
