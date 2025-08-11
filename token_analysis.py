"""Tokenization helpers based on the ``tiktoken`` library.

This module provides a small wrapper around :mod:`tiktoken` to inspect the
relationship between text and token indices. It can be useful for analysing an
LLM's token landscape without requiring access to the full model.
"""
from __future__ import annotations

from typing import Iterable, List

try:  # pragma: no cover - import is trivial
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    tiktoken = None  # type: ignore


class TokenAnalyzer:
    """Analyze tokens for a given encoding.

    If :mod:`tiktoken` is not installed a very small whitespace based tokenizer
    is used as a fallback so that the module remains functional for testing and
    experimentation.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        if tiktoken is not None:
            try:
                self.encoding = tiktoken.get_encoding(encoding_name)
                self._vocab: dict[str, int] | None = None
            except Exception:  # pragma: no cover - fallback if encoding download fails
                self.encoding = None
                self._vocab = {}
                self._next_index = 0
        else:  # pragma: no cover - minimal fallback for missing dependency
            self.encoding = None
            self._vocab = {}
            self._next_index = 0

    def encode(self, text: str) -> List[int]:
        """Return the list of token ids representing ``text``."""
        if self.encoding is None:  # fallback tokenizer
            tokens = text.split()
            for tok in tokens:
                if tok not in self._vocab:  # type: ignore[operator]
                    self._vocab[tok] = self._next_index  # type: ignore[index]
                    self._next_index += 1
            return [self._vocab[tok] for tok in tokens]  # type: ignore[index]
        return self.encoding.encode(text)

    def decode(self, tokens: Iterable[int]) -> str:
        """Return the text representation of ``tokens``."""
        if self.encoding is None:  # fallback tokenizer
            reverse = {v: k for k, v in self._vocab.items()}  # type: ignore[union-attr]
            return " ".join(reverse[t] for t in tokens)
        return self.encoding.decode(list(tokens))

    def token_count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        return len(self.encode(text))

    def token_index(self, token: str) -> int:
        """Return the token id for ``token``.

        Parameters
        ----------
        token:
            Text that should correspond to a single token. If the provided text
            maps to more than one token an error is raised.
        """
        if self.encoding is None:  # fallback tokenizer
            return self.encode(token)[0]

        token_ids = self.encode(token)
        if len(token_ids) != 1:
            raise ValueError("Provided text does not map to a single token")
        return token_ids[0]

    def vocabulary_size(self) -> int:
        """Return the size of the vocabulary."""
        if self.encoding is None:  # pragma: no cover - simple in fallback mode
            return len(self._vocab)
        return self.encoding.n_vocab
