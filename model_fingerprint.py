"""Model fingerprinting and clustering from generated token traces."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Mapping, Sequence

from ollama_client import OllamaClient
from token_analysis import TokenAnalyzer


TokenCounts = Mapping[int, int]


@dataclass(frozen=True)
class ModelFingerprint:
    """A compact behavioral fingerprint for one model."""

    model: str
    token_counts: Counter[int]
    responses: tuple[str, ...]
    prompt_count: int
    sample_count: int

    @property
    def total_tokens(self) -> int:
        """Return the total number of observed tokens."""
        return sum(self.token_counts.values())

    @property
    def unique_tokens(self) -> int:
        """Return the number of distinct observed token ids."""
        return len(self.token_counts)

    @property
    def token_diversity(self) -> float:
        """Return unique tokens divided by total tokens."""
        if self.total_tokens == 0:
            return 0.0
        return self.unique_tokens / self.total_tokens


def build_model_fingerprint(
    model: str,
    prompts: Sequence[str],
    client: OllamaClient | None = None,
    analyzer: TokenAnalyzer | None = None,
    samples_per_prompt: int = 1,
) -> ModelFingerprint:
    """Generate responses for prompts and return a token-count fingerprint.

    The fingerprint is behavioral: it describes the text a model produced under
    a controlled prompt set. It does not inspect model weights or internals.
    """
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be at least 1")

    client = client or OllamaClient()
    analyzer = analyzer or TokenAnalyzer()

    counts: Counter[int] = Counter()
    responses: list[str] = []

    for prompt in prompts:
        for _ in range(samples_per_prompt):
            response = client.generate(model, prompt, stream=False)
            responses.append(response)
            counts.update(analyzer.encode(response))

    return ModelFingerprint(
        model=model,
        token_counts=counts,
        responses=tuple(responses),
        prompt_count=len(prompts),
        sample_count=len(responses),
    )


def cosine_similarity(left: TokenCounts, right: TokenCounts) -> float:
    """Return cosine similarity between sparse token-count vectors."""
    if not left or not right:
        return 0.0

    shared_tokens = set(left).intersection(right)
    dot_product = sum(left[token] * right[token] for token in shared_tokens)
    left_norm = sqrt(sum(count * count for count in left.values()))
    right_norm = sqrt(sum(count * count for count in right.values()))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def similarity_matrix(fingerprints: Sequence[ModelFingerprint]) -> Dict[str, Dict[str, float]]:
    """Return pairwise cosine similarities for model fingerprints."""
    matrix: Dict[str, Dict[str, float]] = {}
    for left in fingerprints:
        matrix[left.model] = {}
        for right in fingerprints:
            matrix[left.model][right.model] = cosine_similarity(
                left.token_counts,
                right.token_counts,
            )
    return matrix


def cluster_fingerprints(
    fingerprints: Sequence[ModelFingerprint],
    similarity_threshold: float = 0.75,
) -> List[List[str]]:
    """Cluster model names by token fingerprint similarity.

    This uses connected components over a similarity graph. Two models are
    linked when their cosine similarity is greater than or equal to the
    threshold.
    """
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")

    names = [fingerprint.model for fingerprint in fingerprints]
    by_name = {fingerprint.model: fingerprint for fingerprint in fingerprints}
    neighbors = {name: set() for name in names}

    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            similarity = cosine_similarity(
                by_name[left_name].token_counts,
                by_name[right_name].token_counts,
            )
            if similarity >= similarity_threshold:
                neighbors[left_name].add(right_name)
                neighbors[right_name].add(left_name)

    clusters: list[list[str]] = []
    visited: set[str] = set()

    for name in names:
        if name in visited:
            continue

        stack = [name]
        cluster: list[str] = []
        visited.add(name)

        while stack:
            current = stack.pop()
            cluster.append(current)
            for neighbor in sorted(neighbors[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        clusters.append(sorted(cluster))

    return clusters


def top_token_cloud(
    fingerprint: ModelFingerprint,
    analyzer: TokenAnalyzer | None = None,
    limit: int = 25,
) -> List[Dict[str, object]]:
    """Return top token-cloud entries for visualization."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    entries: list[dict[str, object]] = []
    for token_id, count in fingerprint.token_counts.most_common(limit):
        entry: dict[str, object] = {"token_id": token_id, "count": count}
        if analyzer is not None:
            entry["text"] = analyzer.decode([token_id])
        entries.append(entry)
    return entries


def build_fingerprints_for_models(
    models: Iterable[str],
    prompts: Sequence[str],
    client: OllamaClient | None = None,
    analyzer: TokenAnalyzer | None = None,
    samples_per_prompt: int = 1,
) -> List[ModelFingerprint]:
    """Build fingerprints for multiple models from the same prompt suite."""
    shared_client = client or OllamaClient()
    shared_analyzer = analyzer or TokenAnalyzer()
    return [
        build_model_fingerprint(
            model,
            prompts,
            client=shared_client,
            analyzer=shared_analyzer,
            samples_per_prompt=samples_per_prompt,
        )
        for model in models
    ]
