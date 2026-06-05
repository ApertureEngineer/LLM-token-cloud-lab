import sys
from pathlib import Path
from collections import Counter
from math import isclose

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model_fingerprint import (
    ModelFingerprint,
    build_model_fingerprint,
    cluster_fingerprints,
    cosine_similarity,
    similarity_matrix,
    top_token_cloud,
)
from token_analysis import TokenAnalyzer


class DummyClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, model: str, prompt: str, stream: bool = False) -> str:
        self.calls.append((model, prompt, stream))
        return self.responses.pop(0)


def fingerprint(model: str, counts: Counter[int]) -> ModelFingerprint:
    return ModelFingerprint(
        model=model,
        token_counts=counts,
        responses=(),
        prompt_count=0,
        sample_count=0,
    )


def test_build_model_fingerprint_counts_generated_response_tokens() -> None:
    analyzer = TokenAnalyzer()
    client = DummyClient(["hello world", "hello again"])

    result = build_model_fingerprint(
        "dummy",
        ["prompt-a", "prompt-b"],
        client=client,
        analyzer=analyzer,
    )

    expected = Counter(analyzer.encode("hello world"))
    expected.update(analyzer.encode("hello again"))

    assert result.model == "dummy"
    assert result.token_counts == expected
    assert result.prompt_count == 2
    assert result.sample_count == 2
    assert result.total_tokens == sum(expected.values())
    assert result.unique_tokens == len(expected)
    assert client.calls == [
        ("dummy", "prompt-a", False),
        ("dummy", "prompt-b", False),
    ]


def test_cosine_similarity_compares_sparse_token_vectors() -> None:
    assert isclose(
        cosine_similarity(Counter({1: 2, 2: 2}), Counter({1: 2, 2: 2})),
        1.0,
    )
    assert cosine_similarity(Counter({1: 1}), Counter({2: 1})) == 0.0


def test_cluster_fingerprints_groups_similar_models() -> None:
    alpha = fingerprint("alpha", Counter({1: 5, 2: 3, 3: 1}))
    beta = fingerprint("beta", Counter({1: 4, 2: 3, 3: 1}))
    gamma = fingerprint("gamma", Counter({10: 4, 11: 4}))

    clusters = cluster_fingerprints([alpha, beta, gamma], similarity_threshold=0.95)

    assert clusters == [["alpha", "beta"], ["gamma"]]


def test_similarity_matrix_contains_all_model_pairs() -> None:
    alpha = fingerprint("alpha", Counter({1: 1}))
    beta = fingerprint("beta", Counter({1: 1, 2: 1}))

    matrix = similarity_matrix([alpha, beta])

    assert isclose(matrix["alpha"]["alpha"], 1.0)
    assert matrix["alpha"]["beta"] == matrix["beta"]["alpha"]
    assert isclose(matrix["beta"]["beta"], 1.0)


def test_top_token_cloud_returns_most_common_tokens() -> None:
    fp = fingerprint("alpha", Counter({1: 2, 2: 5, 3: 1}))

    cloud = top_token_cloud(fp, limit=2)

    assert cloud == [
        {"token_id": 2, "count": 5},
        {"token_id": 1, "count": 2},
    ]
