"""Utilities to allow two local OLLAMA models to converse with each other."""
from __future__ import annotations

from typing import List, Tuple

from ollama_client import OllamaClient


def have_conversation(
    model_a: str,
    model_b: str,
    prompt: str,
    turns: int = 4,
    client: OllamaClient | None = None,
) -> List[Tuple[str, str]]:
    """Have two models converse by generating responses alternately.

    Parameters
    ----------
    model_a, model_b:
        Names of the models participating in the conversation.
    prompt:
        Initial text to seed the conversation.
    turns:
        Number of turns in the conversation. Each turn represents a single
        model response. Thus, ``turns`` of 4 will produce two responses from each
        model.
    client:
        Optional :class:`OllamaClient` instance. If omitted a new client will be
        created.

    Returns
    -------
    list of tuple
        A list of ``(model_name, response)`` pairs in the order they were
        produced.
    """

    if client is None:
        client = OllamaClient()

    history: List[Tuple[str, str]] = []
    conversation_context = prompt
    current_model = model_a

    for _ in range(turns):
        response = client.generate(current_model, conversation_context, stream=False)
        history.append((current_model, response))
        conversation_context += f"\n{current_model}: {response}"
        current_model = model_b if current_model == model_a else model_a

    return history


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Make two OLLAMA models talk to each other.")
    parser.add_argument("prompt", help="Initial prompt to start the conversation")
    parser.add_argument("--model-a", default="llama2", help="Name of the first model")
    parser.add_argument("--model-b", default="llama2", help="Name of the second model")
    parser.add_argument("--turns", type=int, default=4, help="Number of turns in the conversation")
    args = parser.parse_args()

    history = have_conversation(args.model_a, args.model_b, args.prompt, args.turns)
    for model, text in history:
        print(f"{model}: {text}")


if __name__ == "__main__":
    main()
