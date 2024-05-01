from typing import Optional

from spice.models import EmbeddingModel, TextModel, TranscriptionModel


def text_request_cost(model: TextModel, input_tokens: int, output_tokens: int) -> Optional[float]:
    if model.input_cost is not None and model.output_cost is not None:
        return (model.input_cost * input_tokens + model.output_cost * output_tokens) / 1_000_000
    else:
        return None


def embeddings_request_cost(model: EmbeddingModel, input_tokens: int) -> Optional[float]:
    if model.input_cost is not None:
        return (model.input_cost * input_tokens) / 1_000_000
    else:
        return None


def transcription_request_cost(model: TranscriptionModel, input_length: float) -> Optional[float]:
    if model.input_cost is not None:
        return (model.input_cost * input_length) / 100
    else:
        return None


def print_stream(text: str) -> None:
    print(text, end="", flush=True)
