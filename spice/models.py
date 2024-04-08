from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from spice.errors import InvalidModelError
from spice.providers import ANTHROPIC, OPEN_AI, Provider

# Used to fetch a model by name
models: List[Model] = []


@dataclass
class Model:
    name: str
    """The model name to use when calling the provider."""

    provider: Provider | None
    """The provider this model belongs to. If this is None, a provider must always be passed alongside this model."""

    def __post_init__(self):
        models.append(self)


@dataclass
class TextModel(Model):
    input_cost: Optional[int] = field(default=None)
    """This model's cost per input token in cents / million tokens."""

    output_cost: Optional[int] = field(default=None)
    """This model's cost per output token in cents / million tokens."""

    context_length: Optional[int] = field(default=None)
    """The maximum context length for this model."""


@dataclass
class VisionModel(TextModel):
    pass


@dataclass
class EmbeddingModel(Model):
    input_cost: Optional[int] = field(default=None)
    """This model's cost per input token in cents / million tokens."""


@dataclass
class TranscriptionModel(Model):
    input_cost: Optional[int] = field(default=None)
    """This model's cost in cents / 100 seconds"""


@dataclass
class UnknownModel(TextModel, EmbeddingModel, TranscriptionModel):
    pass


# TODO: Add more models
GPT_4_0125_PREVIEW = TextModel("gpt-4-0125-preview", OPEN_AI, input_cost=1000, output_cost=3000, context_length=128000)
GPT_4_1106_VISION_PREVIEW = VisionModel(
    "gpt-4-1106-vision-preview", OPEN_AI, input_cost=1000, output_cost=3000, context_length=128000
)
GPT_35_TURBO_0125 = TextModel("gpt-3.5-turbo-0125", OPEN_AI, input_cost=50, output_cost=150, context_length=16385)

CLAUDE_3_OPUS_20240229 = TextModel(
    "claude-3-opus-20240229", ANTHROPIC, input_cost=1500, output_cost=7500, context_length=200000
)
CLAUDE_3_HAIKU_20240307 = TextModel(
    "claude-3-haiku-20240307", ANTHROPIC, input_cost=25, output_cost=125, context_length=200000
)

TEXT_EMBEDDING_3_LARGE = EmbeddingModel("text-embedding-3-large", OPEN_AI, input_cost=13)
TEXT_EMBEDDING_3_SMALL = EmbeddingModel("text-embedding-3-small", OPEN_AI, input_cost=2)
TEXT_EMBEDDING_ADA_002 = EmbeddingModel("text-embedding-ada-002", OPEN_AI, input_cost=10)

WHISPER_1 = TranscriptionModel("whisper-1", OPEN_AI, input_cost=1)


def get_model_from_name(model_name: str) -> Model:
    # Search backwards; this way user defined models take priority
    for model in reversed(models):
        if model.name == model_name:
            return model

    return UnknownModel(model_name, None)
