from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional

from spice.providers import ANTHROPIC, OPEN_AI, Provider

# Used to fetch a model by name
models: List[Model] = []


# We give the superclass every single instance variable regardless of model type so that doing getattr isn't necessary when you have a generic Model
@dataclass
class Model:
    name: str
    """The model name to use when calling the provider."""

    provider: Provider | None
    """The provider this model belongs to. If this is None, a provider must always be passed alongside this model."""

    input_cost: Optional[int] = field(default=None)
    """This model's cost per input token in cents / million tokens, or cents / 100 seconds for transcription models."""

    output_cost: Optional[int] = field(default=None)
    """This model's cost per output token in cents / million tokens."""

    context_length: Optional[int] = field(default=None)
    """The maximum context length for this model."""

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

    context_length: Optional[int] = field(default=None)
    """The maximum context length for this model."""


@dataclass
class TranscriptionModel(Model):
    input_cost: Optional[int] = field(default=None)
    """This model's cost in cents / 100 seconds."""


@dataclass
class UnknownModel(TextModel, EmbeddingModel, TranscriptionModel):
    pass


GPT_4_TURBO = TextModel("gpt-4-turbo", OPEN_AI, input_cost=1000, output_cost=3000, context_length=128000)
"""Warning: This model always points to OpenAI's latest GPT-4-Turbo model, so the input and output costs may incorrect. We recommend using specific versions of GPT-4-Turbo instead."""

GPT_4_TURBO_2024_04_09 = TextModel(
    "gpt-4-turbo-2024-04-09", OPEN_AI, input_cost=1000, output_cost=3000, context_length=128000
)

GPT_4_TURBO_PREVIEW = TextModel(
    "gpt-4-turbo-preview", OPEN_AI, input_cost=1000, output_cost=3000, context_length=128000
)
GPT_4_0125_PREVIEW = TextModel("gpt-4-0125-preview", OPEN_AI, input_cost=1000, output_cost=3000, context_length=128000)
GPT_4_1106_PREVIEW = TextModel("gpt-4-1106-preview", OPEN_AI, input_cost=1000, output_cost=3000, context_length=128000)
GPT_4_VISION_PREVIEW = VisionModel(
    "gpt-4-vision-preview", OPEN_AI, input_cost=1000, output_cost=3000, context_length=128000
)
GPT_4_1106_VISION_PREVIEW = VisionModel(
    "gpt-4-1106-vision-preview", OPEN_AI, input_cost=1000, output_cost=3000, context_length=128000
)

GPT_4 = TextModel("gpt-4", OPEN_AI, input_cost=3000, output_cost=6000, context_length=8192)
"""Warning: This model always points to OpenAI's latest GPT-4 model, so the input and output costs may incorrect. We recommend using specific versions of GPT-4 instead."""

GPT_4_0613 = TextModel("gpt-4-0613", OPEN_AI, input_cost=3000, output_cost=6000, context_length=8192)

GPT_4_32K = TextModel("gpt-4-32k", OPEN_AI, input_cost=6000, output_cost=12000, context_length=32768)
"""Warning: This model always points to OpenAI's latest GPT-4-32k model, so the input and output costs may incorrect. We recommend using specific versions of GPT-4-32k instead."""

GPT_4_32K_0613 = TextModel("gpt-4-32k-0613", OPEN_AI, input_cost=6000, output_cost=12000, context_length=32768)

GPT_35_TURBO = TextModel("gpt-3.5-turbo", OPEN_AI, input_cost=50, output_cost=150, context_length=16385)
"""Warning: This model always points to OpenAI's latest GPT-3.5-Turbo model, so the input and output costs may incorrect. We recommend using specific versions of GPT-3.5-Turbo instead."""

GPT_35_TURBO_0125 = TextModel("gpt-3.5-turbo-0125", OPEN_AI, input_cost=50, output_cost=150, context_length=16385)
GPT_35_TURBO_1106 = TextModel("gpt-3.5-turbo-1106", OPEN_AI, input_cost=100, output_cost=200, context_length=16385)

GPT_35_TURBO_0613 = TextModel("gpt-3.5-turbo-0613", OPEN_AI, input_cost=150, output_cost=200, context_length=4096)
GPT_35_TURBO_16K_0613 = TextModel(
    "gpt-3.5-turbo-16k-0613", OPEN_AI, input_cost=300, output_cost=400, context_length=16385
)


CLAUDE_3_OPUS_20240229 = TextModel(
    "claude-3-opus-20240229", ANTHROPIC, input_cost=1500, output_cost=7500, context_length=200000
)
OPUS = CLAUDE_3_OPUS_20240229
CLAUDE_3_SONNET_20240229 = TextModel(
    "claude-3-sonnet-20240229", ANTHROPIC, input_cost=300, output_cost=1500, context_length=200000
)
SONNET = CLAUDE_3_SONNET_20240229
CLAUDE_3_HAIKU_20240307 = TextModel(
    "claude-3-haiku-20240307", ANTHROPIC, input_cost=25, output_cost=125, context_length=200000
)
HAIKU = CLAUDE_3_HAIKU_20240307


TEXT_EMBEDDING_3_LARGE = EmbeddingModel("text-embedding-3-large", OPEN_AI, input_cost=13, context_length=8191)
TEXT_EMBEDDING_3_SMALL = EmbeddingModel("text-embedding-3-small", OPEN_AI, input_cost=2, context_length=8191)
TEXT_EMBEDDING_ADA_002 = EmbeddingModel("text-embedding-ada-002", OPEN_AI, input_cost=10, context_length=8191)

WHISPER_1 = TranscriptionModel("whisper-1", OPEN_AI, input_cost=1)


def get_model_from_name(model_name: str) -> Model:
    # Search backwards; this way user defined models take priority
    for model in reversed(models):
        if model.name == model_name:
            return model

    if model_name.startswith("ft:"):
        adjusted_name = model_name.split(":")[1]
        for model in reversed(models):
            if model.name == adjusted_name:
                if "gpt-3.5-turbo" in model_name:
                    input_cost = 300
                    output_cost = 600
                elif "gpt-4" in model_name:
                    input_cost = 4500
                    output_cost = 9000
                else:
                    input_cost = None
                    output_cost = None
                model = dataclasses.replace(model, name=model_name, input_cost=input_cost, output_cost=output_cost)
                return model

    return UnknownModel(model_name, None)
