from __future__ import annotations

from dataclasses import dataclass
from typing import List

from spice.errors import InvalidModelError
from spice.providers import ANTHROPIC, OPEN_AI, Provider

# Used to fetch a model by name
models: List[Model] = []


# TODO: Add costs to models
@dataclass
class Model:
    name: str
    provider: Provider | None

    def __post_init__(self):
        models.append(self)


@dataclass
class UnknownModel(Model):
    pass


@dataclass
class TextModel(Model):
    context_length: int


@dataclass
class VisionModel(TextModel):
    pass


@dataclass
class EmbeddingModel(Model):
    pass


@dataclass
class TranscriptionModel(Model):
    pass


# TODO: Add more models
GPT_4_0125_PREVIEW = TextModel("gpt-4-0125-preview", OPEN_AI, 128000)
GPT_4_VISION_PREVIEW = VisionModel("gpt-4-vision-preview", OPEN_AI, 128000)
GPT_35_TURBO_0125 = TextModel("gpt-3.5-turbo-0125", OPEN_AI, 16385)

CLAUDE_3_OPUS_20240229 = TextModel("claude-3-opus-20240229", ANTHROPIC, 200000)
CLAUDE_3_HAIKU_20240307 = TextModel("claude-3-haiku-20240307", ANTHROPIC, 200000)

WHISPER_1 = TranscriptionModel("whisper-1", OPEN_AI)

TEXT_EMBEDDING_3_LARGE = EmbeddingModel("text-embedding-3-large", OPEN_AI)
TEXT_EMBEDDING_3_SMALL = EmbeddingModel("text-embedding-3-small", OPEN_AI)
TEXT_EMBEDDING_ADA_002 = EmbeddingModel("text-embedding-ada-002", OPEN_AI)


def get_model_from_name(model_name: str) -> Model:
    # Search backwards; this way user defined models take priority
    for model in reversed(models):
        if model.name == model_name:
            return model

    return UnknownModel(model_name, None)
