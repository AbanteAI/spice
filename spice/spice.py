from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import AsyncIterator, Dict, List, Literal, Optional, cast

from spice.errors import InvalidModelError
from spice.models import EmbeddingModel, Model, TextModel, TranscriptionModel, UnknownModel, get_model_from_name
from spice.providers import Provider, get_provider_from_name
from spice.spice_message import SpiceMessage
from spice.utils import count_messages_tokens, count_string_tokens
from spice.wrapped_clients import WrappedClient

ResponseFormatType = Dict[str, Literal["text", "json"]]


@dataclass
class SpiceCallArgs:
    model: str
    messages: List[SpiceMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[ResponseFormatType] = None


@dataclass
class SpiceResponse:
    """
    Contains a collection of information about a completed LLM call.
    """

    call_args: SpiceCallArgs
    text: str
    total_time: float
    input_tokens: int
    output_tokens: int
    completed: bool
    # TODO: Add cost

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def characters_per_second(self) -> float:
        return len(self.text) / self.total_time


class StreamingSpiceResponse:
    """
    Returned from a streaming llm call. Can be iterated over asynchronously to retrieve the content.
    """

    def __init__(self, call_args: SpiceCallArgs, client: WrappedClient, stream: AsyncIterator):
        self._call_args = call_args
        self._text = []
        self._start_time = timer()
        self._end_time = None
        self._input_tokens = None
        self._output_tokens = None
        self._finished = False
        self._client = client
        self._stream = stream

    def __aiter__(self):
        return self

    async def __anext__(self):
        with self._client.catch_and_convert_errors():
            try:
                content = None
                while content is None:
                    chunk = await anext(self._stream)
                    content, input_tokens, output_tokens = self._client.process_chunk(chunk)
                    if input_tokens is not None:
                        self._input_tokens = input_tokens
                    if output_tokens is not None:
                        self._output_tokens = output_tokens
                self._text.append(content)
                return content
            except StopAsyncIteration:
                self._end_time = timer()
                self._finished = True
                raise

    def current_response(self) -> SpiceResponse:
        """
        Returns a SpiceResponse containing the response as it's been received so far.
        Will not wait for the LLM call to finish.
        """
        if not self._finished or self._end_time is None:
            self._end_time = timer()

        full_output = "".join(self._text)

        # TODO: this message counting methods are just for OpenAI,
        # if other providers also don't send token counts when streaming or are interrupted
        # then we need to add counting methods for them as well.
        # Easiest way to do this would be to add count tokens functions to wrapped client
        input_tokens = self._input_tokens
        if input_tokens is None:
            input_tokens = count_messages_tokens(self._call_args.messages, self._call_args.model)
        output_tokens = self._output_tokens
        if output_tokens is None:
            output_tokens = count_string_tokens(full_output, self._call_args.model, full_message=False)

        return SpiceResponse(
            self._call_args,
            full_output,
            self._end_time - self._start_time,
            input_tokens,
            output_tokens,
            self._finished,
        )

    async def complete_response(self) -> SpiceResponse:
        """
        Waits until the entire LLM call finishes, collects it, and returns its SpiceResponse.
        """
        async for _ in self:
            pass
        return self.current_response()


class Spice:
    def __init__(
        self,
        default_text_model: Optional[TextModel | str] = None,
        default_embeddings_model: Optional[EmbeddingModel | str] = None,
        provider: Optional[Provider | str] = None,
        model_aliases: Optional[Dict[str, Model | str]] = None,
    ):
        if isinstance(default_text_model, str):
            text_model = get_model_from_name(default_text_model)
        else:
            text_model = default_text_model
        if text_model and not isinstance(text_model, (TextModel, UnknownModel)):
            raise InvalidModelError("Default text model must be a text model")
        self._default_text_model = text_model

        if isinstance(default_embeddings_model, str):
            embeddings_model = get_model_from_name(default_embeddings_model)
        else:
            embeddings_model = default_embeddings_model
        if embeddings_model and not isinstance(embeddings_model, (EmbeddingModel, UnknownModel)):
            raise InvalidModelError("Default embeddings model must be an embeddings model")
        self._default_embeddings_model = embeddings_model

        if isinstance(provider, str):
            provider = get_provider_from_name(provider)
        self._provider = provider

        # TODO: Should we validate model aliases?
        self._model_aliases = model_aliases

    def _get_client(self, model: Model) -> WrappedClient:
        if self._provider is not None:
            return self._provider.get_client()
        else:
            if model.provider is None:
                raise InvalidModelError("Provider is required when unknown models are used")
            return model.provider.get_client()

    def _get_text_model(self, model: Optional[Model | str]) -> TextModel | UnknownModel:
        if model is None:
            if self._default_text_model is None:
                raise InvalidModelError("Model is required when default text model is not set at initialization")
            model = self._default_text_model

        if self._model_aliases is not None and model in self._model_aliases:
            model = self._model_aliases[model]

        if isinstance(model, str):
            model = get_model_from_name(model)

        if not isinstance(model, (TextModel, UnknownModel)):
            raise InvalidModelError(f"Model {model} is not a text model")

        return model

    def _fix_call_args(
        self,
        messages: List[SpiceMessage],
        model: Model,
        stream: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
        response_format: Optional[ResponseFormatType],
    ):
        # Not all providers support response format
        if response_format is not None:
            if response_format == {"type": "text"}:
                response_format = None

        return SpiceCallArgs(
            model=model.name,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    async def get_response(
        self,
        messages: List[SpiceMessage],
        model: Optional[TextModel | str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[ResponseFormatType] = None,
    ) -> SpiceResponse:
        text_model = self._get_text_model(model)
        client = self._get_client(text_model)
        call_args = self._fix_call_args(messages, text_model, False, temperature, max_tokens, response_format)

        start_time = timer()
        with client.catch_and_convert_errors():
            chat_completion = await client.get_chat_completion_or_stream(call_args)
        end_time = timer()
        input_tokens, output_tokens = client.get_input_and_output_tokens(chat_completion)

        response = SpiceResponse(
            call_args, client.extract_text(chat_completion), end_time - start_time, input_tokens, output_tokens, True
        )
        return response

    async def stream_response(
        self,
        messages: List[SpiceMessage],
        model: Optional[TextModel | str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[ResponseFormatType] = None,
    ) -> StreamingSpiceResponse:
        text_model = self._get_text_model(model)
        client = self._get_client(text_model)
        call_args = self._fix_call_args(messages, text_model, True, temperature, max_tokens, response_format)

        with client.catch_and_convert_errors():
            stream = await client.get_chat_completion_or_stream(call_args)
        stream = cast(AsyncIterator, stream)
        return StreamingSpiceResponse(call_args, client, stream)

    def _get_embedding_model(self, model: Optional[Model | str]) -> EmbeddingModel | UnknownModel:
        if model is None:
            if self._default_embeddings_model is None:
                raise InvalidModelError("Model is required when default embeddings model is not set at initialization")
            model = self._default_embeddings_model

        if self._model_aliases is not None and model in self._model_aliases:
            model = self._model_aliases[model]

        if isinstance(model, str):
            model = get_model_from_name(model)

        if not isinstance(model, (EmbeddingModel, UnknownModel)):
            raise InvalidModelError(f"Model {model} is not a embedding model")

        return model

    async def get_embeddings(
        self, input_texts: List[str], model: Optional[EmbeddingModel | str] = None
    ) -> List[List[float]]:
        embedding_model = self._get_embedding_model(model)
        client = self._get_client(embedding_model)

        return await client.get_embeddings(input_texts, embedding_model.name)

    def get_embeddings_sync(
        self, input_texts: List[str], model: Optional[EmbeddingModel | str] = None
    ) -> List[List[float]]:
        embedding_model = self._get_embedding_model(model)
        client = self._get_client(embedding_model)

        return client.get_embeddings_sync(input_texts, embedding_model.name)

    def _get_transcription_model(self, model: Model | str) -> TranscriptionModel | UnknownModel:
        if self._model_aliases is not None and model in self._model_aliases:
            model = self._model_aliases[model]

        if isinstance(model, str):
            model = get_model_from_name(model)

        if not isinstance(model, (TranscriptionModel, UnknownModel)):
            raise InvalidModelError(f"Model {model} is not a transcription model")

        return model

    async def get_transcription(self, audio_path: Path, model: TranscriptionModel | str) -> str:
        transciption_model = self._get_transcription_model(model)
        client = self._get_client(transciption_model)

        return await client.get_transcription(audio_path, transciption_model.name)
