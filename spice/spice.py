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
    """The call arguments given to the model that created this response."""

    text: str
    """The total text sent by the model."""

    total_time: float
    """
    How long it took for the response to be completed.
    May be inaccurate for streamed responses if not iterated over and completed immediately.
    """

    input_tokens: int
    """The number of input tokens given in this response."""

    output_tokens: int
    """The number of output tokens given by the model in this response."""

    completed: bool
    """Whether or not this response was fully completed. This will only ever be false for streaming responses."""

    # TODO: Add cost

    @property
    def total_tokens(self) -> int:
        """The total tokens, input and output, in this response."""
        return self.input_tokens + self.output_tokens

    @property
    def characters_per_second(self) -> float:
        """The characters per second that the model output. May be inaccurate for streamed responses if not iterated over and completed immediately."""
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
    """
    The Spice client. The majority of the time, only one Spice client should be initialized and used.
    Automatically handles multiple providers and their respective errors.
    """

    def __init__(
        self,
        default_text_model: Optional[TextModel | str] = None,
        default_embeddings_model: Optional[EmbeddingModel | str] = None,
        model_aliases: Optional[Dict[str, Model | str]] = None,
    ):
        """
        Creates a new Spice client.

        Args:
            default_text_model: The default model that will be used for chat completions if no other model is given.
            Will raise an InvalidModelError if the model is not a text model.

            default_embeddings_model: The default model that will be used for embeddings if no other model is given.
            Will raise an InvalidModelError if the model is not an embeddings model.

            model_aliases: A custom model name map.
        """

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

        # TODO: Should we validate model aliases?
        self._model_aliases = model_aliases

    def load_provider(self, provider: Provider | str):
        """
        Loads the specified provider and raises a NoAPIKeyError if no valid api key for the provider is found.
        Providers not preloaded will be loaded on first use, and the NoAPIKeyError will be raised when they are used.
        """

        if isinstance(provider, str):
            provider = get_provider_from_name(provider)
        provider.get_client()

    def _get_client(self, model: Model, provider: Optional[Provider | str]) -> WrappedClient:
        if provider is not None:
            if isinstance(provider, str):
                provider = get_provider_from_name(provider)
            return provider.get_client()
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
        provider: Optional[Provider | str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[ResponseFormatType] = None,
    ) -> SpiceResponse:
        """
        Asynchronously retrieves a chat completion response.

        Args:
            messages: The list of messages given as context for the completion.

            model: The model to use. Must be a text based model. If no model is given, will use the default text model
            the client was initialized with. If the model is unknown to Spice, a provider must be given.
            Will raise an InvalidModelError if the model is not a text model, or if the model is unknown and no provider was given.

            provider: The provider to use. If specified, will override the model's default provider if known. Must be specified if an unknown model is used.

            temperature: The temperature to give the model.

            max_tokens: The maximum tokens the model can output.

            response_format: For valid models, will set the response format to 'text' or 'json'.
            If the provider/model does not support response_format, this argument will be ignored.
        """

        text_model = self._get_text_model(model)
        client = self._get_client(text_model, provider)
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
        provider: Optional[Provider | str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[ResponseFormatType] = None,
    ) -> StreamingSpiceResponse:
        """
        Asynchronously retrieves a chat completion stream that can be iterated over asynchronously.

        Args:
            messages: The list of messages given as context for the completion.

            model: The model to use. Must be a text based model. If no model is given, will use the default text model
            the client was initialized with. If the model is unknown to Spice, a provider must be given.
            Will raise an InvalidModelError if the model is not a text model, or if the model is unknown and no provider was given.

            provider: The provider to use. If specified, will override the model's default provider if known. Must be specified if an unknown model is used.

            temperature: The temperature to give the model.

            max_tokens: The maximum tokens the model can output.

            response_format: For valid models, will set the response format to 'text' or 'json'.
            If the provider/model does not support response_format, this argument will be ignored.
        """

        text_model = self._get_text_model(model)
        client = self._get_client(text_model, provider)
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
        self,
        input_texts: List[str],
        model: Optional[EmbeddingModel | str] = None,
        provider: Optional[Provider | str] = None,
    ) -> List[List[float]]:
        """
        Asynchronously retrieves embeddings for a list of text.

        Args:
            input_texts: The texts to generate embeddings for.

            model: The embedding model to use. If no model is given, will use the default embedding model
            the client was initialized with. If the model is unknown to Spice, a provider must be given.
            Will raise an InvalidModelError if the model is not a embedding model, or if the model is unknown and no provider was given.

            provider: The provider to use. If specified, will override the model's default provider if known. Must be specified if an unknown model is used.
        """

        embedding_model = self._get_embedding_model(model)
        client = self._get_client(embedding_model, provider)

        return await client.get_embeddings(input_texts, embedding_model.name)

    def get_embeddings_sync(
        self,
        input_texts: List[str],
        model: Optional[EmbeddingModel | str] = None,
        provider: Optional[Provider | str] = None,
    ) -> List[List[float]]:
        """
        Synchronously retrieves embeddings for a list of text.

        Args:
            input_texts: The texts to generate embeddings for.

            model: The embedding model to use. If no model is given, will use the default embedding model
            the client was initialized with. If the model is unknown to Spice, a provider must be given.
            Will raise an InvalidModelError if the model is not a embedding model, or if the model is unknown and no provider was given.

            provider: The provider to use. If specified, will override the model's default provider if known. Must be specified if an unknown model is used.
        """

        embedding_model = self._get_embedding_model(model)
        client = self._get_client(embedding_model, provider)

        return client.get_embeddings_sync(input_texts, embedding_model.name)

    def _get_transcription_model(self, model: Model | str) -> TranscriptionModel | UnknownModel:
        if self._model_aliases is not None and model in self._model_aliases:
            model = self._model_aliases[model]

        if isinstance(model, str):
            model = get_model_from_name(model)

        if not isinstance(model, (TranscriptionModel, UnknownModel)):
            raise InvalidModelError(f"Model {model} is not a transcription model")

        return model

    async def get_transcription(
        self,
        audio_path: Path,
        model: TranscriptionModel | str,
        provider: Optional[Provider | str] = None,
    ) -> str:
        """
        Asynchronously retrieves embeddings for a list of text.

        Args:
            audio_path: The path to the audio file to transcribe.

            model: The model to use. Must be a transciption model. If the model is unknown to Spice, a provider must be given.
            Will raise an InvalidModelError if the model is not a transciption model, or if the model is unknown and no provider was given.

            provider: The provider to use. If specified, will override the model's default provider if known. Must be specified if an unknown model is used.
        """

        transciption_model = self._get_transcription_model(model)
        client = self._get_client(transciption_model, provider)

        return await client.get_transcription(audio_path, transciption_model.name)
