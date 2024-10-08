from __future__ import annotations

import glob
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from timeit import default_timer as timer
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Collection,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    cast,
)

import httpx
from jinja2 import DictLoader, Environment
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import BaseModel, Field

from spice.call_args import SpiceCallArgs
from spice.errors import InvalidModelError, UnknownModelError
from spice.models import EmbeddingModel, Model, TextModel, TranscriptionModel, get_model_from_name
from spice.providers import Provider, get_provider_from_name
from spice.retry_strategy import Behavior, RetryStrategy
from spice.retry_strategy.default_strategy import DefaultRetryStrategy
from spice.spice_message import SpiceMessage, SpiceMessages
from spice.utils import (
    embeddings_request_cost,
    print_stream,
    string_identity,
    text_request_cost,
    transcription_request_cost,
)
from spice.wrapped_clients import TextAndTokens, WrappedClient

T = TypeVar("T")


class SpiceResponse(BaseModel, Generic[T]):
    call_args: SpiceCallArgs = Field(
        description="""The call arguments given to the model that created this response."""
    )
    text: str = Field(description="""The total text sent by the model.""")
    total_time: float = Field(
        description="""How long it took for the response to be completed.
        May be inaccurate for incomplete streamed responses."""
    )
    input_tokens: int = Field(
        description="""The number of input tokens given in this response.
        May be inaccurate for incomplete streamed responses."""
    )
    cache_creation_input_tokens: int = Field(
        description="""The number of input tokens cached as part of this response."""
    )
    cache_read_input_tokens: int = Field(
        description="""The number of input tokens read from cache as part of this response."""
    )
    output_tokens: int = Field(
        description="""The number of output tokens given by the model in this response.
        May be inaccurate for incomplete streamed responses."""
    )
    reasoning_tokens: int = Field(
        description="""The number of reasoning tokens given by the model in this response. These are
        also counted in output_tokens. Only applies to OpenAI o1 models.""",
    )
    completed: bool = Field(
        description="""Whether or not this response was fully completed.
        This will only ever be false for incomplete streamed responses."""
    )
    cost: Optional[float] = Field(
        default=None,
        description="""The cost of this request in cents.
        May be inaccurate for incompleted streamed responses.
        Will be None if the cost of the model used is not known.""",
    )

    def __init__(self, result: Optional[T] = None, **kwargs):
        super().__init__(**kwargs)
        self._result = result

    @property
    def result(self) -> T:
        """The result of this response, if a converter was given, otherwise the raw text."""
        if self._result is None:
            return self.text  # type: ignore
        return self._result

    @property
    def total_tokens(self) -> int:
        """The total tokens, input and output, in this response."""
        return self.input_tokens + self.output_tokens

    @property
    def characters_per_second(self) -> float:
        """
        The characters per second that the model output.
        May be inaccurate for streamed responses if not iterated over and completed immediately.
        """
        return len(self.text) / self.total_time


class StreamingSpiceResponse:
    """
    Returned from a streaming llm call. Can be iterated over asynchronously to retrieve the content.
    """

    def __init__(
        self,
        model: TextModel,
        call_args: SpiceCallArgs,
        client: WrappedClient,
        stream: AsyncIterator,
        callback: Optional[Callable[[SpiceResponse], None]] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
    ):
        self._model = model
        self._call_args = call_args
        self._text = []
        self._start_time = timer()
        self._end_time = None
        self._input_tokens = None
        self._cache_creation_input_tokens = None
        self._cache_read_input_tokens = None
        self._output_tokens = None
        self._finished = False
        self._client = client
        self._stream = stream
        self._callback = callback
        self._streaming_callback = streaming_callback

    def __aiter__(self):
        return self

    async def __anext__(self):
        with self._client.catch_and_convert_errors():
            try:
                content = None
                while content is None:
                    chunk = await anext(self._stream)
                    text_and_tokens = self._client.process_chunk(chunk, self._call_args)
                    content = text_and_tokens.text
                    if text_and_tokens.input_tokens is not None:
                        self._input_tokens = text_and_tokens.input_tokens
                    if text_and_tokens.cache_creation_input_tokens is not None:
                        self._cache_creation_input_tokens = text_and_tokens.cache_creation_input_tokens
                    if text_and_tokens.cache_read_input_tokens is not None:
                        self._cache_read_input_tokens = text_and_tokens.cache_read_input_tokens
                    if text_and_tokens.output_tokens is not None:
                        self._output_tokens = text_and_tokens.output_tokens
                if self._streaming_callback is not None:
                    self._streaming_callback(content)
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

        input_tokens = self._input_tokens
        if input_tokens is None:
            input_tokens = self._client.count_messages_tokens(self._call_args.messages, self._model)
        cache_creation_input_tokens = self._cache_creation_input_tokens
        if cache_creation_input_tokens is None:
            cache_creation_input_tokens = 0
        cache_read_input_tokens = self._cache_read_input_tokens
        if cache_read_input_tokens is None:
            cache_read_input_tokens = 0
        output_tokens = self._output_tokens
        if output_tokens is None:
            output_tokens = self._client.count_string_tokens(full_output, self._model, full_message=False)

        cost = text_request_cost(
            self._model, input_tokens, cache_creation_input_tokens, cache_read_input_tokens, output_tokens
        )
        response = SpiceResponse(
            call_args=self._call_args,
            text=full_output,
            total_time=self._end_time - self._start_time,
            input_tokens=input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=0,
            completed=self._finished,
            cost=cost,
        )
        if self._callback is not None:
            self._callback(response)

        return response

    async def complete_response(self) -> SpiceResponse:
        """
        Waits until the entire LLM call finishes, collects it, and returns its SpiceResponse.
        """
        async for _ in self:
            pass
        return self.current_response()


@dataclass
class EmbeddingResponse:
    embeddings: List[List[float]]
    """The list of embeddings of the list of input texts."""

    total_time: float
    """How long it took for the response to be completed."""

    input_tokens: int
    """The number of input tokens given in this response."""

    cost: Optional[float]
    """The cost of this request in cents. Will be None if the cost of the model used is not known."""


@dataclass
class TranscriptionResponse:
    text: str
    """The transcription of the input audio."""

    total_time: float
    """How long it took for the response to be completed."""

    input_length: float
    """The length of the input audio in seconds."""

    cost: Optional[float]
    """The cost of this request in cents. Will be None if the cost of the model used is not known."""


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
        logging_dir: Optional[Path | str] = None,
        logging_callback: Optional[Callable[[SpiceResponse, str, str], None]] = None,
        default_temperature: Optional[float] = None,
    ):
        """
        Creates a new Spice client.

        Args:
            default_text_model: The default model that will be used for chat completions if no other model is given.
            Will raise an InvalidModelError if the model is not a text model.

            default_embeddings_model: The default model that will be used for embeddings if no other model is given.
            Will raise an InvalidModelError if the model is not an embeddings model.

            model_aliases: A custom model name map.

            logging_dir: If not None, will log all api calls to the given directory.

            logging_callback: If not None, will call the given function with the SpiceResponse, the name of the run,
            and the name of the call after every call finishes.

            default_temperature: The default temperature to use for chat completions if no other temperature is given.
        """

        if isinstance(default_text_model, str):
            text_model = get_model_from_name(default_text_model)
        else:
            text_model = default_text_model
        if text_model and not isinstance(text_model, TextModel):
            raise InvalidModelError("Default text model must be a text model")
        self._default_text_model = text_model

        if isinstance(default_embeddings_model, str):
            embeddings_model = get_model_from_name(default_embeddings_model)
        else:
            embeddings_model = default_embeddings_model
        if embeddings_model and not isinstance(embeddings_model, EmbeddingModel):
            raise InvalidModelError("Default embeddings model must be an embeddings model")
        self._default_embeddings_model = embeddings_model
        self._default_temperature = default_temperature

        # TODO: Should we validate model aliases?
        self._model_aliases = model_aliases

        self._total_cost: float = 0
        self._prompts: Dict[str, str] = {}

        self.logging_dir = None if logging_dir is None else Path(logging_dir).expanduser()
        self.logging_callback = logging_callback
        self.new_run("spice")

    def new_messages(self) -> SpiceMessages:
        """
        Returns a new SpiceMessages object.
        """
        return SpiceMessages(self)

    def new_run(self, name: str):
        """
        Create a new run. All llm calls will be logged in a folder with the run name and a timestamp.
        """
        timestamp = datetime.now().strftime("%m%d%y_%H%M%S")
        self._cur_run = f"{name}_{timestamp}"
        self._cur_logged_names = defaultdict(int)
        self._log_prompts()

    def _log_prompts(self):
        if self.logging_dir is not None:
            logging_dir = self.logging_dir / self._cur_run
            logging_dir.mkdir(exist_ok=True, parents=True)
            with open(logging_dir / "prompts.json", "w") as file:
                json.dump(self._prompts, file)

    def _log_response(self, response: SpiceResponse, name: Optional[str] = None):
        base_name = "spice" if name is None else name

        if self.logging_dir is not None:
            full_name = f"{base_name}_{self._cur_logged_names[base_name]}.json"
            self._cur_logged_names[base_name] += 1
            response_json = response.model_dump_json()

            logging_dir = self.logging_dir / self._cur_run
            logging_dir.mkdir(exist_ok=True, parents=True)
            with (logging_dir / full_name).open("w") as file:
                file.write(f"{response_json}\n")
        if self.logging_callback is not None:
            self.logging_callback(response, self._cur_run, base_name)

    @property
    def total_cost(self) -> float:
        """The total cost in cents of all api calls made through this Spice client."""
        return self._total_cost

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
                raise UnknownModelError("Provider is required when unknown models are used")
            return model.provider.get_client()

    def _get_model(self, model: Model | str) -> Model:
        if self._model_aliases is not None and model in self._model_aliases:
            model = self._model_aliases[model]

        if isinstance(model, str):
            model = get_model_from_name(model)

        return model

    def _get_text_model(self, model: Optional[Model | str]) -> TextModel:
        if model is None:
            if self._default_text_model is None:
                raise InvalidModelError("Model is required when default text model is not set at initialization")
            model = self._default_text_model

        if self._model_aliases is not None and model in self._model_aliases:
            model = self._model_aliases[model]

        if isinstance(model, str):
            model = get_model_from_name(model)

        if not isinstance(model, TextModel):
            raise InvalidModelError(f"Model {model} is not a text model")

        return model

    def _fix_call_args(
        self,
        messages: List[SpiceMessage],
        model: Model,
        stream: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
        response_format: Optional[ResponseFormat],
    ):
        # Not all providers support response format
        if response_format is not None:
            if response_format == {"type": "text"}:
                response_format = None

        return SpiceCallArgs(
            model=model.name,
            messages=list(messages),  # convert from SpiceMessages so we can serialize
            stream=stream,
            temperature=self._default_temperature if temperature is None else temperature,
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
        response_format: Optional[ResponseFormat] = None,
        name: Optional[str] = None,
        validator: Optional[Callable[[str], bool]] = None,
        converter: Callable[[str], T] = string_identity,
        streaming_callback: Optional[Callable[[str], None]] = None,
        retries: int = 0,
        retry_strategy: Optional[RetryStrategy[T]] = None,
    ) -> SpiceResponse[T]:
        """
        Asynchronously retrieves a chat completion response.

        Args:
            messages: The list of messages given as context for the completion.
            Will raise an ImageError if any invalid images are given.

            model: The model to use. Must be a text based model. If no model is given, will use the default text model
            the client was initialized with. If the model is unknown to Spice, a provider must be given.
            Will raise an InvalidModelError if the model is not a text model.
            Will raise an UnknownModelError if the model is unknown and no provider was given.

            provider: The provider to use. If specified, will override the model's default provider if known.
            Must be specified if an unknown model is used.

            temperature: The temperature to give the model.

            max_tokens: The maximum tokens the model can output.

            response_format: For valid models, will set the response format to 'text' or 'json'.
            If the provider/model does not support response_format, this argument will be ignored.

            name: If given, will be given this name when logged.

            validator: If given, will be called with the text of the response.
            If it returns False, the response will be discarded and another attempt will be made.

            converter: If given, will be called with the text of the response.
            The result of the converter will be the result of the response.
            If the converter throws an exception, the response will be discarded
            and another attempt will be made up to retries times.

            streaming_callback: If given, will be called with the text of the response as it is received.

            retries: The number of times to retry getting a valid response. If 0, will not retry.

            retry_strategy: The strategy to use for retrying the request. If None, a DefaultRetryStrategy will be used.
            The strategy determines which model responses will be accepted and on an invalid response how the call_args
            will be modified.
        """
        if retry_strategy is None:
            retry_strategy = DefaultRetryStrategy(validator, converter, retries)

        cost = 0
        attempt_number = 0
        text_model = self._get_text_model(model)
        call_args = self._fix_call_args(
            messages, text_model, streaming_callback is not None, temperature, max_tokens, response_format
        )
        while True:
            start_time = timer()
            text_model = self._get_text_model(call_args.model)
            client = self._get_client(text_model, provider)

            with client.catch_and_convert_errors():
                if streaming_callback is not None:
                    stream = await client.get_chat_completion_or_stream(call_args)
                    stream = cast(AsyncIterator, stream)
                    streaming_spice_response = StreamingSpiceResponse(
                        text_model, call_args, client, stream, None, streaming_callback
                    )
                    chat_completion = await streaming_spice_response.complete_response()
                    if streaming_callback == print_stream:
                        print()
                    text_and_tokens = TextAndTokens(
                        text=chat_completion.text,
                        input_tokens=chat_completion.input_tokens,
                        cache_creation_input_tokens=chat_completion.cache_creation_input_tokens,
                        cache_read_input_tokens=chat_completion.cache_read_input_tokens,
                        output_tokens=chat_completion.output_tokens,
                    )
                else:
                    chat_completion = await client.get_chat_completion_or_stream(call_args)
                    text_and_tokens = client.extract_text_and_tokens(chat_completion, call_args)

            completion_cost = text_request_cost(
                text_model,
                text_and_tokens.input_tokens,  # type: ignore
                text_and_tokens.cache_creation_input_tokens,  # type: ignore
                text_and_tokens.cache_read_input_tokens,  # type: ignore
                text_and_tokens.output_tokens,  # type: ignore
            )
            if completion_cost is not None:
                cost += completion_cost
                self._total_cost += completion_cost

            end_time = timer()

            behavior, next_call_args, result, call_name = retry_strategy.decide(
                call_args,
                attempt_number,
                text_and_tokens.text,  # type: ignore
                name or "",
            )
            response = SpiceResponse(
                call_args=call_args,
                text=text_and_tokens.text,  # type: ignore
                total_time=end_time - start_time,
                input_tokens=text_and_tokens.input_tokens,  # type: ignore
                cache_creation_input_tokens=text_and_tokens.cache_creation_input_tokens,  # type: ignore
                cache_read_input_tokens=text_and_tokens.cache_read_input_tokens,  # type: ignore
                output_tokens=text_and_tokens.output_tokens,  # type: ignore
                reasoning_tokens=text_and_tokens.reasoning_tokens or 0,  # type: ignore
                completed=True,
                cost=cost,
                result=result,
            )
            self._log_response(response, call_name)
            if behavior == Behavior.RETURN:
                return response
            else:
                attempt_number += 1
                call_args = next_call_args

    async def stream_response(
        self,
        messages: List[SpiceMessage],
        model: Optional[TextModel | str] = None,
        provider: Optional[Provider | str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[ResponseFormat] = None,
        name: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> StreamingSpiceResponse:
        """
        Asynchronously retrieves a chat completion stream that can be iterated over asynchronously.

        Args:
            messages: The list of messages given as context for the completion.
            Will raise an ImageError if any invalid images are given.

            model: The model to use. Must be a text based model. If no model is given, will use the default text model
            the client was initialized with. If the model is unknown to Spice, a provider must be given.
            Will raise an InvalidModelError if the model is not a text model.
            Will raise an UnknownModelError if the model is unknown and no provider was given.

            provider: The provider to use. If specified, will override the model's default provider if known.
            Must be specified if an unknown model is used.

            temperature: The temperature to give the model.

            max_tokens: The maximum tokens the model can output.

            response_format: For valid models, will set the response format to 'text' or 'json'.
            If the provider/model does not support response_format, this argument will be ignored.

            name: If given, will be given this name when logged.
        """
        text_model = self._get_text_model(model)
        client = self._get_client(text_model, provider)
        call_args = self._fix_call_args(messages, text_model, True, temperature, max_tokens, response_format)

        with client.catch_and_convert_errors():
            stream = await client.get_chat_completion_or_stream(call_args)
        stream = cast(AsyncIterator, stream)

        def callback(response: SpiceResponse, cache: List[float] = [0]):
            if response.cost is not None:
                self._total_cost += response.cost - cache[0]
                cache[0] = response.cost
            # TODO: Do we want to log multiple times? Our old log might be incomplete, which is why this is still in,
            # but probably not necessary.
            self._log_response(response, name)

        return StreamingSpiceResponse(text_model, call_args, client, stream, callback, streaming_callback)

    def _get_embedding_model(self, model: Optional[Model | str]) -> EmbeddingModel:
        if model is None:
            if self._default_embeddings_model is None:
                raise InvalidModelError("Model is required when default embeddings model is not set at initialization")
            model = self._default_embeddings_model

        if self._model_aliases is not None and model in self._model_aliases:
            model = self._model_aliases[model]

        if isinstance(model, str):
            model = get_model_from_name(model)

        if not isinstance(model, EmbeddingModel):
            raise InvalidModelError(f"Model {model} is not a embedding model")

        return model

    async def get_embeddings(
        self,
        input_texts: List[str],
        model: Optional[EmbeddingModel | str] = None,
        provider: Optional[Provider | str] = None,
    ) -> EmbeddingResponse:
        """
        Asynchronously retrieves embeddings for a list of text.

        Args:
            input_texts: The texts to generate embeddings for.

            model: The embedding model to use. If no model is given, will use the default embedding model
            the client was initialized with. If the model is unknown to Spice, a provider must be given.
            Will raise an InvalidModelError if the model is not a embedding model.
            Will raise an UnknownModelError if the model is unknown and no provider was given.

            provider: The provider to use. If specified, will override the model's default provider if known.
            Must be specified if an unknown model is used.
        """

        embedding_model = self._get_embedding_model(model)
        client = self._get_client(embedding_model, provider)

        input_tokens = sum(client.count_string_tokens(text, embedding_model, False) for text in input_texts)
        cost = embeddings_request_cost(embedding_model, input_tokens)
        if cost is not None:
            self._total_cost += cost

        start_time = timer()
        with client.catch_and_convert_errors():
            embeddings = await client.get_embeddings(input_texts, embedding_model.name)
        end_time = timer()

        return EmbeddingResponse(embeddings, end_time - start_time, input_tokens, cost)

    def get_embeddings_sync(
        self,
        input_texts: List[str],
        model: Optional[EmbeddingModel | str] = None,
        provider: Optional[Provider | str] = None,
    ) -> EmbeddingResponse:
        """
        Synchronously retrieves embeddings for a list of text.

        Args:
            input_texts: The texts to generate embeddings for.

            model: The embedding model to use. If no model is given, will use the default embedding model
            the client was initialized with. If the model is unknown to Spice, a provider must be given.
            Will raise an InvalidModelError if the model is not a embedding model.
            Will raise an UnknownModelError if the model is unknown and no provider was given.

            provider: The provider to use. If specified, will override the model's default provider if known.
            Must be specified if an unknown model is used.
        """

        embedding_model = self._get_embedding_model(model)
        client = self._get_client(embedding_model, provider)

        input_tokens = sum(client.count_string_tokens(text, embedding_model, False) for text in input_texts)
        cost = embeddings_request_cost(embedding_model, input_tokens)
        if cost is not None:
            self._total_cost += cost

        start_time = timer()
        with client.catch_and_convert_errors():
            embeddings = client.get_embeddings_sync(input_texts, embedding_model.name)
        end_time = timer()

        return EmbeddingResponse(embeddings, end_time - start_time, input_tokens, cost)

    def _get_transcription_model(self, model: Model | str) -> TranscriptionModel:
        if self._model_aliases is not None and model in self._model_aliases:
            model = self._model_aliases[model]

        if isinstance(model, str):
            model = get_model_from_name(model)

        if not isinstance(model, TranscriptionModel):
            raise InvalidModelError(f"Model {model} is not a transcription model")

        return model

    async def get_transcription(
        self,
        audio_path: Path | str,
        model: TranscriptionModel | str,
        provider: Optional[Provider | str] = None,
    ) -> TranscriptionResponse:
        """
        Asynchronously retrieves embeddings for a list of text.

        Args:
            audio_path: The path to the audio file to transcribe.

            model: The model to use. Must be a transciption model.
            If the model is unknown to Spice, a provider must be given.
            Will raise an InvalidModelError if the model is not a transciption model.
            Will raise an UnknownModelError if the model is unknown and no provider was given.

            provider: The provider to use. If specified, will override the model's default provider if known.
            Must be specified if an unknown model is used.
        """

        transcription_model = self._get_transcription_model(model)
        client = self._get_client(transcription_model, provider)

        start_time = timer()
        with client.catch_and_convert_errors():
            transcription, input_length = await client.get_transcription(
                Path(audio_path).expanduser().resolve(), transcription_model.name
            )
        end_time = timer()

        cost = transcription_request_cost(transcription_model, input_length)
        if cost is not None:
            self._total_cost += cost

        return TranscriptionResponse(transcription, end_time - start_time, input_length, cost)

    def count_tokens(
        self, text: str, model: Model | str, provider: Optional[Provider | str] = None, is_message: bool = False
    ) -> int:
        """
        Calculates the tokens in the given text. Will not be accurate for a chat completion prompt.
        Use count_prompt_tokens to get the exact amount of tokens for a prompt.

        Args:
            text: The text to count the tokens of.

            model: The model whose tokenizer will be used.
            Will raise UnknownModelError if the model is unknown and no provider is given.

            provider: The provider to use. If specified, will override the model's default provider if known.
            Must be specified if an unknown model is used.

            is_message: If true, will include the extra tokens that messages in chat completions add on.
            Most of the time, you'll want to keep this false.
        """

        model = self._get_model(model)
        client = self._get_client(model, provider)

        return client.count_string_tokens(text, model, is_message)

    def count_prompt_tokens(
        self, messages: Collection[SpiceMessage], model: Model | str, provider: Optional[Provider | str] = None
    ) -> int:
        """
        Calculates the tokens that the messages would have if used in a chat completion.

        Args:
            messages: The messages to count the tokens of.

            model: The model whose tokenizer will be used.
            Will raise UnknownModelError if the model is unknown and no provider is given.

            provider: The provider to use. If specified, will override the model's default provider if known.
            Must be specified if an unknown model is used.
        """

        model = self._get_model(model)
        client = self._get_client(model, provider)

        return client.count_messages_tokens(messages, model)

    ### Prompts ###

    _prompts_dirty = False

    def store_prompt(self, prompt: str, name: str):
        """
        Stores a prompt under the given name

        Args:
            prompt: The prompt to store.

            name: The name of the prompt.
            If the name collides with the name of a previously loaded prompt, the previous prompt will be overwritten.
        """

        if name in self._prompts:
            # Overwriting prompt; should we do anything?
            pass
        self._prompts[name] = prompt
        self._prompts_dirty = True

    def get_prompt(self, name: str) -> str:
        """
        Gets the prompt currently mapped to the given name.

        Args:
            name: The name of the prompt.
        """

        if self._prompts_dirty:
            self._log_prompts()
            self._prompts_dirty = False

        return self._prompts[name]

    def get_rendered_prompt(self, name: str, **context: Any) -> str:
        """
        Gets the prompt currently mapped to the given name and renders it using jinja.

        Args:
            name: The name of the prompt.

            context: The jinja kwargs to render the prompt with.
        """

        if self._prompts_dirty:
            self._log_prompts()
            self._prompts_dirty = False

        return Environment(loader=DictLoader(self._prompts)).get_template(name).render(context)

    def load_prompt(self, file_path: Path | str, name: Optional[str] = None):
        """
        Loads a prompt from the given file.

        Args:
            file_path: The path to the file. Must be a text encoded file.

            name: The name of the prompt. If no name is given, the name will be the file name without the extension;
            i.e., the name of /path/to/prompt.txt will be `prompt`.
            If the name collides with the name of a previously loaded prompt, the previous prompt will be overwritten.
        """

        file_path = Path(file_path).expanduser().resolve()

        try:
            prompt = file_path.read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            raise

        if name is None:
            name = file_path.name.rsplit(".", 1)[0]

        self.store_prompt(prompt, name)

    def _load_toml_dict(self, toml_dict: Dict[str, Any], name: str):
        for k, v in toml_dict.items():
            new_name = f"{name}.{k}"
            if isinstance(v, str):
                self.store_prompt(v, new_name)
            elif isinstance(v, Dict):
                self._load_toml_dict(v, new_name)
            else:
                raise ValueError("Invalid prompt TOML: TOML must only contain strings for prompts.")

    def load_toml_prompts(self, file_path: Path | str, name: Optional[str] = None):
        """
        Loads prompts from a given toml file. The names of the prompts in the file will be prefixed by `{name}.`,
        so a file named prompts.toml containing `prompt1 = "Hi!"` would load the prompt as `prompts.prompt1`.
        Only available for Python 3.11 (when tomllib was introduced)!

        Args:
            file_path: The path to the file. Must be a text encoded file containing valid TOML.

            name: The name of the prompt. If no name is given, the name will be the file name without the extension;
            i.e., the name of /path/to/prompt.toml will be `prompt`.
            If the name collides with the name of a previously loaded prompt, the previous prompt will be overwritten.
        """

        import tomllib

        file_path = Path(file_path).expanduser().resolve()

        try:
            toml = file_path.read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            raise

        if name is None:
            name = file_path.name.rsplit(".", 1)[0]

        self._load_toml_dict(tomllib.loads(toml), name)

    def load_dir(self, dir_path: Path | str):
        """
        Loads a prompts from a given directory.
        Will recursively load all text encoded .txt (and .toml if using Python 3.11) files
        in the directory and its subdirectories.
        Prompt names will be the filenames with their extension stripped with a `.` separating each subdirectory.
        For example, this directory would have these prompt names:

        ```
        dir_path/
            sub_dir/
                prompt_1.toml        - sub_dir.prompt_1.{TOML prompt names}
                not_a_prompt.jpg
            prompt_2.txt            - prompt_2
            another_prompt.txt      - another_prompt.txt
        ```
        If any name collides with the name of a previously loaded prompt, the previous prompt will be overwritten.

        Args:
            dir_path: The path to the directory.
        """

        try:
            import tomllib

            use_toml = True
        except ImportError:
            use_toml = False

        dir_path = Path(dir_path).expanduser().resolve()
        if not dir_path.exists():
            raise FileNotFoundError()

        file_paths = glob.glob(f"{dir_path}/**/*.txt", recursive=True) + glob.glob(
            f"{dir_path}/**/*.toml", recursive=True
        )
        for file_path in file_paths:
            file_path = Path(file_path)

            try:
                prompt = file_path.read_text()
            except (UnicodeDecodeError, FileNotFoundError):
                continue

            rel_path = file_path.relative_to(dir_path)
            name = ".".join(
                (part if i != len(rel_path.parts) - 1 else part.rsplit(".", 1)[0])
                for i, part in enumerate(rel_path.parts)
            )
            if str(file_path).endswith(".toml"):
                if use_toml:
                    self._load_toml_dict(tomllib.loads(prompt), name)  # pyright: ignore[reportPossiblyUnboundVariable]
            elif str(file_path).endswith(".txt"):
                self.store_prompt(prompt, name)

    def load_url(self, url: str, name: str):
        """
        Loads a prompt from the given url. Will send a GET request to the url and expects a response with either the
        following schema or a toml file (.toml only available for Python 3.11 and onwards):
        ```
        {
            "prompts": [
                {
                    "name": "<prompt_name>",
                    "prompt": "<prompt>"
                }, ...
            ]
        }
        ```
        TOML will only be loaded if the url points to a .toml file.
        All prompt names will be prefixed with `name.`
        If any name collides with the name of a previously loaded prompt, the previous prompt will be overwritten.

        Args:
            url: The url.

            name: The name to prefix the prompts with.
        """

        response = httpx.get(url).content.decode("utf-8")
        if url.endswith(".toml"):
            import tomllib

            try:
                prompts = tomllib.loads(response)
            except tomllib.TOMLDecodeError:
                raise
            self._load_toml_dict(prompts, name)

        else:
            try:
                prompts = json.loads(response)
            except JSONDecodeError:
                raise

            for prompt_object in prompts["prompts"]:
                name, prompt = prompt_object["name"], prompt_object["prompt"]
                self.store_prompt(prompt, name)
