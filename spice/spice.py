from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from timeit import default_timer as timer
from typing import AsyncIterator, Callable, Collection, Dict, List, Literal, Optional, Sequence, cast

import httpx
from openai.types.chat.completion_create_params import ResponseFormat

from spice.errors import InvalidModelError, UnknownModelError
from spice.models import EmbeddingModel, Model, TextModel, TranscriptionModel, get_model_from_name
from spice.providers import Provider, get_provider_from_name
from spice.spice_message import SpiceMessage
from spice.utils import embeddings_request_cost, text_request_cost, transcription_request_cost
from spice.wrapped_clients import WrappedClient


@dataclass
class SpiceCallArgs:
    model: str
    messages: Collection[SpiceMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[ResponseFormat] = None


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
    """How long it took for the response to be completed. May be inaccurate for incomplete streamed responses."""

    input_tokens: int
    """The number of input tokens given in this response. May be inaccurate for incomplete streamed responses."""

    output_tokens: int
    """The number of output tokens given by the model in this response. May be inaccurate for incomplete streamed responses."""

    completed: bool
    """Whether or not this response was fully completed. This will only ever be false for incomplete streamed responses."""

    cost: Optional[float]
    """The cost of this request in cents. May be inaccurate for incompleted streamed responses. Will be None if the cost of the model used is not known."""

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

    def __init__(
        self,
        model: TextModel,
        call_args: SpiceCallArgs,
        client: WrappedClient,
        stream: AsyncIterator,
        cost_callback: Callable[[float], None],
    ):
        self._model = model
        self._call_args = call_args
        self._text = []
        self._start_time = timer()
        self._end_time = None
        self._input_tokens = None
        self._output_tokens = None
        self._finished = False
        self._client = client
        self._stream = stream
        self._cost_added = 0
        self._cost_callback = cost_callback

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

        input_tokens = self._input_tokens
        if input_tokens is None:
            input_tokens = self._client.count_messages_tokens(self._call_args.messages, self._model)
        output_tokens = self._output_tokens
        if output_tokens is None:
            output_tokens = self._client.count_string_tokens(full_output, self._model, full_message=False)

        cost = text_request_cost(self._model, input_tokens, output_tokens)
        if cost is not None:
            new_cost = cost - self._cost_added
            self._cost_added = cost
            self._cost_callback(new_cost)

        return SpiceResponse(
            self._call_args,
            full_output,
            self._end_time - self._start_time,
            input_tokens,
            output_tokens,
            self._finished,
            cost,
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

        # TODO: Should we validate model aliases?
        self._model_aliases = model_aliases

        self._total_cost: float = 0
        self._prompts: Dict[str, str] = {}

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
        messages: Collection[SpiceMessage],
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
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    async def get_response(
        self,
        messages: Collection[SpiceMessage],
        model: Optional[TextModel | str] = None,
        provider: Optional[Provider | str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> SpiceResponse:
        """
        Asynchronously retrieves a chat completion response.

        Args:
            messages: The list of messages given as context for the completion.
            Will raise an ImageError if any invalid images are given.

            model: The model to use. Must be a text based model. If no model is given, will use the default text model
            the client was initialized with. If the model is unknown to Spice, a provider must be given.
            Will raise an InvalidModelError if the model is not a text model.
            Will raise an UnknownModelError if the model is unknown and no provider was given.

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
        text, input_tokens, output_tokens = client.extract_text_and_tokens(chat_completion)

        cost = text_request_cost(text_model, input_tokens, output_tokens)
        if cost is not None:
            self._total_cost += cost

        response = SpiceResponse(call_args, text, end_time - start_time, input_tokens, output_tokens, True, cost)
        return response

    async def stream_response(
        self,
        messages: Collection[SpiceMessage],
        model: Optional[TextModel | str] = None,
        provider: Optional[Provider | str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[ResponseFormat] = None,
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

        def cost_callback(cost):
            self._total_cost += cost

        return StreamingSpiceResponse(text_model, call_args, client, stream, cost_callback)

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
    ) -> List[List[float]]:
        """
        Asynchronously retrieves embeddings for a list of text.

        Args:
            input_texts: The texts to generate embeddings for.

            model: The embedding model to use. If no model is given, will use the default embedding model
            the client was initialized with. If the model is unknown to Spice, a provider must be given.
            Will raise an InvalidModelError if the model is not a embedding model.
            Will raise an UnknownModelError if the model is unknown and no provider was given.

            provider: The provider to use. If specified, will override the model's default provider if known. Must be specified if an unknown model is used.
        """

        embedding_model = self._get_embedding_model(model)
        client = self._get_client(embedding_model, provider)

        input_tokens = sum(client.count_string_tokens(text, embedding_model, False) for text in input_texts)
        cost = embeddings_request_cost(embedding_model, input_tokens)
        if cost is not None:
            self._total_cost += cost

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
            Will raise an InvalidModelError if the model is not a embedding model.
            Will raise an UnknownModelError if the model is unknown and no provider was given.

            provider: The provider to use. If specified, will override the model's default provider if known. Must be specified if an unknown model is used.
        """

        embedding_model = self._get_embedding_model(model)
        client = self._get_client(embedding_model, provider)

        input_tokens = sum(client.count_string_tokens(text, embedding_model, False) for text in input_texts)
        cost = embeddings_request_cost(embedding_model, input_tokens)
        if cost is not None:
            self._total_cost += cost

        return client.get_embeddings_sync(input_texts, embedding_model.name)

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
    ) -> str:
        """
        Asynchronously retrieves embeddings for a list of text.

        Args:
            audio_path: The path to the audio file to transcribe.

            model: The model to use. Must be a transciption model. If the model is unknown to Spice, a provider must be given.
            Will raise an InvalidModelError if the model is not a transciption model.
            Will raise an UnknownModelError if the model is unknown and no provider was given.

            provider: The provider to use. If specified, will override the model's default provider if known. Must be specified if an unknown model is used.
        """

        transcription_model = self._get_transcription_model(model)
        client = self._get_client(transcription_model, provider)

        transcription, input_length = await client.get_transcription(
            Path(audio_path).expanduser().resolve(), transcription_model.name
        )
        cost = transcription_request_cost(transcription_model, input_length)
        if cost is not None:
            self._total_cost += cost

        return transcription

    ### Prompts ###
    def store_prompt(self, prompt: str, name: str):
        """
        Stores a prompt under the given name

        Args:
            prompt: The prompt to store.

            name: The name of the prompt. If the name collides with the name of a previously loaded prompt, the previous prompt will be overwritten.
        """

        if name in self._prompts:
            # Overwriting prompt; should we do anything?
            pass
        self._prompts[name] = prompt

    def get_prompt(self, name: str) -> str:
        """
        Gets the prompt currently mapped to the given name.

        Args:
            name: The name of the prompt.
        """

        return self._prompts[name]

    def load_prompt(self, file_path: Path | str, name: Optional[str] = None):
        """
        Loads a prompt from the given file.

        Args:
            file_path: The path to the file. Must be a text encoded file.

            name: The name of the prompt. If no name is given, the name will be the file name without the extension; i.e., the name of /path/to/prompt.txt will be `prompt`.
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

    def load_dir(self, dir_path: Path | str):
        """
        Loads a prompts from a given directory. Will recursively load all text encoded .txt files in the directory and its subdirectories.
        Prompt names will be the filenames with their .txt extension stripped with a `.` separating each subdirectory. For example, this directory would have these prompt names:
        If any name collides with the name of a previously loaded prompt, the previous prompt will be overwritten.

        ```
        dir_path/
            sub_dir/
                prompt_1.txt        - sub_dir.prompt_1
                not_a_prompt.jpg
            prompt_2.txt            - prompt_2
            another_prompt.txt      - another_prompt.txt
        ```

        Args:
            dir_path: The path to the directory.
        """

        dir_path = Path(dir_path).expanduser().resolve()
        if not dir_path.exists():
            raise FileNotFoundError()

        file_paths = glob.glob(f"{dir_path}/**/*.txt", recursive=True)
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
            self.store_prompt(prompt, name)

    def load_url(self, url: str):
        """
        Loads a prompt from the given url. Will send a GET request to the url and expects a response with the following schema:
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
        If any name collides with the name of a previously loaded prompt, the previous prompt will be overwritten.

        Args:
            url: The url.
        """

        raw_json = httpx.get(url).content.decode("utf-8")
        try:
            prompts = json.loads(raw_json)
        except JSONDecodeError:
            raise

        try:
            for prompt_object in prompts["prompts"]:
                name, prompt = prompt_object["name"], prompt_object["prompt"]
                self.store_prompt(prompt, name)
        except:
            raise
