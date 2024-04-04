from __future__ import annotations

from dataclasses import dataclass
from timeit import default_timer as timer
from typing import AsyncIterator, Dict, List, Literal, Optional, cast

from spice.client_manager import (
    _get_client,
    _get_clients_from_env,
    _get_provider_from_model_name,
    _validate_model_aliases,
)
from spice.errors import SpiceError
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
        if self._end_time is None:
            self._end_time = timer()

        full_output = "".join(self._text)

        # TODO: this message counting methods are just for OpenAI,
        # if other providers also don't send token counts when streaming or are interrupted
        # then we need to add counting methods for them as well.
        # Easiest way to do this would be to add count tokens functions to wrapped client
        if self._input_tokens is None:
            self._input_tokens = count_messages_tokens(self._call_args.messages, self._call_args.model)
        if self._output_tokens is None:
            self._output_tokens = count_string_tokens(full_output, self._call_args.model, full_message=False)

        return SpiceResponse(
            self._call_args,
            full_output,
            self._end_time - self._start_time,
            self._input_tokens,
            self._output_tokens,
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
        model: Optional[str] = None,
        provider: Optional[str] = None,
        model_aliases: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self._default_model = model

        if model is not None:
            if model_aliases is not None:
                raise SpiceError("model_aliases not supported when model is set")
            self._model_aliases = None
            if provider is None:
                provider = _get_provider_from_model_name(model)
            self._default_client = _get_client(provider)
        else:
            if provider is not None:
                self._default_client = _get_client(provider)
            else:
                self._default_client = None
                self._clients = _get_clients_from_env()

            self._model_aliases = model_aliases
            if model_aliases is not None:
                _validate_model_aliases(
                    self._model_aliases,
                    self._clients if self._default_client is None else {provider: self._default_client},
                )

    def _get_model(self, model: Optional[str]):
        if model is None:
            if self._default_model is None:
                raise SpiceError("Model argument is required when default model is not set at initialization")
            model = self._default_model

        if self._model_aliases is not None:
            if model in self._model_aliases:
                model = self._model_aliases[model]["model"]
            else:
                raise SpiceError(f"Unknown model alias: {model}")
        return model

    def _get_client(self, model: str):
        if self._default_client is not None:
            return self._default_client
        else:
            provider = _get_provider_from_model_name(model)
            if provider not in self._clients:
                raise SpiceError(f"Provider {provider} is not currently supported.")
            return self._clients[provider]

    def _fix_call_args(
        self,
        messages: List[SpiceMessage],
        model: str,
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
            model=model,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    async def get_response(
        self,
        messages: List[SpiceMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[ResponseFormatType] = None,
    ) -> SpiceResponse:
        model = self._get_model(model)
        client = self._get_client(model)
        call_args = self._fix_call_args(messages, model, False, temperature, max_tokens, response_format)

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
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[ResponseFormatType] = None,
    ) -> StreamingSpiceResponse:
        model = self._get_model(model)
        client = self._get_client(model)
        call_args = self._fix_call_args(messages, model, True, temperature, max_tokens, response_format)

        with client.catch_and_convert_errors():
            stream = await client.get_chat_completion_or_stream(call_args)
        stream = cast(AsyncIterator, stream)
        return StreamingSpiceResponse(call_args, client, stream)
