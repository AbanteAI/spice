from __future__ import annotations

from dataclasses import dataclass
from timeit import default_timer as timer
from typing import AsyncIterator, Callable, Optional, cast

from spice.client_manager import (
    _get_client,
    _get_clients_from_env,
    _get_provider_from_model_name,
    _validate_model_aliases,
)
from spice.errors import SpiceError
from spice.utils import count_messages_tokens, count_string_tokens
from spice.wrapped_clients import WrappedClient


@dataclass
class SpiceCallArgs:
    model: str
    messages: list[dict]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[dict] = None


class SpiceResponse:
    def __init__(self, call_args: SpiceCallArgs, logging_callback: Optional[Callable[[SpiceResponse], None]]):
        self.call_args = call_args
        self._logging_callback = logging_callback

        self._stream: Optional[Callable[[], AsyncIterator[str]]] = None
        self._text = None
        self._start_time = timer()
        self._first_token_time: Optional[float] = None
        self._end_time = None
        self._input_tokens = None
        self._output_tokens = None

    def finalize(self, text: str, input_tokens: Optional[int], output_tokens: Optional[int]):
        self._end_time = timer()
        self._text = text
        # TODO: this message counting methods are just for OpenAI,
        # if other providers also don't send token counts when streaming
        # then we need to add counting methods for them as well.
        # TODO: input/output will also be none if streaming is interrupted
        # so we need to identify the provider here to handle correctly
        if input_tokens is None:
            self._input_tokens = count_messages_tokens(self.call_args.messages, self.call_args.model)
        else:
            self._input_tokens = input_tokens
        if output_tokens is None:
            self._output_tokens = count_string_tokens(self.text, self.call_args.model, full_message=False)
        else:
            self._output_tokens = output_tokens
        if self._logging_callback is not None:
            self._logging_callback(self)

    @property
    def stream(self) -> Callable[[], AsyncIterator[str]]:
        if self._stream is None:
            raise SpiceError("Stream not set! Did you use stream=True?")
        return self._stream

    @property
    def text(self) -> str:
        if self._text is None:
            raise SpiceError("Text not set! Did you iterate over the stream?")
        return self._text

    @property
    def time_to_first_token(self) -> float:
        if self._stream is None or self._first_token_time is None:
            raise SpiceError("Time to first token not tracked for non-streaming responses")
        return self._first_token_time - self._start_time

    @property
    def total_time(self) -> float:
        if self._end_time is None:
            raise SpiceError("Total time not tracked! finalize() must be called first.")
        return self._end_time - self._start_time

    @property
    def input_tokens(self) -> int:
        if self._input_tokens is None:
            raise SpiceError("Input tokens not set! finalize() must be called first.")
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        if self._output_tokens is None:
            raise SpiceError("Output tokens not set! finalize() must be called first.")
        return self._output_tokens

    @property
    def total_tokens(self) -> int:
        if self._input_tokens is None or self._output_tokens is None:
            raise SpiceError("Token counts not set! finalize() must be called first.")
        return self._input_tokens + self._output_tokens

    @property
    def characters_per_second(self) -> float:
        return len(self.text) / self.total_time


class Spice:
    def __init__(self, model=None, provider=None, model_aliases=None):
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

    async def call_llm(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict[str, str]] = None,
        logging_callback: Optional[Callable[[SpiceResponse], None]] = None,
    ) -> SpiceResponse:
        if model is None:
            if self._default_model is None:
                raise SpiceError("model argument is required when default model is not set at initialization")
            model = self._default_model

        if self._model_aliases is not None:
            if model in self._model_aliases:
                model = self._model_aliases[model]["model"]
            else:
                raise SpiceError(f"Unknown model alias: {model}")

        model = cast(str, model)

        if self._default_client is not None:
            client = self._default_client
        else:
            provider = _get_provider_from_model_name(model)
            if provider not in self._clients:
                raise SpiceError(f"Provider {provider} is not set up for model {model}")
            client = self._clients[provider]

        # not all providers support response format
        if response_format is not None:
            if response_format == {"type": "text"}:
                response_format = None

        # gpt-4-vision has low default max_tokens
        if max_tokens is None and model == "gpt-4-vision-preview":
            max_tokens = 4096

        response = SpiceResponse(
            call_args=SpiceCallArgs(
                model=model,
                messages=messages,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            ),
            logging_callback=logging_callback,
        )

        with client.catch_and_convert_errors():
            chat_completion_or_stream = await client.get_chat_completion_or_stream(
                model=model,
                messages=messages,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

        if stream:
            await self._get_streaming_response(client, chat_completion_or_stream, response)
        else:
            input_tokens, output_tokens = client.get_input_and_output_tokens(chat_completion_or_stream)
            response.finalize(
                text=client.extract_text(chat_completion_or_stream),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return response

    async def _get_streaming_response(self, client: WrappedClient, stream, response: SpiceResponse) -> SpiceResponse:
        text_list = []

        async def wrapped_stream() -> AsyncIterator[str]:
            input_tokens = None
            output_tokens = None

            try:
                with client.catch_and_convert_errors():
                    async for chunk in stream:
                        content, _input_tokens, _output_tokens = client.process_chunk(chunk)
                        if _input_tokens is not None:
                            input_tokens = _input_tokens
                        if _output_tokens is not None:
                            output_tokens = _output_tokens
                        if content is not None:
                            if response._first_token_time is None:
                                response._first_token_time = timer()
                            text_list.append(content)
                            yield content
            finally:
                # TODO: find better way to do this
                # this finally block is executed even if the stream is interrupted,
                # but it can be delayed - it runs when the reference to the genereator
                # is garbage collected and there is an await in user code
                response.finalize(
                    text="".join(text_list),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

        response._stream = wrapped_stream
        return response
