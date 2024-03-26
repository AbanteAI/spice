from dataclasses import dataclass
from timeit import default_timer as timer

from spice.client_manager import (
    _get_client,
    _get_clients_from_env,
    _get_provider_from_model_name,
    _validate_model_aliases,
)
from spice.errors import SpiceError
from spice.utils import count_messages_tokens, count_string_tokens


@dataclass
class SpiceCallArgs:
    model: str
    messages: list[dict]
    stream: bool = False
    temperature: float = None
    max_tokens: int = None
    response_format: dict = None


class SpiceResponse:
    def __init__(self, call_args, logging_callback=None):
        self.call_args = call_args
        self._logging_callback = logging_callback

        self._stream = None
        self._text = None
        self._start_time = timer()
        self._first_token_time = None
        self._end_time = None
        # TODO: should these be _input_tokens and _output_tokens?
        self.input_tokens = None
        self.output_tokens = None

    def finalize(self, text, input_tokens, output_tokens):
        self._end_time = timer()
        self._text = text
        # TODO: this message counting methods are just for OpenAI,
        # if other providers also don't send token counts when streaming
        # then we need to add counting methods for them as well
        if input_tokens is None:
            self.input_tokens = count_messages_tokens(self.call_args.messages, self.call_args.model)
        else:
            self.input_tokens = input_tokens
        if output_tokens is None:
            self.output_tokens = count_string_tokens(self.text, self.call_args.model, full_message=False)
        else:
            self.output_tokens = output_tokens
        # TODO: ensure callback is called even if there's an exception or keyboard interrupt
        if self._logging_callback is not None:
            self._logging_callback(self)

    @property
    def stream(self):
        if self._stream is None:
            raise SpiceError("Stream not set! Did you use stream=True?")
        return self._stream

    @property
    def text(self):
        if self._text is None:
            raise SpiceError("Text not set! Did you iterate over the stream?")
        return self._text

    @property
    def time_to_first_token(self):
        if self._stream is None:
            raise SpiceError("Time to first token not tracked for non-streaming responses")
        return self._first_token_time - self._start_time

    @property
    def total_time(self):
        return self._end_time - self._start_time

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    @property
    def characters_per_second(self):
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
        messages,
        model=None,
        stream=False,
        temperature=None,
        max_tokens=None,
        response_format=None,
        logging_callback=None,
    ):
        if model is None:
            if self._default_model is None:
                raise SpiceError("model argument is required when default model is not set at initialization")
            model = self._default_model

        if self._model_aliases is not None:
            if model in self._model_aliases:
                model = self._model_aliases[model]["model"]
            else:
                raise SpiceError(f"Unknown model alias: {model}")

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

        chat_completion_or_stream = await client.get_chat_completion_or_stream(
            model, messages, stream, temperature, max_tokens, response_format
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

    async def _get_streaming_response(self, client, stream, response):
        text_list = []

        async def wrapped_stream():
            input_tokens = None
            output_tokens = None

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

            response.finalize(
                text="".join(text_list),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        response._stream = wrapped_stream
        return response
