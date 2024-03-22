import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from timeit import default_timer as timer

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI

load_dotenv()


class SpiceError(Exception):
    pass


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
        self._call_args = call_args
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
        self.input_tokens = input_tokens
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


def _get_client(provider):
    if provider is None:
        if os.getenv("anthropic_api_key"):
            provider = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("AZURE_OPENAI_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            provider = "azure"
        else:
            raise SpiceError("No recognized API keys set")
        print(f"Using provider: {provider}")

    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if key is None:
            raise SpiceError("ANTHROPIC_API_KEY not set")
        return WrappedAnthropicClient(key)
    elif provider == "openai":
        base_url = os.getenv("OPENAI_API_BASE")
        key = os.getenv("OPENAI_API_KEY")
        if key is None:
            if base_url:
                key = "dummy_key_base_url"
            else:
                raise SpiceError("OPENAI_API_KEY not set")
        return WrappedOpenAIClient(key, base_url)
    elif provider == "azure":
        key = os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if key is None:
            raise SpiceError("AZURE_OPENAI_KEY not set")
        if endpoint is None:
            raise SpiceError("AZURE_OPENAI_ENDPOINT not set")
        return WrappedAzureClient(key, endpoint)
    else:
        raise SpiceError(f"Unknown provider: {provider}")


class Spice:
    def __init__(self, provider=None):
        self._client = _get_client(provider)

    async def call_llm(
        self,
        model,
        messages,
        stream=False,
        temperature=None,
        max_tokens=None,
        response_format=None,
        logging_callback=None,
    ):
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

        chat_completion_or_stream = await self._client.get_chat_completion_or_stream(
            model, messages, stream, temperature, max_tokens, response_format
        )

        if stream:
            await self._get_streaming_response(chat_completion_or_stream, response)
        else:
            input_tokens, output_tokens = self._client.get_input_and_output_tokens(chat_completion_or_stream)
            response.finalize(
                text=self._client.extract_text(chat_completion_or_stream),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return response

    async def _get_streaming_response(self, stream, response):
        text_list = []
        input_tokens = None
        output_tokens = None

        async def wrapped_stream():
            async for chunk in stream:
                content, _input_tokens, _output_tokens = self._client.process_chunk(chunk)
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


class WrappedClient(ABC):
    @abstractmethod
    async def get_chat_completion_or_stream(
        self, model, messages, stream, temperature, max_tokens, response_format
    ): ...

    @abstractmethod
    def process_chunk(self, chunk): ...

    @abstractmethod
    def extract_text(self, chat_completion): ...

    @abstractmethod
    def get_input_and_output_tokens(self, chat_completion): ...


class WrappedOpenAIClient(WrappedClient):
    def __init__(self, key, base_url=None):
        self._client = AsyncOpenAI(api_key=key, base_url=base_url)

    async def get_chat_completion_or_stream(self, model, messages, stream, temperature, max_tokens, response_format):
        # WrappedOpenAIClient can be used with a proxy to a non openai llm, which may not support response_format
        maybe_response_format_kwargs = {"response_format": response_format} if response_format is not None else {}

        return await self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            **maybe_response_format_kwargs,
        )

    def process_chunk(self, chunk):
        content = chunk.choices[0].delta.content
        return content, None, None

    def extract_text(self, chat_completion):
        return chat_completion.choices[0].message.content

    def get_input_and_output_tokens(self, chat_completion):
        usage = chat_completion.usage
        return usage.prompt_tokens, usage.completion_tokens


class WrappedAzureClient(WrappedOpenAIClient):
    def __init__(self, key, endpoint):
        self._client = AsyncAzureOpenAI(
            api_key=key,
            api_version="2023-12-01-preview",
            azure_endpoint=endpoint,
        )


class WrappedAnthropicClient(WrappedClient):
    def __init__(self, key):
        self._client = AsyncAnthropic(api_key=key)

    async def get_chat_completion_or_stream(self, model, messages, stream, temperature, max_tokens, response_format):
        if messages[0]["role"] == "system":
            system = messages[0]["content"]
            messages = messages[1:]

        if response_format is not None:
            raise SpiceError("response_format not supported by anthropic")

        # max_tokens is required by anthropic api
        if max_tokens is None:
            max_tokens = 4096

        # temperature is optional but can't be None
        maybe_temperature_kwargs = {"temperature": temperature} if temperature is not None else {}

        return await self._client.messages.create(
            model=model,
            system=system,
            messages=messages,
            stream=stream,
            max_tokens=max_tokens,
            **maybe_temperature_kwargs,
        )

    def process_chunk(self, chunk):
        content = None
        input_tokens = None
        output_tokens = None
        if chunk.type == "content_block_delta":
            content = chunk.delta.text
        elif chunk.type == "message_start":
            input_tokens = chunk.message.usage.input_tokens
        elif chunk.type == "message_delta":
            output_tokens = chunk.usage.output_tokens
        return content, input_tokens, output_tokens

    def extract_text(self, chat_completion):
        return chat_completion.content[0].text

    def get_input_and_output_tokens(self, chat_completion):
        return chat_completion.usage.input_tokens, chat_completion.usage.output_tokens
