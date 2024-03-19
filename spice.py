# TODO: async
# TODO: sync and async examples?

import os
from abc import ABC, abstractmethod
from timeit import default_timer as timer

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


class SpiceError(Exception):
    pass


class SpiceResponse:
    def __init__(self, stream=None, text=None, cost=None, input_tokens=None, output_tokens=None):
        self._stream = stream
        self._text = text
        self._cost = cost
        self._start_time = None
        self._first_token_time = None
        self._end_time = None
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

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
    def __init__(self, model):
        self.model = model

        if "gpt" in self.model:
            self._client = WrappedOpenAIClient()
        elif "claude" in self.model:
            self._client = WrappedAnthropicClient()
        else:
            raise ValueError(f"Unknown model {model}")

    async def call_llm(self, system_message, messages, stream=False):
        start_time = timer()
        chat_completion_or_stream = await self._client.get_chat_completion_or_stream(
            self.model, system_message, messages, stream
        )

        if stream:
            response = await self._get_streaming_response(chat_completion_or_stream)
        else:
            input_tokens, output_tokens = self._client.get_input_and_output_tokens(chat_completion_or_stream)
            response = SpiceResponse(
                text=self._client.extract_text(chat_completion_or_stream),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            response._end_time = timer()

        response._start_time = start_time

        return response

    async def _get_streaming_response(self, stream):
        text_list = []

        async def wrapped_stream():
            async for chunk in stream:
                content, input_tokens, output_tokens = self._client.process_chunk(chunk)
                if input_tokens is not None:
                    response.input_tokens = input_tokens
                if output_tokens is not None:
                    response.output_tokens = output_tokens
                if content is not None:
                    if response._first_token_time is None:
                        response._first_token_time = timer()
                    text_list.append(content)
                    yield content

            response._text = "".join(text_list)
            response._end_time = timer()

        response = SpiceResponse(
            stream=wrapped_stream,
        )

        return response


class WrappedClient(ABC):
    @abstractmethod
    async def get_chat_completion_or_stream(self, model, system_message, messages, stream): ...

    @abstractmethod
    def process_chunk(self, chunk): ...

    @abstractmethod
    def extract_text(self, chat_completion): ...

    @abstractmethod
    def get_input_and_output_tokens(self, chat_completion): ...


class WrappedOpenAIClient(WrappedClient):
    def __init__(self):
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def get_chat_completion_or_stream(self, model, system_message, messages, stream):
        _messages = [
            {
                "role": "system",
                "content": system_message,
            }
        ] + messages
        return await self._client.chat.completions.create(
            messages=_messages,
            model=model,
            temperature=0.3,
            stream=stream,
        )

    def process_chunk(self, chunk):
        content = chunk.choices[0].delta.content
        return content, None, None

    def extract_text(self, chat_completion):
        return chat_completion.choices[0].message.content

    def get_input_and_output_tokens(self, chat_completion):
        usage = chat_completion.usage
        return usage.prompt_tokens, usage.completion_tokens


class WrappedAnthropicClient(WrappedClient):
    def __init__(self):
        self._client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def get_chat_completion_or_stream(self, model, system_message, messages, stream):
        return await self._client.messages.create(
            max_tokens=1024,
            system=system_message,
            messages=messages,
            model=model,
            temperature=0.3,
            stream=stream,
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
