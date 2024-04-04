from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, ContextManager, Dict, List, Optional

import anthropic
import openai
from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam, MessageStreamEvent
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from spice.errors import APIConnectionError, AuthenticationError, SpiceError

if TYPE_CHECKING:
    from spice.spice import SpiceCallArgs


class WrappedClient(ABC):
    @abstractmethod
    async def get_chat_completion_or_stream(
        self, call_args: SpiceCallArgs
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk] | Message | AsyncIterator[MessageStreamEvent]: ...

    @abstractmethod
    def process_chunk(self, chunk) -> tuple[Optional[str], Optional[int], Optional[int]]: ...

    @abstractmethod
    def extract_text(self, chat_completion) -> str: ...

    @abstractmethod
    def get_input_and_output_tokens(self, chat_completion) -> tuple[int, int]: ...

    @abstractmethod
    def catch_and_convert_errors(self) -> ContextManager[None]: ...


class WrappedOpenAIClient(WrappedClient):
    def __init__(self, key, base_url=None):
        self._client = AsyncOpenAI(api_key=key, base_url=base_url)

    async def get_chat_completion_or_stream(self, call_args: SpiceCallArgs):
        # WrappedOpenAIClient can be used with a proxy to a non openai llm, which may not support response_format
        maybe_response_format_kwargs: Dict[str, Any] = (
            {"response_format": call_args.response_format} if call_args.response_format is not None else {}
        )

        # GPT-4-vision has low default max_tokens
        if call_args.max_tokens is None and call_args.model == "gpt-4-vision-preview":
            max_tokens = 4096
        else:
            max_tokens = call_args.max_tokens

        return await self._client.chat.completions.create(
            model=call_args.model,
            messages=call_args.messages,
            stream=call_args.stream,
            temperature=call_args.temperature,
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

    @contextmanager
    def catch_and_convert_errors(self):
        try:
            yield
        except openai.APIConnectionError as e:
            raise APIConnectionError(f"OpenAI Error: {e.message}") from e
        except openai.AuthenticationError as e:
            raise AuthenticationError(f"OpenAI Error: {e.message}") from e


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

    async def get_chat_completion_or_stream(self, call_args: SpiceCallArgs):
        if call_args.messages[0]["role"] == "system":
            system = call_args.messages[0]["content"]
            messages = call_args.messages[1:]
        else:
            system = ""
            messages = call_args.messages

        if call_args.response_format is not None:
            raise SpiceError("response_format is not supported by Anthropic")

        # max_tokens is required by anthropic api
        if call_args.max_tokens is None:
            max_tokens = 4096
        else:
            max_tokens = call_args.max_tokens

        # temperature is optional but can't be None
        maybe_temperature_kwargs: dict[str, Any] = (
            {"temperature": call_args.temperature} if call_args.temperature is not None else {}
        )

        # TODO:
        # convert messages to anthropic format (images and system messages are handled differently than OpenAI, whose format we use)
        converted_messages: List[MessageParam] = []
        for message in messages:
            pass

        return await self._client.messages.create(
            model=call_args.model,
            system=system,
            messages=messages,  # pyright: ignore TODO: Convert the messages
            stream=call_args.stream,
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

    @contextmanager
    def catch_and_convert_errors(self):
        try:
            yield
        except anthropic.APIConnectionError as e:
            raise APIConnectionError(f"Anthropic Error: {e.message}") from e
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(f"Anthropic Error: {e.message}") from e
