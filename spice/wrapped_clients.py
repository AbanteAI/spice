from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, ContextManager, Dict, List, Optional, Sequence

import anthropic
import openai
import tiktoken
from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam, MessageStreamEvent
from openai import AsyncAzureOpenAI, AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from PIL import Image
from typing_extensions import override

from spice.errors import APIConnectionError, AuthenticationError, InvalidModelError, SpiceError
from spice.models import GPT_35_TURBO_0125, Model
from spice.providers import OPEN_AI
from spice.spice_message import SpiceMessage

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
    def extract_text_and_tokens(self, chat_completion) -> tuple[str, int, int]: ...

    @abstractmethod
    def catch_and_convert_errors(self) -> ContextManager[None]: ...

    @abstractmethod
    def count_messages_tokens(self, messages: List[SpiceMessage], model: Model | str) -> int:
        """
        Returns the number of tokens used by a prompt if it was sent to an API for a chat completion.
        """

    @abstractmethod
    def count_string_tokens(self, message: str, model: Model | str, full_message: bool) -> int:
        """
        Calculates the tokens in this message. Will not be accurate for a prompt.
        Use count_messages_tokens to get the exact amount of tokens for a prompt.
        If full_message is true, will include the extra 4 tokens used in a chat completion by this message
        if this message is part of a prompt. Do not full_message to true for a response.
        """

    @abstractmethod
    async def get_embeddings(self, input_texts: List[str], model: str) -> Sequence[Sequence[float]]: ...

    @abstractmethod
    def get_embeddings_sync(self, input_texts: List[str], model: str) -> Sequence[Sequence[float]]: ...

    @abstractmethod
    async def get_transcription(self, audio_path: Path, model: str) -> str: ...


class WrappedOpenAIClient(WrappedClient):
    def __init__(self, key, base_url=None):
        self._sync_client = OpenAI(api_key=key, base_url=base_url)
        self._client = AsyncOpenAI(api_key=key, base_url=base_url)

    @override
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

    @override
    def process_chunk(self, chunk):
        content = chunk.choices[0].delta.content
        return content, None, None

    @override
    def extract_text_and_tokens(self, chat_completion):
        return (
            chat_completion.choices[0].message.content,
            chat_completion.usage.prompt_tokens,
            chat_completion.usage.completion_tokens,
        )

    @override
    @contextmanager
    def catch_and_convert_errors(self):
        # TODO: Do we catch all errors? I think we should catch APIStatusError
        try:
            yield
        except openai.APIConnectionError as e:
            raise APIConnectionError(f"OpenAI Error: {e.message}") from e
        except openai.AuthenticationError as e:
            raise AuthenticationError(f"OpenAI Error: {e.message}") from e

    def _get_encoding_for_model(self, model: Model | str) -> tiktoken.Encoding:
        if isinstance(model, Model):
            model = model.name
        try:
            # OpenAI fine-tuned models are named `ft:<base model>:<name>:<id>`. If tiktoken
            # can't match the full string, it tries to match on startswith, e.g. 'gpt-4'
            _model = model.split(":")[1] if model.startswith("ft:") else model
            return tiktoken.encoding_for_model(_model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    @override
    def count_messages_tokens(self, messages: List[SpiceMessage], model: Model | str) -> int:
        """
        Adapted from https://platform.openai.com/docs/guides/text-generation/managing-tokens
        """

        encoding = self._get_encoding_for_model(model)

        num_tokens = 0
        for message in messages:
            # every message follows <|start|>{role/name}\n{content}<|end|>\n
            # this has 5 tokens (start token, role, \n, end token, \n), but we count the role token later
            num_tokens += 4
            for key, value in message.items():
                if isinstance(value, list) and key == "content":
                    for entry in value:
                        if entry["type"] == "text":
                            num_tokens += len(encoding.encode(entry["text"]))
                        if entry["type"] == "image_url":
                            image_base64: str = entry["image_url"]["url"].split(",")[1]
                            image_bytes: bytes = base64.b64decode(image_base64)
                            image = Image.open(io.BytesIO(image_bytes))
                            size = image.size
                            # As described here: https://platform.openai.com/docs/guides/vision/calculating-costs
                            scale = min(1, 2048 / max(size))
                            size = (int(size[0] * scale), int(size[1] * scale))
                            scale = min(1, 768 / min(size))
                            size = (int(size[0] * scale), int(size[1] * scale))
                            num_tokens += 85 + 170 * ((size[0] + 511) // 512) * ((size[1] + 511) // 512)
                elif isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens -= 1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <|start|>assistant
        return num_tokens

    @override
    def count_string_tokens(self, message: str, model: Model | str, full_message: bool) -> int:
        encoding = self._get_encoding_for_model(model)
        return len(encoding.encode(message, disallowed_special=())) + (4 if full_message else 0)

    @override
    async def get_embeddings(self, input_texts: List[str], model: str) -> Sequence[Sequence[float]]:
        embeddings = (await self._client.embeddings.create(input=input_texts, model=model)).data
        sorted_embeddings = sorted(embeddings, key=lambda e: e.index)
        return [result.embedding for result in sorted_embeddings]

    @override
    def get_embeddings_sync(self, input_texts: List[str], model: str) -> Sequence[Sequence[float]]:
        embeddings = self._sync_client.embeddings.create(input=input_texts, model=model).data
        sorted_embeddings = sorted(embeddings, key=lambda e: e.index)
        return [result.embedding for result in sorted_embeddings]

    @override
    async def get_transcription(self, audio_path: Path, model: str) -> str:
        audio_file = open(audio_path, "rb")
        transcript = await self._client.audio.transcriptions.create(
            model=model,
            file=audio_file,
        )
        return transcript.text


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

    @override
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

        # TODO: convert messages to anthropic format (images and system messages are handled differently than OpenAI, whose format we use)
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

    @override
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

    @override
    def extract_text_and_tokens(self, chat_completion):
        return (
            chat_completion.content[0].text,
            chat_completion.usage.input_tokens,
            chat_completion.usage.output_tokens,
        )

    @override
    @contextmanager
    def catch_and_convert_errors(self):
        try:
            yield
        except anthropic.APIConnectionError as e:
            raise APIConnectionError(f"Anthropic Error: {e.message}") from e
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(f"Anthropic Error: {e.message}") from e

    # Anthropic doesn't give us a way to count tokens, so we just use OpenAI's token counting functions and multiply by a pre-determined multiplier
    class _FakeWrappedOpenAIClient(WrappedOpenAIClient):
        def __init__(self):
            pass

    _fake_openai_client = _FakeWrappedOpenAIClient()
    # TODO: Figure out a good starting multiplier
    _anthropic_token_multiplier = 1

    @override
    def count_messages_tokens(self, messages: List[SpiceMessage], model: Model | str) -> int:
        return (
            self._fake_openai_client.count_messages_tokens(messages, GPT_35_TURBO_0125)
            * self._anthropic_token_multiplier
        )

    @override
    def count_string_tokens(self, message: str, model: Model | str, full_message: bool) -> int:
        return (
            self._fake_openai_client.count_string_tokens(message, GPT_35_TURBO_0125, full_message)
            * self._anthropic_token_multiplier
        )

    @override
    async def get_embeddings(self, input_texts: List[str], model: str) -> Sequence[Sequence[float]]:
        raise InvalidModelError()

    @override
    def get_embeddings_sync(self, input_texts: List[str], model: str) -> Sequence[Sequence[float]]:
        raise InvalidModelError()

    @override
    async def get_transcription(self, audio_path: Path, model: str) -> str:
        raise InvalidModelError()
