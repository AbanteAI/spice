from __future__ import annotations

import base64
import io
import mimetypes
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Collection,
    ContextManager,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import anthropic
import httpx
import openai
import tiktoken
from anthropic import AsyncAnthropic
from anthropic.types import ImageBlockParam, Message, MessageStreamEvent, TextBlockParam
from openai import AsyncAzureOpenAI, AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
)
from PIL import Image
from pydantic import BaseModel
from typing_extensions import override

from spice.call_args import SpiceCallArgs
from spice.errors import APIConnectionError, APIError, AuthenticationError, ImageError, InvalidModelError
from spice.spice_message import SpiceMessage

if TYPE_CHECKING:
    from spice.models import Model


# a MessageParam with more constrained structure
class ConstrainedAnthropicMessageParam(TypedDict):
    content: List[Union[TextBlockParam, ImageBlockParam]]
    role: Literal["user", "assistant"]


class TextAndTokens(BaseModel):
    text: Optional[str] = None
    input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class WrappedClient(ABC):
    @abstractmethod
    async def get_chat_completion_or_stream(
        self, call_args: SpiceCallArgs
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk] | Message | AsyncIterator[MessageStreamEvent]: ...

    @abstractmethod
    def process_chunk(self, chunk, call_args: SpiceCallArgs) -> TextAndTokens: ...

    @abstractmethod
    def extract_text_and_tokens(self, chat_completion, call_args: SpiceCallArgs) -> TextAndTokens: ...

    @abstractmethod
    def catch_and_convert_errors(self) -> ContextManager[None]: ...

    @abstractmethod
    def count_messages_tokens(self, messages: Collection[SpiceMessage], model: Model | str) -> int:
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
    async def get_embeddings(self, input_texts: List[str], model: str) -> List[List[float]]: ...

    @abstractmethod
    def get_embeddings_sync(self, input_texts: List[str], model: str) -> List[List[float]]: ...

    @abstractmethod
    async def get_transcription(self, audio_path: Path, model: str) -> tuple[str, float]: ...


def _spice_message_to_openai_content_part(
    message: SpiceMessage,
) -> Union[ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam]:
    if message.content.type == "text":
        return {"type": "text", "text": message.content.text}
    elif message.content.type == "image_url":
        return {"type": "image_url", "image_url": {"url": message.content.image_url}}
    else:
        raise ValueError(f"Unknown content type: {message.content.type}")


class WrappedOpenAIClient(WrappedClient):
    def __init__(self, key, base_url=None):
        self._sync_client = OpenAI(api_key=key, base_url=base_url)
        self._client = AsyncOpenAI(api_key=key, base_url=base_url)

    def _convert_messages(self, messages: Collection[SpiceMessage]) -> List[ChatCompletionMessageParam]:
        converted_messages = []
        for message in messages:
            content_part = _spice_message_to_openai_content_part(message)
            if converted_messages and converted_messages[-1]["role"] == message.role:
                converted_messages[-1]["content"].append(content_part)
            else:
                converted_messages.append({"role": message.role, "content": [content_part]})
        return converted_messages

    @override
    async def get_chat_completion_or_stream(self, call_args: SpiceCallArgs):
        # WrappedOpenAIClient can be used with a proxy to a non openai llm, which may not support response_format
        maybe_kwargs: Dict[str, Any] = {}
        if call_args.response_format is not None and "type" in call_args.response_format:
            maybe_kwargs["response_format"] = call_args.response_format
        if call_args.stream:
            maybe_kwargs["stream_options"] = {"include_usage": True}

        # If using vision you have to set max_tokens or api errors
        if call_args.max_tokens is None and "gpt-4" in call_args.model:
            max_tokens = 4096
        else:
            max_tokens = call_args.max_tokens

        converted_messages = self._convert_messages(call_args.messages)

        return await self._client.chat.completions.create(
            model=call_args.model,
            messages=converted_messages,
            stream=call_args.stream,
            temperature=call_args.temperature,
            max_tokens=max_tokens,
            **maybe_kwargs,
        )

    @override
    def process_chunk(self, chunk, call_args: SpiceCallArgs):
        chunk = cast(ChatCompletionChunk, chunk)
        content = None
        if len(chunk.choices) > 0:
            content = chunk.choices[0].delta.content
        input_tokens = None
        output_tokens = None
        if chunk.usage is not None:
            input_tokens = chunk.usage.prompt_tokens
            output_tokens = chunk.usage.completion_tokens
        return TextAndTokens(text=content, input_tokens=input_tokens, output_tokens=output_tokens)

    @override
    def extract_text_and_tokens(self, chat_completion, call_args: SpiceCallArgs):
        return TextAndTokens(
            text=chat_completion.choices[0].message.content,
            input_tokens=chat_completion.usage.prompt_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            output_tokens=chat_completion.usage.completion_tokens,
        )

    @override
    @contextmanager
    def catch_and_convert_errors(self):
        # TODO: Do we catch all errors? I think we should catch APIStatusError
        try:
            yield
        except openai.APIConnectionError as e:
            raise APIConnectionError(f"OpenAI Connection Error: {e.message}") from e
        except openai.AuthenticationError as e:
            raise AuthenticationError(f"OpenAI Authentication Error: {e.message}") from e
        except openai.APIStatusError as e:
            raise APIError(f"OpenAI Status Error: {e.message}") from e

    def _get_encoding_for_model(self, model: Model | str) -> tiktoken.Encoding:
        from spice.models import Model

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
    def count_messages_tokens(self, messages: Collection[SpiceMessage], model: Model | str) -> int:
        """
        Adapted from https://platform.openai.com/docs/guides/text-generation/managing-tokens
        """

        encoding = self._get_encoding_for_model(model)

        num_tokens = 0
        for message in messages:
            # every message follows <|start|>{role/name}\n{content}<|end|>\n
            # this has 5 tokens (start token, role, \n, end token, \n), but we count the role token later
            num_tokens += 4
            content = message.content
            if content.type == "text":
                num_tokens += len(encoding.encode(content.text))
            elif content.type == "image_url":
                image_base64: str = content.image_url.split(",")[1]
                image_bytes: bytes = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_bytes))
                size = image.size
                # As described here: https://platform.openai.com/docs/guides/vision/calculating-costs
                scale = min(1, 2048 / max(size))
                size = (int(size[0] * scale), int(size[1] * scale))
                scale = min(1, 768 / min(size))
                size = (int(size[0] * scale), int(size[1] * scale))
                num_tokens += 85 + 170 * ((size[0] + 511) // 512) * ((size[1] + 511) // 512)
            else:
                raise ValueError(f"Unknown content type: {content.type}")
        num_tokens += 2  # every reply is primed with <|start|>assistant
        return num_tokens

    @override
    def count_string_tokens(self, message: str, model: Model | str, full_message: bool) -> int:
        encoding = self._get_encoding_for_model(model)
        return len(encoding.encode(message, disallowed_special=())) + (4 if full_message else 0)

    @override
    async def get_embeddings(self, input_texts: List[str], model: str) -> List[List[float]]:
        embeddings = (await self._client.embeddings.create(input=input_texts, model=model)).data
        sorted_embeddings = sorted(embeddings, key=lambda e: e.index)
        return [result.embedding for result in sorted_embeddings]

    @override
    def get_embeddings_sync(self, input_texts: List[str], model: str) -> List[List[float]]:
        embeddings = self._sync_client.embeddings.create(input=input_texts, model=model).data
        sorted_embeddings = sorted(embeddings, key=lambda e: e.index)
        return [result.embedding for result in sorted_embeddings]

    @override
    async def get_transcription(self, audio_path: Path, model: str) -> tuple[str, float]:
        audio_file = open(audio_path, "rb")
        transcript = await self._client.audio.transcriptions.create(
            model=model, file=audio_file, response_format="verbose_json"
        )
        return (transcript.text, transcript.duration)  # pyright: ignore


class WrappedAzureClient(WrappedOpenAIClient):
    def __init__(self, key, endpoint):
        self._client = AsyncAzureOpenAI(
            api_key=key,
            api_version="2023-12-01-preview",
            azure_endpoint=endpoint,
        )

    @override
    def process_chunk(self, chunk, call_args: SpiceCallArgs):
        # In Azure, the first chunk only contains moderation metadata, and an empty choices array
        if not chunk.choices:
            return TextAndTokens()
        else:
            return super().process_chunk(chunk, call_args)


def _spice_message_to_anthropic_block_param(message: SpiceMessage) -> Union[TextBlockParam, ImageBlockParam]:
    if message.content.type == "text":
        block_param = {"type": "text", "text": message.content.text}
    elif message.content.type == "image_url":
        image_url = message.content.image_url
        if image_url.startswith("http"):
            try:
                response = httpx.get(image_url)
            except Exception:
                raise ImageError(f"Error fetching image {image_url}.")
            media_type = response.headers.get("content-type", mimetypes.guess_type(image_url)[0])
            image = base64.b64encode(response.content).decode("utf-8")
        else:
            media_type = image_url.split(";", maxsplit=1)[0].replace("data:", "")
            image = image_url.split(";base64,", maxsplit=1)[1]
        block_param = {
            "type": "image",
            "source": {"type": "base64", "data": image, "media_type": media_type},
        }
    # ignoring types because cache_control is still in beta
    if message.cache:
        block_param["cache_control"] = {"type": "ephemeral"}  # type: ignore
    return block_param  # type: ignore


class WrappedAnthropicClient(WrappedClient):
    def __init__(self, key):
        self._client = AsyncAnthropic(api_key=key)

    def _convert_messages(
        self, messages: Collection[SpiceMessage], add_json_brace: bool
    ) -> Tuple[List[TextBlockParam], List[ConstrainedAnthropicMessageParam]]:
        system_block_params: List[TextBlockParam] = []
        converted_messages: List[ConstrainedAnthropicMessageParam] = []

        for message in messages:
            if message.role == "system":
                if converted_messages:
                    raise ValueError("System messages must be at the start of the conversation.")
                if message.content.type != "text":
                    raise ValueError("System messages must be text.")
                system_block_params.append(_spice_message_to_anthropic_block_param(message))  # type: ignore
            else:
                block_param = _spice_message_to_anthropic_block_param(message)
                if converted_messages and converted_messages[-1]["role"] == message.role:
                    converted_messages[-1]["content"].append(block_param)
                else:
                    converted_messages.append({"role": message.role, "content": [block_param]})

        if add_json_brace:
            if not converted_messages or converted_messages[-1]["role"] != "assistant":
                converted_messages.append({"role": "assistant", "content": []})
            converted_messages[-1]["content"].append({"type": "text", "text": "{"})

        return system_block_params, converted_messages

    @override
    async def get_chat_completion_or_stream(self, call_args: SpiceCallArgs):
        add_json_brace = (
            call_args.response_format is not None and call_args.response_format.get("type", "text") == "json_object"
        )

        # max_tokens is required by anthropic api
        if call_args.max_tokens is None:
            max_tokens = 4096
        else:
            max_tokens = call_args.max_tokens

        # temperature is optional but can't be None
        maybe_temperature_kwargs: dict[str, Any] = (
            {"temperature": call_args.temperature} if call_args.temperature is not None else {}
        )

        system, converted_messages = self._convert_messages(call_args.messages, add_json_brace)

        return await self._client.messages.create(
            model=call_args.model,
            system=system,
            messages=converted_messages,  # type: ignore
            stream=call_args.stream,
            max_tokens=max_tokens,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            **maybe_temperature_kwargs,
        )

    @override
    def process_chunk(self, chunk, call_args: SpiceCallArgs):
        content = None
        input_tokens = None
        cache_creation_input_tokens = None
        cache_read_input_tokens = None
        output_tokens = None
        if chunk.type == "content_block_delta":
            content = chunk.delta.text
        elif chunk.type == "message_start":
            if call_args.response_format is not None and call_args.response_format.get("type") == "json_object":
                content = "{"
            input_tokens = chunk.message.usage.input_tokens
            cache_creation_input_tokens = chunk.message.usage.cache_creation_input_tokens
            cache_read_input_tokens = chunk.message.usage.cache_read_input_tokens
            output_tokens = chunk.message.usage.output_tokens
        elif chunk.type == "message_delta":
            output_tokens = chunk.usage.output_tokens
        return TextAndTokens(
            text=content,
            input_tokens=input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            output_tokens=output_tokens,
        )

    @override
    def extract_text_and_tokens(self, chat_completion, call_args: SpiceCallArgs):
        add_brace = call_args.response_format is not None and call_args.response_format.get("type") == "json_object"
        return TextAndTokens(
            text=("{" if add_brace else "") + chat_completion.content[0].text,
            input_tokens=chat_completion.usage.input_tokens,
            cache_creation_input_tokens=chat_completion.usage.cache_creation_input_tokens,
            cache_read_input_tokens=chat_completion.usage.cache_read_input_tokens,
            output_tokens=chat_completion.usage.output_tokens,
        )

    @override
    @contextmanager
    def catch_and_convert_errors(self):
        try:
            yield
        except anthropic.APIConnectionError as e:
            raise APIConnectionError(f"Anthropic Connection Error: {e.message}") from e
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(f"Anthropic Authentication Error: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise APIError(f"Anthropic Status Error: {e.message}") from e

    # Anthropic doesn't give us a way to count tokens, so we just use OpenAI's token counting functions and multiply by a pre-determined multiplier
    class _FakeWrappedOpenAIClient(WrappedOpenAIClient):
        def __init__(self):
            pass

    _fake_openai_client = _FakeWrappedOpenAIClient()
    _anthropic_token_multiplier = 1.25

    @override
    def count_messages_tokens(self, messages: Collection[SpiceMessage], model: Model | str) -> int:
        from spice.models import GPT_35_TURBO_0125

        return int(
            self._fake_openai_client.count_messages_tokens(messages, GPT_35_TURBO_0125)
            * self._anthropic_token_multiplier
        )

    @override
    def count_string_tokens(self, message: str, model: Model | str, full_message: bool) -> int:
        from spice.models import GPT_35_TURBO_0125

        return int(
            self._fake_openai_client.count_string_tokens(message, GPT_35_TURBO_0125, full_message)
            * self._anthropic_token_multiplier
        )

    @override
    async def get_embeddings(self, input_texts: List[str], model: str) -> List[List[float]]:
        raise InvalidModelError()

    @override
    def get_embeddings_sync(self, input_texts: List[str], model: str) -> List[List[float]]:
        raise InvalidModelError()

    @override
    async def get_transcription(self, audio_path: Path, model: str) -> tuple[str, float]:
        raise InvalidModelError()
