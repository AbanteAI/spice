from __future__ import annotations

import base64
import io
import mimetypes
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Collection, ContextManager, Dict, List, Optional, Tuple

import anthropic
import httpx
import openai
import tiktoken
from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam, MessageStreamEvent, TextBlockParam
from openai import AsyncAzureOpenAI, AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from PIL import Image
from typing_extensions import override

from spice.errors import APIConnectionError, APIError, AuthenticationError, ImageError, InvalidModelError
from spice.spice_message import VALID_MIMETYPES, SpiceMessage

if TYPE_CHECKING:
    from spice.models import Model
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


class WrappedOpenAIClient(WrappedClient):
    def __init__(self, key, base_url=None):
        self._sync_client = OpenAI(api_key=key, base_url=base_url)
        self._client = AsyncOpenAI(api_key=key, base_url=base_url)

    @override
    async def get_chat_completion_or_stream(self, call_args: SpiceCallArgs):
        # WrappedOpenAIClient can be used with a proxy to a non openai llm, which may not support response_format
        maybe_response_format_kwargs: Dict[str, Any] = (
            {"response_format": call_args.response_format}
            if call_args.response_format is not None and "type" in call_args.response_format
            else {}
        )

        # GPT-4-vision has low default max_tokens
        if call_args.max_tokens is None and "gpt-4" in call_args.model and "vision-preview" in call_args.model:
            max_tokens = 4096
        else:
            max_tokens = call_args.max_tokens

        return await self._client.chat.completions.create(
            model=call_args.model,
            messages=list(call_args.messages),
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


class WrappedAnthropicClient(WrappedClient):
    def __init__(self, key):
        self._client = AsyncAnthropic(api_key=key)

    def _convert_messages(
        self, messages: Collection[SpiceMessage], add_json_brace: bool
    ) -> Tuple[str, List[MessageParam]]:
        # Anthropic handles both images and system messages different from OpenAI, only allows alternating user / assistant messages,
        # and doesn't support tools / function calling (still in beta, and doesn't support streaming)

        system = ""
        converted_messages: List[MessageParam] = []
        start = True
        cur_role = ""
        for message in messages:
            if message["role"] == "system" and start:
                if system:
                    system += "\n\n"
                system += message["content"]
                continue

            # First message must be user, and text content cannot be empty / whitespace only
            if start and message["role"] != "user":
                cur_role = "user"
                converted_messages.append({"role": "user", "content": [{"type": "text", "text": "-"}]})

            start = False

            # Anthropic messages can either be a string or list of objects; since user messages can have images, they should always be a list objects.
            # Assistant messages should always be strings to keep them simple.
            match message["role"]:
                case "system":
                    if cur_role == "user":
                        message_object: TextBlockParam = {"type": "text", "text": f"\n\nSystem:\n{message['content']}"}
                        converted_messages[-1]["content"].append(message_object)  # pyright: ignore
                    else:
                        cur_role = "user"
                        message_object: TextBlockParam = {"type": "text", "text": f"System:\n{message['content']}"}
                        converted_messages.append({"role": "user", "content": [message_object]})
                case "assistant":
                    content = message.get("content", "")
                    if content is None:
                        content = ""
                    if cur_role == "assistant":
                        converted_messages[-1]["content"] += f"\n\n{content}"  # pyright: ignore
                    else:
                        cur_role = "assistant"
                        converted_messages.append({"role": "assistant", "content": content})
                case "user":
                    if isinstance(message["content"], str):
                        if cur_role == "user":
                            converted_messages[-1]["content"].append(  # pyright: ignore
                                {"type": "text", "text": f"\n\n{message['content']}"}
                            )
                        else:
                            converted_messages.append(
                                {"role": "user", "content": [{"type": "text", "text": f"{message['content']}"}]}
                            )
                    else:
                        first = cur_role == "user"
                        content = []
                        for sub_content in message["content"]:
                            if sub_content["type"] == "text":
                                content.append(
                                    {"type": "text", "text": ("\n\n" if first else "") + sub_content["text"]}
                                )
                            else:
                                # This can either be base64 encoded data or a url; Anthropic only accepts base64 encoded data
                                image = sub_content["image_url"]["url"]
                                if image.startswith("http"):
                                    try:
                                        response = httpx.get(image)
                                    except:
                                        raise ImageError(f"Error fetching image {image}.")

                                    media_type = response.headers.get("content-type", mimetypes.guess_type(image)[0])
                                    image = base64.b64encode(response.content).decode("utf-8")
                                else:
                                    media_type = image.split(";", maxsplit=1)[0].replace("data:", "")
                                    image = image.split(";base64,", maxsplit=1)[1]

                                if media_type not in VALID_MIMETYPES:
                                    raise ImageError(
                                        f"Invalid image at {image}: Image must be a png, jpg, gif, or webp image."
                                    )

                                content.append(
                                    {
                                        "type": "image",
                                        "source": {"type": "base64", "data": image, "media_type": media_type},
                                    }
                                )

                            first = False
                        if cur_role == "user":
                            converted_messages[-1]["content"].extend(content)  # pyright: ignore
                        else:
                            converted_messages.append({"role": "user", "content": content})

                    cur_role = "user"
                case "tool":
                    # Right now anthropic tool use is in beta and doesn't support some things like streaming, but once it releases we can modify this.
                    pass
                case "function":
                    # Deprecated, nobody should use this
                    pass

        if add_json_brace and converted_messages:
            if converted_messages[-1]["role"] == "assistant" and not converted_messages[-1]["content"]:
                converted_messages[-1]["content"] += "{"  # pyright: ignore
            elif converted_messages[-1]["role"] == "user":
                converted_messages.append({"role": "assistant", "content": "{"})

        return system, converted_messages

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
            messages=converted_messages,
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
