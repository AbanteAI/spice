import os
from abc import ABC, abstractmethod

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()


class SpiceError(Exception):
    pass


class SpiceResponse:
    def __init__(self, stream=None, text=None, cost=None, usage=None):
        self._stream = stream
        self._text = text
        self._cost = cost
        self._usage = usage

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


class Spice:
    def __init__(self, model):
        self.model = model

        if self.model == "gpt-4-0125-preview":
            self._provider = "openai"
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self._translator = OpenAITranslator()
        elif "claude" in self.model:
            self._provider = "anthropic"
            self._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self._translator = AnthropicTranslator()
        else:
            raise ValueError(f"Unknown model {model}")

    def call_llm(self, system_message, messages, stream=False):
        if self._provider == "anthropic":
            return _call_llm_anthropic(self._client, self.model, system_message, messages, stream, self._translator)
        else:
            return _call_llm_openai(self._client, self.model, system_message, messages, stream, self._translator)


def _call_llm_anthropic(client, model, system_message, messages, stream, translator):
    chat_completion_or_stream = client.messages.create(
        max_tokens=1024,
        system=system_message,
        messages=messages,
        model=model,
        temperature=0.3,
        stream=stream,
    )

    if stream:
        return _get_streaming_response(chat_completion_or_stream, translator)
    else:
        return SpiceResponse(
            text=chat_completion_or_stream.content[0].text,
            cost=None,
            usage=chat_completion_or_stream.usage,
        )


def _call_llm_openai(client, model, system_message, messages, stream, translator):
    _messages = [
        {
            "role": "system",
            "content": system_message,
        }
    ] + messages
    chat_completion_or_stream = client.chat.completions.create(
        messages=_messages,
        model=model,
        temperature=0.3,
        stream=stream,
    )

    if stream:
        return _get_streaming_response(chat_completion_or_stream, translator)
    else:
        return SpiceResponse(
            text=chat_completion_or_stream.choices[0].message.content,
            usage=chat_completion_or_stream.usage,
        )


def _get_streaming_response(stream, translator):
    text_list = []

    def wrapped_stream():
        for chunk in stream:
            content = translator.process_chunk(chunk)
            text_list.append(content)
            yield content
        response._text = "".join(text_list)

    response = SpiceResponse(
        stream=wrapped_stream,
    )

    return response


class APITranslator(ABC):
    @abstractmethod
    def process_chunk(self, chunk):
        pass


class OpenAITranslator(APITranslator):
    def process_chunk(self, chunk):
        content = chunk.choices[0].delta.content
        if content is None:
            content = ""
        return content


class AnthropicTranslator(APITranslator):
    def process_chunk(self, chunk):
        content = ""
        if chunk.type == "content_block_delta":
            content = chunk.delta.text
        return content
