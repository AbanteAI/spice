# TODO: track timing, usage, cost
# TODO: async

import os
from abc import ABC, abstractmethod
from timeit import default_timer as timer

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()


class SpiceError(Exception):
    pass


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


class Timing(BaseModel):
    start_time: float = Field(default=None)
    first_token_time: float = Field(default=None)
    end_time: float = Field(default=None)

    @property
    def time_to_first_token(self):
        return self.first_token_time - self.start_time

    @property
    def total_time(self):
        return self.end_time - self.start_time


class SpiceResponse:
    def __init__(self, stream=None, text=None, cost=None, usage=None):
        self._stream = stream
        self._text = text
        self._cost = cost
        self._usage = usage
        self._timing = Timing()

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
    def timing(self):
        return self._timing


class Spice:
    def __init__(self, model):
        self.model = model

        if "gpt" in self.model:
            self._client = WrappedOpenAIClient()
        elif "claude" in self.model:
            self._client = WrappedAnthropicClient()
        else:
            raise ValueError(f"Unknown model {model}")

    def call_llm(self, system_message, messages, stream=False):
        # TODO: create response here?
        start_time = timer()
        chat_completion_or_stream = self._client.get_chat_completion_or_stream(
            self.model, system_message, messages, stream
        )

        if stream:
            response = self._get_streaming_response(chat_completion_or_stream)
        else:
            response = SpiceResponse(
                text=self._client.extract_text(chat_completion_or_stream),
                usage=chat_completion_or_stream.usage,
            )
            response.timing.end_time = timer()

        response.timing.start_time = start_time

        return response

    def _get_streaming_response(self, stream):
        text_list = []

        def wrapped_stream():
            for chunk in stream:
                content = self._client.process_chunk(chunk)
                if content and response.timing.first_token_time is None:
                    response.timing.first_token_time = timer()
                text_list.append(content)
                yield content
            response._text = "".join(text_list)
            response.timing.end_time = timer()

        response = SpiceResponse(
            stream=wrapped_stream,
        )

        return response


class WrappedClient(ABC):
    @abstractmethod
    def get_chat_completion_or_stream(self, model, system_message, messages, stream):
        pass

    @abstractmethod
    def process_chunk(self, chunk):
        pass

    @abstractmethod
    def extract_text(self, chat_completion):
        pass


class WrappedOpenAIClient(WrappedClient):
    def __init__(self):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_chat_completion_or_stream(self, model, system_message, messages, stream):
        _messages = [
            {
                "role": "system",
                "content": system_message,
            }
        ] + messages
        return self._client.chat.completions.create(
            messages=_messages,
            model=model,
            temperature=0.3,
            stream=stream,
        )

    def process_chunk(self, chunk):
        content = chunk.choices[0].delta.content
        if content is None:
            content = ""
        return content

    def extract_text(self, chat_completion):
        return chat_completion.choices[0].message.content


class WrappedAnthropicClient(WrappedClient):
    def __init__(self):
        self._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def get_chat_completion_or_stream(self, model, system_message, messages, stream):
        return self._client.messages.create(
            max_tokens=1024,
            system=system_message,
            messages=messages,
            model=model,
            temperature=0.3,
            stream=stream,
        )

    def process_chunk(self, chunk):
        content = ""
        if chunk.type == "content_block_delta":
            content = chunk.delta.text
        return content

    def extract_text(self, chat_completion):
        return chat_completion.content[0].text
