# TODO: track timing, usage, cost
# TODO: async

import os
import time
from abc import ABC, abstractmethod

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

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
    time_called: float
    time_first_token: float
    time_end: float


class SpiceResponse:
        self._timing = timing
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
        if self._timing is None:
            raise SpiceError("Timing not set!")
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
        start_time = time.time()
        chat_completion_or_stream = self._client.get_chat_completion_or_stream(
            self.model, system_message, messages, stream
        )

        if stream:
            return self._get_streaming_response(chat_completion_or_stream)
        else:
            end_time = time.time()
            timing = Timing(time_called=start_time, time_first_token=None, time_end=end_time)
            return SpiceResponse(
                text=self._client.extract_text(chat_completion_or_stream),
                usage=chat_completion_or_stream.usage,
                timing=timing,
            )
            )

    def _get_streaming_response(self, stream):
        text_list = []

        def wrapped_stream():
            first_token_time = None
            for chunk in stream:
                content = self._client.process_chunk(chunk)
                if first_token_time is None:
                    first_token_time = time.time()
                text_list.append(content)
                yield content
            response._text = "".join(text_list)
        end_time = time.time()
        timing = Timing(time_called=start_time, time_first_token=first_token_time, time_end=end_time)

        response = SpiceResponse(
            stream=wrapped_stream,
            timing=timing,
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
