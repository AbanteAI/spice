import os

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class SpiceError(Exception):
    pass


class SpiceResponse:
    # will be returned by call_llm
    # if stream, response will contain a generator, accessed with .stream()
    # other attributes (cost, complete text response, etc.) will be available as well
    # if stream=False, all attributes will be available immediately
    # if stream=True, will error if trying to access attributes before stream is exhausted

    def __init__(self, stream_generator, full_text=None, cost=None):
        self._stream_generator = stream_generator
        self._full_text = full_text
        self._cost = cost

    def _generator_exhausted(self):
        if self._stream_generator is None:
            return True

        # hmm, not easy to check a generator... maybe we need to wrap it?
        # returning true for now
        return True

    @property
    def stream(self):
        return self._stream_generator

    @property
    def text(self):
        if self._full_text is None:
            raise SpiceError("Full text not set")
        if not self._generator_exhausted():
            raise SpiceError("Cannot access full text until stream is exhausted")
        return self._full_text


class SpiceClient:
    def __init__(self, model):
        if model == "gpt-4-0125-preview":
            self._provider = "openai"
        elif model == "claude-3-opus-20240229":
            self._provider = "anthropic"
        else:
            raise ValueError(f"Unknown model {model}")

        self.model = model

        if self._provider == "openai":
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self._provider == "anthropic":
            self._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def call_llm(self, system_message, messages, stream=False):
        if self._provider == "anthropic":
            chat_completion_or_stream = self._client.messages.create(
                max_tokens=1024,
                system=system_message,
                messages=messages,
                model=self.model,
                temperature=0.3,
                stream=stream,
            )
        else:
            _messages = [
                {
                    "role": "system",
                    "content": system_message,
                }
            ] + messages
            chat_completion_or_stream = self._client.chat.completions.create(
                messages=_messages,
                model=self.model,
                temperature=0.3,
                stream=stream,
            )

        if stream:
            return self._stream_generator(chat_completion_or_stream)
        else:
            if self._provider == "anthropic":
                sr = SpiceResponse(
                    stream_generator=None,
                    full_text=chat_completion_or_stream.content[0].text,
                    cost=None,
                )
            else:
                sr = SpiceResponse(
                    stream_generator=None,
                    full_text=chat_completion_or_stream.choices[0].message.content,
                    cost=None,
                )
            return sr

    def _stream_generator(self, stream):
        for chunk in stream:
            if self._provider == "anthropic":
                content = ""
                if chunk.type == "content_block_delta":
                    content = chunk.delta.text
            else:
                content = chunk.choices[0].delta.content
                if content is None:
                    content = ""
            yield content
