import os

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

_use_anthropic = True


class SpiceClient:
    def __init__(self, model):
        if model == "gpt-4-0125-preview":
            self._provider = "openai"
        elif model == "claude-3-opus-20240229":
            self._provider = "anthropic"
        else:
            raise ValueError(f"Unknown model {model}")

        self.model = model

        load_dotenv()
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
                response = chat_completion_or_stream.content[0].text
            else:
                response = chat_completion_or_stream.choices[0].message.content
            return response

    def _stream_generator(self, stream):
        for chunk in stream:
            if self._provider == "anthropic":
                content = ""
                if chunk.type == "content_block_delta":
                    content = chunk.delta.text
            else:
                content = chunk.choices[0].delta.content
            if content is not None:
                yield content
