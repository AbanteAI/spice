import os

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_anthropic_client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
_openai_model = "gpt-4-0125-preview"
_anthropic_model = "claude-3-opus-20240229"

_use_anthropic = True


def call_llm(system_message, messages, stream=False):
    if _use_anthropic:
        chat_completion_or_stream = _anthropic_client.messages.create(
            max_tokens=1024,
            system=system_message,
            messages=messages,
            model=_anthropic_model,
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
        chat_completion_or_stream = _openai_client.chat.completions.create(
            messages=_messages,
            model=_openai_model,
            temperature=0.3,
            stream=stream,
        )

    if stream:
        return _stream_generator(chat_completion_or_stream)
    else:
        if _use_anthropic:
            response = chat_completion_or_stream.content[0].text
        else:
            response = chat_completion_or_stream.choices[0].message.content
        return response


def _stream_generator(stream):
    for chunk in stream:
        if _use_anthropic:
            content = ""
            if chunk.type == "content_block_delta":
                content = chunk.delta.text
        else:
            content = chunk.choices[0].delta.content
        if content is not None:
            yield content
