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

    def __init__(self, stream_generator=None, full_text=None, cost=None, usage=None):
        self._stream_generator = stream_generator
        self._full_text = full_text
        self._cost = cost
        self._usage = usage

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
        self.model = model

        if self.model == "gpt-4-0125-preview":
            self._provider = "openai"
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.model == "claude-3-opus-20240229":
            self._provider = "anthropic"
            self._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unknown model {model}")

    def call_llm(self, system_message, messages, stream=False):
        # just route to 4 separate functions? how to handle other models better than that?
        # make the SpiceResponse here, then send to functions to get filled based on provider / stream handlers?

        if self._provider == "anthropic":
            return _call_llm_anthropic(
                self._client, self.model, system_message, messages, stream
            )
        else:
            return _call_llm_openai(
                self._client, self.model, system_message, messages, stream
            )


def _call_llm_anthropic(client, model, system_message, messages, stream):
    chat_completion_or_stream = client.messages.create(
        max_tokens=1024,
        system=system_message,
        messages=messages,
        model=model,
        temperature=0.3,
        stream=stream,
    )

    if stream:
        return _get_streaming_response_anthropic(chat_completion_or_stream)
    else:
        return SpiceResponse(
            full_text=chat_completion_or_stream.content[0].text,
            cost=None,
            usage=chat_completion_or_stream.usage,
        )


def _get_streaming_response_anthropic(stream):
    full_text_list = []

    def wrapped_stream():
        for chunk in stream:
            content = ""
            if chunk.type == "content_block_delta":
                content = chunk.delta.text
            full_text_list.append(content)
            yield content
        response._full_text = "".join(full_text_list)

    response = SpiceResponse(
        stream_generator=wrapped_stream,
    )

    return response


def _call_llm_openai(client, model, system_message, messages, stream):
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
        return _get_streaming_response_openai(chat_completion_or_stream)
    else:
        return SpiceResponse(
            full_text=chat_completion_or_stream.choices[0].message.content,
            usage=chat_completion_or_stream.usage,
        )


def _get_streaming_response_openai(stream):
    full_text_list = []

    def wrapped_stream():
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is None:
                content = ""
            full_text_list.append(content)
            yield content
        response._full_text = "".join(full_text_list)

    response = SpiceResponse(
        stream_generator=wrapped_stream,
    )

    return response
