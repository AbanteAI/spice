from contextlib import contextmanager
from pathlib import Path
from typing import AsyncIterator, Collection, Iterator, List

from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as CompletionChoice
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from typing_extensions import override

from spice.errors import InvalidModelError
from spice.models import Model
from spice.spice import SpiceCallArgs
from spice.spice_message import SpiceMessage
from spice.wrapped_clients import TextAndTokens, WrappedClient


async def convert_string_to_asynciter(
    input_str: str,
    chunk_size: int = 5,
) -> AsyncIterator[ChatCompletionChunk]:
    for i in range(0, len(input_str), chunk_size):
        yield ChatCompletionChunk(
            id="",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content=input_str[i : i + chunk_size],
                    ),
                )
            ],
            created=0,
            model="",
            object="chat.completion.chunk",
        )


class WrappedTestClient(WrappedClient):
    """
    A wrapped client that can be used in tests. Accepts what it should respond with in its constructor.

    Stores all calls to get_chat_completion_or_stream in the calls attribute for testing.
    """

    def __init__(self, response: str | Iterator[str]):
        self.calls = list[SpiceCallArgs]()
        if isinstance(response, str):
            self.response = iter(response)
        else:
            self.response = response

    @override
    async def get_chat_completion_or_stream(
        self, call_args: SpiceCallArgs
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        self.calls.append(call_args)
        response = next(self.response)

        if call_args.stream:
            return convert_string_to_asynciter(response)
        else:
            return ChatCompletion(
                id="",
                choices=[
                    CompletionChoice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(
                            content=response,
                            role="assistant",
                        ),
                    )
                ],
                created=0,
                model="",
                object="chat.completion",
            )

    @override
    @contextmanager
    def catch_and_convert_errors(self):
        yield

    @override
    def count_messages_tokens(self, messages: Collection[SpiceMessage], model: Model | str) -> int:
        return 0

    @override
    def count_string_tokens(self, message: str, model: Model | str, full_message: bool) -> int:
        return 0

    @override
    def process_chunk(self, chunk: ChatCompletionChunk, call_args: SpiceCallArgs) -> TextAndTokens:
        return TextAndTokens(
            text=chunk.choices[0].delta.content or "",
            input_tokens=None,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            output_tokens=None,
        )

    @override
    def extract_text_and_tokens(self, chat_completion, call_args: SpiceCallArgs) -> TextAndTokens:
        return TextAndTokens(
            text=chat_completion.choices[0].message.content,
            input_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            output_tokens=0,
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
