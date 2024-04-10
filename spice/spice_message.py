import base64
import mimetypes
from collections import UserList
from pathlib import Path
from typing import Iterable, Optional

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from spice.errors import ImageError

SpiceMessage = ChatCompletionMessageParam
"""The message format Spice uses and accepts. Exactly the same as OpenAI's message format."""

VALID_MIMETYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]


def user_message(content: str) -> ChatCompletionUserMessageParam:
    """Creates a user message with the given content."""
    return {"role": "user", "content": content}


def system_message(content: str) -> ChatCompletionSystemMessageParam:
    """Creates a system message with the given content."""
    return {"role": "system", "content": content}


def assistant_message(content: str) -> ChatCompletionAssistantMessageParam:
    """Creates an assistant message with the given content."""
    return {"role": "assistant", "content": content}


def image_bytes_message(image_bytes: bytes, media_type: str) -> ChatCompletionUserMessageParam:
    """Creates a user message containing the given image bytes."""
    if media_type not in VALID_MIMETYPES:
        raise ImageError(f"Invalid image type {media_type}: Image must be a png, jpg, gif, or webp image.")

    image = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image}"}}],
    }


def file_image_message(file_path: Path | str) -> ChatCompletionUserMessageParam:
    """Creates a user message with the image at the given path. The image must be a png, jpg, gif, or webp image."""

    file_path = Path(file_path).expanduser().resolve()
    if not file_path.exists():
        raise ImageError(f"Invalid image at {file_path}: file does not exist.")

    media_type = mimetypes.guess_type(file_path)[0]
    if media_type not in VALID_MIMETYPES:
        raise ImageError(f"Invalid image at {file_path}: Image must be a png, jpg, gif, or webp image.")

    with file_path.open("rb") as file:
        image_bytes = file.read()

    return image_bytes_message(image_bytes, media_type)


def http_image_message(url: str) -> ChatCompletionUserMessageParam:
    """Creates a user message with an image from the given url."""
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ImageError(f"Invalid image URL {url}: Must be http or https protocol.")

    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}


class SpiceMessages(UserList[SpiceMessage]):
    """
    A light wrapper around the messages list simplifing adding new messages.
    """

    def __init__(self, initlist: Optional[Iterable[SpiceMessage]] = None):
        super().__init__(initlist)

    def add_user_message(self, content: str):
        """Appends a user message with the given content."""
        self.data.append(user_message(content))

    def add_system_message(self, content: str):
        """Appends a system message with the given content."""
        self.data.append(system_message(content))

    def add_assistant_message(self, content: str):
        """Appends an assistant message with the given content."""
        self.data.append(assistant_message(content))

    def add_image_bytes_message(self, image_bytes: bytes, media_type: str):
        """Appends a user message with the given image bytes."""
        self.data.append(image_bytes_message(image_bytes, media_type))

    def add_file_image_message(self, file_path: Path | str):
        """Appends a user message with the image from the given file. The image must be a png, jpg, gif, or webp image."""
        self.data.append(file_image_message(file_path))

    def add_http_image_message(self, url: str):
        """Appends a user message with the image from the given url."""
        self.data.append(http_image_message(url))
