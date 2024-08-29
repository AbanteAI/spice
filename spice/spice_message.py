from __future__ import annotations

import base64
import mimetypes
from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from spice.errors import ImageError

if TYPE_CHECKING:
    from spice.spice import Spice

VALID_MIMETYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]

Role = Literal["user", "assistant", "system"]


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: str


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class PromptMetadata(BaseModel):
    name: str
    content: str
    context: Dict[str, Any]


class SpiceMessage(BaseModel):
    role: Role
    content: Union[TextContent, ImageContent]
    cache: bool = Field(default=False)
    prompt_metadata: Optional[PromptMetadata]

    def __init__(
        self,
        role: Role,
        text: Optional[str] = None,
        image_url: Optional[str] = None,
        cache: bool = False,
        prompt_metadata: Optional[PromptMetadata] = None,
    ):
        if text is not None and image_url is not None:
            raise ValueError("Only one of text or image_url can be provided.")
        if text is not None:
            content = TextContent(type="text", text=text)
        elif image_url is not None:
            content = ImageContent(type="image_url", image_url=image_url)
        else:
            raise ValueError("Either text or image_url must be provided.")
        super().__init__(role=role, content=content, cache=cache, prompt_metadata=prompt_metadata)


class SpiceMessages(List[SpiceMessage]):
    """A collection of messages to be sent to an API endpoint."""

    def __init__(self, client: Optional[Spice] = None, messages: Collection[SpiceMessage] = []):
        self._client = client
        self.data: list[SpiceMessage] = [message for message in messages]

    def add_text(self, role: Role, text: str, cache: bool = False) -> SpiceMessages:
        self.data.append(SpiceMessage(role=role, text=text, cache=cache))
        return self

    def add_user_text(self, text: str, cache: bool = False) -> SpiceMessages:
        return self.add_text("user", text, cache)

    def add_system_text(self, text: str, cache: bool = False) -> SpiceMessages:
        return self.add_text("system", text, cache)

    def add_assistant_text(self, text: str, cache: bool = False) -> SpiceMessages:
        return self.add_text("assistant", text, cache)

    def add_user_image_from_url(self, url: str, cache: bool = False) -> SpiceMessages:
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ImageError(f"Invalid image URL {url}: Must be http or https protocol.")
        self.data.append(SpiceMessage(role="user", image_url=url, cache=cache))
        return self

    def add_user_image_from_file(self, file_path: Path | str, cache: bool = False) -> SpiceMessages:
        file_path = Path(file_path).expanduser().resolve()
        if not file_path.exists():
            raise ImageError(f"Invalid image at {file_path}: file does not exist.")
        media_type = mimetypes.guess_type(file_path)[0]
        if media_type not in VALID_MIMETYPES:
            raise ImageError(f"Invalid image type {media_type}: Image must be one of {VALID_MIMETYPES}.")
        with file_path.open("rb") as file:
            image_bytes = file.read()
        image = base64.b64encode(image_bytes).decode("utf-8")
        self.data.append(SpiceMessage(role="user", image_url=f"data:{media_type};base64,{image}", cache=cache))
        return self

    def add_prompt(self, role: Role, name: str, cache: bool = False, **context: Any) -> SpiceMessages:
        """Appends a message with the given role and pre-loaded prompt using jinja to render the context."""
        if self._client is None:
            raise ValueError("Cannot add prompt without a Spice client.")

        self.data.append(
            SpiceMessage(
                role=role,
                text=self._client.get_rendered_prompt(name, **context),
                cache=cache,
                prompt_metadata=PromptMetadata(name=name, content=self._client.get_prompt(name), context=context),
            )
        )
        return self

    def add_user_prompt(self, name: str, cache: bool = False, **context: Any) -> SpiceMessages:
        return self.add_prompt("user", name, cache, **context)

    def add_system_prompt(self, name: str, cache: bool = False, **context: Any) -> SpiceMessages:
        return self.add_prompt("system", name, cache, **context)

    def add_assistant_prompt(self, name: str, cache: bool = False, **context: Any) -> SpiceMessages:
        return self.add_prompt("assistant", name, cache, **context)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return item in self.data

    def copy(self):
        new_copy = SpiceMessages(self._client)
        new_copy.data = self.data.copy()
        return new_copy
