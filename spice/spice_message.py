from __future__ import annotations

import base64
import mimetypes
from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, get_args

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


def _generate_role_specific_methods(cls):
    """Generates add_user_text, add_system_text, etc. methods for SpiceMessages."""
    add_methods = [method for method in dir(cls) if method.startswith("add_")]

    def create_method(original, role):
        def new_method(self, *args, **kwargs):
            return original(self, role, *args, **kwargs)

        return new_method

    for method_name in add_methods:
        original_method = getattr(cls, method_name)
        for role in get_args(Role):
            new_method_name = f"add_{role}_{method_name[4:]}"
            setattr(cls, new_method_name, create_method(original_method, role))

    return cls


@_generate_role_specific_methods
class SpiceMessages(List[SpiceMessage]):
    """A collection of messages to be sent to an API endpoint."""

    def __init__(self, client: Optional[Spice] = None, messages: Collection[SpiceMessage] = []):
        self._client = client
        self.data: list[SpiceMessage] = [message for message in messages]

    def add_text(self, role: Role, text: str, cache: bool = False) -> SpiceMessages:
        """Appends a message with the given role and text."""
        self.data.append(SpiceMessage(role=role, text=text, cache=cache))
        return self

    def add_image_from_url(self, role: Role, url: str, cache: bool = False) -> SpiceMessages:
        """Appends a message with the given role and image from the given url."""
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ImageError(f"Invalid image URL {url}: Must be http or https protocol.")
        self.data.append(SpiceMessage(role=role, image_url=url, cache=cache))
        return self

    def add_image_from_file(self, role: Role, file_path: Path | str, cache: bool = False) -> SpiceMessages:
        """Appends a message with the given role and image from the given file path."""
        file_path = Path(file_path).expanduser().resolve()
        if not file_path.exists():
            raise ImageError(f"Invalid image at {file_path}: file does not exist.")
        media_type = mimetypes.guess_type(file_path)[0]
        if media_type not in VALID_MIMETYPES:
            raise ImageError(f"Invalid image type {media_type}: Image must be one of {VALID_MIMETYPES}.")
        with file_path.open("rb") as file:
            image_bytes = file.read()
        image = base64.b64encode(image_bytes).decode("utf-8")
        self.data.append(SpiceMessage(role=role, image_url=f"data:{media_type};base64,{image}", cache=cache))
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

    if TYPE_CHECKING:

        def __getattribute__(self, name: str) -> Any:
            for role in get_args(Role):
                if name.startswith(f"add_{role}_"):
                    _, suffix = name.split(f"{role}_")
                    return getattr(self, f"add_{suffix}")
