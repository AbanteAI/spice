from dataclasses import dataclass
from typing import Collection, Optional

from openai.types.chat.completion_create_params import ResponseFormat

from spice.spice_message import SpiceMessage


@dataclass
class SpiceCallArgs:
    model: str
    messages: Collection[SpiceMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[ResponseFormat] = None
