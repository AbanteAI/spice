from typing import List, Optional

from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import BaseModel

from spice.spice_message import SpiceMessage


class SpiceCallArgs(BaseModel):
    model: str
    messages: List[SpiceMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[ResponseFormat] = None
