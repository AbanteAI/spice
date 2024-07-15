from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Collection, Dict, Generic, List, Optional, TypeVar, cast

from openai.types.chat.completion_create_params import ResponseFormat

from spice.spice_message import MessagesEncoder, SpiceMessage


@dataclass
class SpiceCallArgs:
    model: str
    messages: Collection[SpiceMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[ResponseFormat] = None
