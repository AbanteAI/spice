from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from spice.spice import SpiceCallArgs

T = TypeVar("T")

class Behavior(Enum):
    RETRY = "retry"
    RETURN = "return"

class RetryStrategy(ABC):
    @abstractmethod
    def decide(self, call_args: SpiceCallArgs, attempt_number: int, model_output: str) -> tuple[Behavior, SpiceCallArgs, Any]:
        pass

    @abstractmethod
    def get_retry_name(self, base_name: str, attempt_number: int) -> str:
        pass

class DefaultRetryStrategy(RetryStrategy):
    def __init__(self, validator: Optional[Callable[[str], bool]] = None, converter: Callable[[str], T] = str, retries: int = 0):
        self.validator = validator
        self.converter = converter
        self.retries = retries

    def decide(self, call_args: SpiceCallArgs, attempt_number: int, model_output: str) -> tuple[Behavior, SpiceCallArgs, Any]:
        if attempt_number == 1 and call_args.temperature is not None:
            call_args.temperature = max(0.2, call_args.temperature)
        elif attempt_number > 1 and call_args.temperature is not None:
            call_args.temperature = max(0.5, call_args.temperature)

        if self.validator and not self.validator(model_output):
            if attempt_number < self.retries:
                return Behavior.RETRY, call_args, None
            else:
                raise ValueError("Failed to get a valid response after all retries")
        try:
            result = self.converter(model_output)
            return Behavior.RETURN, call_args, result
        except Exception:
            if attempt_number < self.retries:
                return Behavior.RETRY, call_args, None
            else:
                raise ValueError("Failed to get a valid response after all retries")

    def get_retry_name(self, base_name: str, attempt_number: int) -> str:
        return f"{base_name}-retry-{attempt_number}-fail"