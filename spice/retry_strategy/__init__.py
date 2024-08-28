from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar

from spice.call_args import SpiceCallArgs

T = TypeVar("T")

# An object that defines how get_response should validate/convert a response. It must implement the decide method.
# It takes in the previous call_args, which attempt number this is, and the model output.
# The method returns a tuple of Behavior, SpiceCallArgs, the result and the name of the run. If
# the Behavior is RETURN, then the result will be returned as the result of the SpiceResponse object. If the Behavior
# is RETRY, then the llm will be called again with the new spice_args. It's up to the Behavior to eventually return
# RETURN or throw an exception.


class Behavior(Enum):
    RETRY = "retry"
    RETURN = "return"


class RetryStrategy(ABC, Generic[T]):
    @abstractmethod
    def decide(
        self, call_args: SpiceCallArgs, attempt_number: int, model_output: str, name: str
    ) -> tuple[Behavior, SpiceCallArgs, T, str]:
        pass
