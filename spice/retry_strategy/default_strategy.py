import dataclasses
from typing import Any, Callable, Optional

from spice.call_args import SpiceCallArgs
from spice.retry_strategy import Behavior, RetryStrategy, T


class DefaultRetryStrategy(RetryStrategy):
    def __init__(
        self, validator: Optional[Callable[[str], bool]] = None, converter: Callable[[str], T] = str, retries: int = 0
    ):
        self.validator = validator
        self.converter = converter
        self.retries = retries

    def decide(
        self, call_args: SpiceCallArgs, attempt_number: int, model_output: str, name: str
    ) -> tuple[Behavior, SpiceCallArgs, Any, str]:
        if attempt_number == 1 and call_args.temperature is not None:
            call_args = dataclasses.replace(call_args, temperature=max(0.2, call_args.temperature))
        elif attempt_number > 1 and call_args.temperature is not None:
            call_args = dataclasses.replace(call_args, temperature=max(0.5, call_args.temperature))

        if self.validator and not self.validator(model_output):
            if attempt_number < self.retries:
                return Behavior.RETRY, call_args, None, f"{name}-retry-{attempt_number}-fail"
            else:
                raise ValueError("Failed to get a valid response after all retries")
        try:
            result = self.converter(model_output)
            return Behavior.RETURN, call_args, result, name
        except Exception:
            if attempt_number < self.retries:
                return Behavior.RETRY, call_args, None, f"{name}-retry-{attempt_number}-fail"
            else:
                raise ValueError("Failed to get a valid response after all retries")
