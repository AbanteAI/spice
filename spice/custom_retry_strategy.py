from spice.retry_strategy import Behavior, RetryStrategy
from spice.spice import SpiceCallArgs
from typing import Any, Callable, Optional

class AddModelResponseRetryStrategy(RetryStrategy):
    def __init__(self, validator: Optional[Callable[[str], bool]] = None, converter: Callable[[str], Any]] = str, retries: int = 0):
        self.validator = validator
        self.converter = converter
        self.retries = retries

    def decide(self, call_args: SpiceCallArgs, attempt_number: int, model_output: str) -> tuple[Behavior, SpiceCallArgs, Any]:
        if self.validator and not self.validator(model_output):
            if attempt_number < self.retries:
                call_args.messages.append({"role": "system", "content": f"Previous response: {model_output}"})
                return Behavior.RETRY, call_args, None
            else:
                raise ValueError("Failed to get a valid response after all retries")
        try:
            result = self.converter(model_output)
            return Behavior.RETURN, call_args, result
        except Exception as e:
            if attempt_number < self.retries:
                call_args.messages.append({"role": "system", "content": f"Previous response: {model_output}, Exception: {str(e)}"})
                return Behavior.RETRY, call_args, None
            else:
                raise ValueError("Failed to get a valid response after all retries")

    def get_retry_name(self, base_name: str, attempt_number: int) -> str:
        return f"{base_name}-retry-{attempt_number}-fail"