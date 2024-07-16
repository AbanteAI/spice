import dataclasses
from typing import Any, Callable, Optional

from spice.retry_strategy import Behavior, RetryStrategy
from spice.spice import SpiceCallArgs
from spice.spice_message import assistant_message, user_message


def default_failure_message(message: str) -> str:
    return f"Failed to convert response for the following reason: {message}\n\nPlease try again."


class ConverterStrategy(RetryStrategy):
    def __init__(
        self,
        converter: Callable[[str], Any],
        retries: int = 0,
        render_failure_message: Callable[[str], str] = default_failure_message,
    ):
        self.converter = converter
        self.retries = retries
        self.render_failure_message = render_failure_message

    def decide(
        self, call_args: SpiceCallArgs, attempt_number: int, model_output: str, name: str
    ) -> tuple[Behavior, SpiceCallArgs, Any, str]:
        try:
            result = self.converter(model_output)
            return Behavior.RETURN, call_args, result, name
        except Exception as e:
            if attempt_number < self.retries:
                messages = list(call_args.messages)
                messages.append(assistant_message(model_output))
                messages.append(user_message(self.render_failure_message(str(e))))
                call_args = dataclasses.replace(call_args, messages=messages)
                return Behavior.RETRY, call_args, None, f"{name}-retry-{attempt_number}-fail"
            else:
                raise ValueError("Failed to get a valid response after all retries")
