from typing import Any, Callable, Tuple

from spice.call_args import SpiceCallArgs
from spice.retry_strategy import Behavior, RetryStrategy
from spice.spice_message import SpiceMessages


def default_failure_message(message: str) -> str:
    return f"Failed to validate response for the following reason: {message}\n\nPlease try again."


class ValidatorStrategy(RetryStrategy):
    """
    Validates the model output and if it fails puts the failure reason in model context.

    """

    def __init__(
        self,
        validator: Callable[[str], Tuple[bool, str]],
        retries: int = 0,
        render_failure_message: Callable[[str], str] = default_failure_message,
    ):
        """
        Args:
            validator: A function that takes in the model output and returns a tuple of a boolean and a message. The
                boolean indicates if the model output is valid and the message is the reason why it is invalid.
            retries: The number of retries to attempt before failing.
            render_failure_message: A function that takes in the failure message and returns a string that will be
                displayed to the llm.
        """
        self.validator = validator
        self.retries = retries
        self.render_failure_message = render_failure_message

    def decide(
        self, call_args: SpiceCallArgs, attempt_number: int, model_output: str, name: str
    ) -> tuple[Behavior, SpiceCallArgs, Any, str]:
        passed, message = self.validator(model_output)
        if not passed:
            if attempt_number < self.retries:
                messages = SpiceMessages(messages=call_args.messages)
                messages.add_assistant_text(model_output)
                messages.add_user_text(self.render_failure_message(message))
                new_call_args = call_args.model_copy(update={"messages": messages})
                return Behavior.RETRY, new_call_args, None, f"{name}-retry-{attempt_number}-fail"
            else:
                raise ValueError("Failed to get a valid response after all retries")
        else:
            return Behavior.RETURN, call_args, model_output, name
