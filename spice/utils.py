from typing import Optional

from spice.models import TextModel


def text_request_cost(model: TextModel, input_tokens: int, output_tokens: int) -> Optional[float]:
    if model.input_cost is not None and model.output_cost is not None:
        return (model.input_cost * input_tokens + model.output_cost * output_tokens) / 1_000_000
    else:
        return None
