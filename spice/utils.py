import base64
import io

import tiktoken
from PIL import Image

from spice.spice_message import SpiceMessage


def fuzzy_model_lookup(model_hint):
    model_hint = str(model_hint).lower()

    models_list = [
        "gpt-4-0125-preview",
        "gpt-3.5-turbo-0125",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]

    for model in models_list:
        if model_hint in model:
            print(f"fuzzy matched: {model}")
            return model


def get_encoding_for_model(model: str) -> tiktoken.Encoding:
    try:
        # OpenAI fine-tuned models are named `ft:<base model>:<name>:<id>`. If tiktoken
        # can't match the full string, it tries to match on startswith, e.g. 'gpt-4'
        _model = model.split(":")[1] if model.startswith("ft:") else model
        return tiktoken.encoding_for_model(_model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_string_tokens(message: str, model: str, full_message: bool) -> int:
    """
    Calculates the tokens in this message. Will NOT be accurate for a full prompt!
    Use count_messages_tokens to get the exact amount of tokens for a prompt.
    If full_message is true, will include the extra 4 tokens used in a chat completion by this message
    if this message is part of a prompt. You do NOT want full_message to be true for a response.
    """
    encoding = get_encoding_for_model(model)
    return len(encoding.encode(message, disallowed_special=())) + (4 if full_message else 0)


def count_messages_tokens(messages: list[SpiceMessage], model: str):
    """
    Returns the number of tokens used by a prompt if it was sent to OpenAI for a chat completion.
    Adapted from https://platform.openai.com/docs/guides/text-generation/managing-tokens
    """
    encoding = get_encoding_for_model(model)

    num_tokens = 0
    for message in messages:
        # every message follows <|start|>{role/name}\n{content}<|end|>\n
        # this has 5 tokens (start token, role, \n, end token, \n), but we count the role token later
        num_tokens += 4
        for key, value in message.items():
            if isinstance(value, list) and key == "content":
                for entry in value:
                    if entry["type"] == "text":
                        num_tokens += len(encoding.encode(entry["text"]))
                    if entry["type"] == "image_url":
                        image_base64: str = entry["image_url"]["url"].split(",")[1]
                        image_bytes: bytes = base64.b64decode(image_base64)
                        image = Image.open(io.BytesIO(image_bytes))
                        size = image.size
                        # As described here: https://platform.openai.com/docs/guides/vision/calculating-costs
                        scale = min(1, 2048 / max(size))
                        size = (int(size[0] * scale), int(size[1] * scale))
                        scale = min(1, 768 / min(size))
                        size = (int(size[0] * scale), int(size[1] * scale))
                        num_tokens += 85 + 170 * ((size[0] + 511) // 512) * ((size[1] + 511) // 512)
            elif isinstance(value, str):
                num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens -= 1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <|start|>assistant
    return num_tokens
