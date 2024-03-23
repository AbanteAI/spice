import asyncio
import os
import sys

import fire

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spice import Spice


# TODO move to library utils?
def get_provider_and_model(model_hint):
    model_hint = str(model_hint).lower()

    models_list = [
        "gpt-4-0125-preview",
        "gpt-3.5-turbo-0125",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]

    # return first match
    for model in models_list:
        if model_hint in model:
            print("Using model:", model)
            if "gpt" in model:
                return "openai", model
            elif "claude" in model:
                return "anthropic", model
            else:
                raise ValueError(f"Unknown provider for model: {model}")


def display_stats(response):
    input_tokens = response.input_tokens
    output_tokens = response.output_tokens
    total_time = response.total_time

    print(f"\n\nlogged: {input_tokens} input tokens, {output_tokens} output tokens, {total_time:.2f}s\n\n")


async def run(model="", stream=False):
    _, model = get_provider_and_model(model)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "list 5 random words"},
    ]

    client = Spice(model)

    response = await client.call_llm(messages=messages, stream=stream, logging_callback=display_stats)

    print(">>>>>>>>>>>>>")
    if stream:
        async for t in response.stream():
            print(t, end="", flush=True)
    else:
        print(response.text)
    print("\n<<<<<<<<<<<<<")

    print(f"Took {response.total_time:.2f}s")

    if stream:
        print(f"Time to first token: {response.time_to_first_token:.2f}s")

    print(f"Input/Output tokens: {response.input_tokens}/{response.output_tokens}")
    print(f"Characters per second: {response.characters_per_second:.2f}")


def run_async(model="", stream=False):
    asyncio.run(run(model, stream))


if __name__ == "__main__":
    fire.Fire(run_async)
