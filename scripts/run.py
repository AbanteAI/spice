import asyncio
import os
import sys

import fire

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spice import Spice


def get_model(model_hint):
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
            return model


async def run(model="", stream=False):
    model = get_model(model)
    system_message = "You are a helpful assistant."
    messages = [
        {"role": "user", "content": "list 5 random words"},
    ]

    client = Spice(model=model)

    response = await client.call_llm(system_message, messages, stream=stream)

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
