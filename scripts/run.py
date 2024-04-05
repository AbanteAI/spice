import asyncio
import os
import sys
from typing import Dict, List

# Modify sys.path to ensure the script can run even when it's not part of the installed library.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spice import Spice
from spice.spice_message import SpiceMessage


async def basic_example():
    client = Spice()

    messages: List[SpiceMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "list 5 random words"},
    ]
    response = await client.get_response(messages=messages, model="gpt-4-0125-preview")

    print(response.text)


async def streaming_example():
    # You can set a default model for the client instead of passing it with each call
    client = Spice(default_text_model="claude-3-opus-20240229")

    messages: List[SpiceMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "list 5 random words"},
    ]
    stream = await client.stream_response(messages=messages)

    async for text in stream:
        print(text, end="", flush=True)
    # Retrieve the complete response from the stream
    response = await stream.complete_response()

    # Response always includes the final text, no need build it from the stream yourself
    print(response.text)

    # Response also includes helpful stats
    print(f"Took {response.total_time:.2f}s")
    print(f"Input/Output tokens: {response.input_tokens}/{response.output_tokens}")


async def multiple_providers_example():
    # Commonly used models and providers have premade constants
    from spice.models import GPT_4_0125_PREVIEW

    # Alias models for easy configuration, even mixing providers
    model_aliases = {
        "task1_model": GPT_4_0125_PREVIEW,
        "task2_model": "claude-3-opus-20240229",
        "task3_model": "claude-3-haiku-20240307",
    }

    client = Spice(model_aliases=model_aliases)

    messages: List[SpiceMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "list 5 random words"},
    ]
    responses = await asyncio.gather(
        client.get_response(messages=messages, model="task1_model"),
        client.get_response(messages=messages, model="task2_model"),
        client.get_response(messages=messages, model="task3_model"),
    )

    for i, response in enumerate(responses, 1):
        print(f"\nModel {i} response:")
        print(response.text)
        print(f"Characters per second: {response.characters_per_second:.2f}")


async def azure_example():
    client = Spice()

    messages: List[SpiceMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "list 5 random words"},
    ]

    # To use Azure, specify the provider and the deployment model name
    response = await client.get_response(messages=messages, model="first-gpt35", provider="azure")
    print(response.text)

    # Alternatively, to make a model and it's provider known to Spice, create a custom Model object
    from spice.models import TextModel
    from spice.providers import AZURE

    AZURE_GPT = TextModel("first-gpt35", AZURE, context_length=16385)
    response = await client.get_response(messages=messages, model=AZURE_GPT)
    print(response.text)

    # Creating the model automatically registers it in Spice's model list, so listing the provider is no longer needed
    response = await client.get_response(messages=messages, model="first-gpt35")
    print(response.text)


async def run_all_examples():
    print("Running Azure example:")
    await azure_example()
    print("Running basic example:")
    await basic_example()
    print("\n\nRunning streaming example:")
    await streaming_example()
    print("\n\nRunning multiple providers example:")
    await multiple_providers_example()


if __name__ == "__main__":
    asyncio.run(run_all_examples())
