import asyncio
import os
import sys

# Modify sys.path to ensure the script can run even when it's not part of the installed library.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spice import Spice


async def basic_example():
    client = Spice()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "list 5 random words"},
    ]

    response = await client.call_llm(messages=messages, model="gpt-4-0125-preview")

    print(response.text)


async def streaming_example():
    # you can set a default model for the client instead of passing it with each call
    client = Spice(model="claude-3-opus-20240229")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "list 5 random words"},
    ]

    response = await client.call_llm(messages=messages, stream=True)

    async for text in response.stream():
        print(text, end="", flush=True)

    # response always includes the final text, no need build it from the stream yourself
    print(response.text)

    # response also includes helpful stats
    print(f"Took {response.total_time:.2f}s")
    print(f"Time to first token: {response.time_to_first_token:.2f}s")
    print(f"Input/Output tokens: {response.input_tokens}/{response.output_tokens}")


async def multiple_providers_example():
    # alias models for easy configuration, even mixing providers
    model_aliases = {
        "task1_model": {"model": "gpt-4-0125-preview"},
        "task2_model": {"model": "claude-3-opus-20240229"},
        "task3_model": {"model": "claude-3-haiku-20240307"},
    }

    client = Spice(model_aliases=model_aliases)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "list 5 random words"},
    ]

    responses = await asyncio.gather(
        client.call_llm(messages=messages, model="task1_model"),
        client.call_llm(messages=messages, model="task2_model"),
        client.call_llm(messages=messages, model="task3_model"),
    )

    for i, response in enumerate(responses, 1):
        print(f"\nModel {i} response:")
        print(response.text)
        print(f"Characters per second: {response.characters_per_second:.2f}")


async def run_all_examples():
    print("Running basic example:")
    await basic_example()
    print("\n\nRunning streaming example:")
    await streaming_example()
    print("\n\nRunning multiple providers example:")
    await multiple_providers_example()


if __name__ == "__main__":
    asyncio.run(run_all_examples())
