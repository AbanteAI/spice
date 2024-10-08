import asyncio
import os
import sys

# Modify sys.path to ensure the script can run even when it's not part of the installed library.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spice import Spice


async def basic_example():
    client = Spice()
    model = "gpt-4o"

    messages = client.new_messages()
    messages.add_system_text("You are a helpful assistant.")
    messages.add_user_text("list 5 random words")

    tokens = client.count_prompt_tokens(messages, model=model)
    print(f"Prompt tokens: {tokens}")

    response = await client.get_response(messages=messages, model=model)
    print(response.text)

    json_response = response.model_dump_json(indent=2)
    print(json_response)


async def streaming_example():
    # You can set a default model for the client instead of passing it with each call
    client = Spice(default_text_model="claude-3-opus-20240229")

    # You can easily load prompts from files, directories, or even urls.
    client.load_prompt("scripts/prompt.txt", name="my prompt")

    # Spice can also automatically render Jinja templates.
    messages = client.new_messages()
    messages.add_system_prompt("my prompt", assistant_name="Friendly Robot")
    messages.add_user_text("list 5 random words")

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

    messages = client.new_messages()
    messages.add_system_text("You are a helpful assistant.")
    messages.add_user_text("list 5 random words")

    responses = await asyncio.gather(
        client.get_response(messages=messages, model="task1_model"),
        client.get_response(messages=messages, model="task2_model"),
        client.get_response(messages=messages, model="task3_model"),
    )

    for i, response in enumerate(responses, 1):
        print(f"\nModel {i} response:")
        print(response.text)
        print(f"Characters per second: {response.characters_per_second:.2f}")
        if response.cost is not None:
            print(f"Cost: ${response.cost / 100:.4f}")

    # Spice also tracks the total cost over multiple models and providers
    print(f"Total Cost: ${client.total_cost / 100:.4f}")


async def azure_example():
    client = Spice()

    messages = client.new_messages()
    messages.add_system_text("You are a helpful assistant.")
    messages.add_user_text("list 5 random words")

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


async def vision_example():
    client = Spice()

    # Spice makes it easy to add images from files or the internet
    from spice.models import CLAUDE_3_OPUS_20240229, GPT_4o

    messages = client.new_messages()
    messages.add_user_image_from_file("~/.mentat/picture.png")
    messages.add_user_text("What do you see?")
    response = await client.get_response(messages, GPT_4o)
    print(response.text)

    messages = (
        client.new_messages()
        .add_user_image_from_file("~/.mentat/picture.png")
        .add_user_text("What do you see? Describe the objects, colors, and style.")
    )
    response = await client.get_response(messages, CLAUDE_3_OPUS_20240229)
    print(response.text)


async def embeddings_and_transcription_example():
    client = Spice()
    input_texts = ["Once upon a time...", "Cinderella"]

    # Spice can easily fetch embeddings and audio transcriptions
    from spice.models import TEXT_EMBEDDING_ADA_002, WHISPER_1

    embeddings = await client.get_embeddings(input_texts, TEXT_EMBEDDING_ADA_002)
    transcription = await client.get_transcription("~/.mentat/logs/audio/talk_transcription.wav", WHISPER_1)

    print(transcription.text)
    print(f"{len(embeddings.embeddings)} embeddings fetched for ${(embeddings.cost or 0) / 100:.2f}")


async def run_all_examples():
    print("Running basic example:")
    await basic_example()
    print("\n\nRunning streaming example:")
    await streaming_example()
    print("\n\nRunning multiple providers example:")
    await multiple_providers_example()
    print("Running Azure example:")
    await azure_example()
    print("\n\nRunning vision example:")
    await vision_example()
    print("\n\nRunning embeddings and transcription example:")
    await embeddings_and_transcription_example()


if __name__ == "__main__":
    asyncio.run(run_all_examples())
