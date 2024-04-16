# Spice

Spice is a light wrapper for AI SDKs like OpenAI's and Anthropic's. Spice simplifies LLM creations, embeddings, and transcriptions without obscuring any underlying parameters or processes. Spice also makes it ridiculously easy to switch between different providers, such as OpenAI and Anthropic, without having to modify your code.

Spice also collects useful information such as tokens used, time spent, and cost for each call, making it easily available no matter which LLM provider is being used.

## Install

Spice is listed under `spiceai` on PyPi. To install, simply `pip install spiceai`.

### API Keys

Spice will automatically load `.env` files in your current directory. To add an API key, either use a `.env` file or set the environment variables manually. These are the current environment variables that Spice will use:

```bash
OPENAI_API_KEY=<api_key> # Required for OpenAI calls
OPENAI_API_BASE=<base_url> # If set, will set the base url for OpenAI calls.

AZURE_OPENAI_KEY=<api_key> # Required for Azure OpenAI calls
AZURE_OPENAI_ENDPOINT=<endpoint_url> # Required for Azure OpenAI calls.

ANTHROPIC_API_KEY=<api_key> # Required for Anthropic calls
```

## Usage Examples

All examples can be found in [scripts/run.py](scripts/run.py)

```python
from spice import Spice

client = Spice()

messages: List[SpiceMessage] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "list 5 random words"},
]
response = await client.get_response(messages=messages, model="gpt-4-0125-preview")

print(response.text)
```

### Streaming

```python
# You can set a default model for the client instead of passing it with each call
client = Spice(default_text_model="claude-3-opus-20240229")

# You can easily load prompts from files, directories, or even urls.
client.load_prompt("prompt.txt", name="my prompt")

# Spice can also automatically render Jinja templates.
messages: List[SpiceMessage] = [
    {"role": "system", "content": client.get_rendered_prompt("my prompt", assistant_name="Ryan Reynolds")},
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
```

### Mixing Providers

```python
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
    if response.cost is not None:
        print(f"Cost: ${response.cost / 100:.4f}")

# Spice also tracks the total cost over multiple models and providers
print(f"Total Cost: ${client.total_cost / 100:.4f}")
```

### Using unknown models

```python
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
```

### Vision models

```python
client = Spice()

# Spice makes it easy to add images from files or the internet
from spice.spice_message import file_image_message, user_message

messages: List[SpiceMessage] = [user_message("What do you see?"), file_image_message("/path/to/image.png")]
response = await client.get_response(messages, GPT_4_1106_VISION_PREVIEW)
print(response.text)

# Alternatively, you can use the SpiceMessages wrapper to easily create your prompts
spice_messages: SpiceMessages = SpiceMessages(client)
spice_messages.add_user_message("What do you see?")
spice_messages.add_file_image_message("https://example.com/image.png")
response = await client.get_response(spice_messages, CLAUDE_3_OPUS_20240229)
print(response.text)
```

### Embeddings and Transcriptions

```python
client = Spice()
input_texts = ["Once upon a time...", "Cinderella"]

# Spice can easily fetch embeddings and audio transcriptions
from spice.models import TEXT_EMBEDDING_ADA_002, WHISPER_1

embeddings = await client.get_embeddings(input_texts, TEXT_EMBEDDING_ADA_002)
transcription = await client.get_transcription("/path/to/audio/file", WHISPER_1)
print(transcription.text)
```
