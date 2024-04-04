# Spice

## Usage Examples

All examples can be found in [scripts/run.py](scripts/run.py)

```python
from spice import Spice

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "list 5 random words"},
]

client = Spice()
response = await client.call_llm(messages=messages, model="gpt-4-0125-preview")
print(response.text)
```

### Streaming

```python
# you can set a default model for the client instead of passing it with each call
client = Spice(model="claude-3-opus-20240229")

# the messages format is automatically converted for the Anthropic API
response = await client.call_llm(messages=messages, stream=True)

# spice wraps the stream to extract just the text you care about
async for text in response.stream():
    print(text, end="", flush=True)

# response always includes the final text, no need build it from the stream yourself
print(response.text)

# response also includes helpful stats
print(f"Took {response.total_time:.2f}s")
print(f"Input/Output tokens: {response.input_tokens}/{response.output_tokens}")
```

### Mixing Providers

```python
# alias model names for easy configuration, even mixing providers
model_aliases = {
    "task1_model": {"model": "gpt-4-0125-preview"},
    "task2_model": {"model": "claude-3-opus-20240229"},
    "task3_model": {"model": "claude-3-haiku-20240307"},
}

client = Spice(model_aliases=model_aliases)

responses = await asyncio.gather(
    client.call_llm(messages=messages, model="task1_model"),
    client.call_llm(messages=messages, model="task2_model"),
    client.call_llm(messages=messages, model="task3_model"),
)

for i, response in enumerate(responses, 1):
    print(f"\nModel {i} response:")
    print(response.text)
    print(f"Characters per second: {response.characters_per_second:.2f}")
```

### Logging Callbacks

```python
client = Spice(model="gpt-3.5-turbo-0125")

# pass a logging function to get a callback after the stream completes
response = await client.call_llm(
    messages=messages, stream=True, logging_callback=lambda response: print(response.text)
)

async for text in response.stream():
    print(text, end="", flush=True)
```
