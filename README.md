# Spice

## Usage Examples

All examples can be found in [scripts/run.py](scripts/run.py)

```python
from spice import Spice

client = Spice()

messages: List[SpiceMessage] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "list 5 random words"},
]
response = await client.completion(messages=messages, model="gpt-4-0125-preview")

print(response.text)
```

### Streaming

```python
# You can set a default model for the client instead of passing it with each call
client = Spice(model="claude-3-opus-20240229")

messages: List[SpiceMessage] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "list 5 random words"},
]
stream = await client.stream_completion(messages=messages)

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
# Alias models for easy configuration, even mixing providers
model_aliases = {
    "task1_model": {"model": "gpt-4-0125-preview"},
    "task2_model": {"model": "claude-3-opus-20240229"},
    "task3_model": {"model": "claude-3-haiku-20240307"},
}

client = Spice(model_aliases=model_aliases)

messages: List[SpiceMessage] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "list 5 random words"},
]
responses = await asyncio.gather(
    client.completion(messages=messages, model="task1_model"),
    client.completion(messages=messages, model="task2_model"),
    client.completion(messages=messages, model="task3_model"),
)

for i, response in enumerate(responses, 1):
    print(f"\nModel {i} response:")
    print(response.text)
    print(f"Characters per second: {response.characters_per_second:.2f}")
```
