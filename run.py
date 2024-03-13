from spice import SpiceClient

# model = "gpt-4-0125-preview"
model = "claude-3-opus-20240229"
stream = False

system_message = "You are a helpful assistant."
messages = [
    {"role": "user", "content": "list 5 random words"},
]

client = SpiceClient(model=model)

response = client.call_llm(system_message, messages, stream=stream)

if stream:
    for t in response.stream():
        print(t, end="")

    print("\n####################\n")

print(response.text)
