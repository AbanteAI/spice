from spice import Spice

# model = "gpt-4-0125-preview"
model = "claude-3-opus-20240229"
stream = True

system_message = "You are a helpful assistant."
messages = [
    {"role": "user", "content": "list 5 random words"},
]

client = Spice(model=model)

response = client.call_llm(system_message, messages, stream=stream)

if stream:
    for t in response.stream():
        print(t, end="")

    print("\n####################\n")

print(response.text)
