from spice import SpiceClient

# model = "gpt-4-0125-preview"
model = "claude-3-opus-20240229"

system_message = "You are a helpful assistant."
messages = [
    {"role": "user", "content": "list 5 random words"},
]

client = SpiceClient(model=model)

# for t in client.call_llm(system_message, messages, stream=True):
#     print(t, end="")

# print("\n####################\n")

spice_response = client.call_llm(system_message, messages, stream=False)
print(spice_response.text)
