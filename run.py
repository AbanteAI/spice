from spice import call_llm

system_message = "You are a helpful assistant."
messages = [
    {"role": "user", "content": "list 5 random words"},
]

for t in call_llm(system_message, messages, stream=True):
    print(t, end="")

print("\n####################\n")

response = call_llm(system_message, messages, stream=False)
print(response)
