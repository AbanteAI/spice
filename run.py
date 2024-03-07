from spice import call_llm

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "list 10 random words"},
]


for t in call_llm(messages, stream=True):
    print(t, end="")

print("\n####################\n")

response = call_llm(messages, stream=False)
print(response)
