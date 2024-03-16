# TODO: use fire and command line args
from spice import Spice

# temporarily leave these lines to quickly switch between models
# model = "gpt-3.5-turbo-0125"
# model = "gpt-4-0125-preview"
model = "claude-3-opus-20240229"
# model = "claude-3-haiku-20240307"
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
print(f"Took {response.total_time:.2f}s")
print(f"Time to first token: {response.time_to_first_token:.2f}s")
