import asyncio
import os
import sys
from timeit import default_timer as timer

from spice.models import o1_mini, o1_preview

# Modify sys.path to ensure the script can run even when it's not part of the installed library.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spice import Spice


async def run_o1():
    client = Spice()

    messages = client.new_messages().add_user_text("list 5 random words")

    models = [o1_mini, o1_preview]

    for model in models:
        print(f"\nRunning {model.name}:")
        start = timer()
        response = await client.get_response(messages=messages, model=model)
        end = timer()
        print(response.text)
        print(f"input tokens: {response.input_tokens}")
        print(f"output tokens: {response.output_tokens}")
        print(f"reasoning tokens: {response.reasoning_tokens}")
        print(f"total time: {end - start:.2f}s")


if __name__ == "__main__":
    asyncio.run(run_o1())
