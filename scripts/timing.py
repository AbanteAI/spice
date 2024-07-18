import asyncio
import os
import sys
from timeit import default_timer as timer
from typing import List

# Modify sys.path to ensure the script can run even when it's not part of the installed library.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spice import Spice, SpiceMessage


async def speed_compare():
    client = Spice()

    messages: List[SpiceMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "list 100 random words"},
    ]

    models = ["gpt-4o-mini", "gpt-4o"]
    runs = 5

    for model in models:
        print(f"Timing {model}:")
        for i in range(runs):
            start = timer()
            response = await client.get_response(messages=messages, model=model)
            end = timer()
            print(f"    Run {i + 1}: {end - start:.1f}s, {response.characters_per_second:.0f} characters/s")


if __name__ == "__main__":
    asyncio.run(speed_compare())
