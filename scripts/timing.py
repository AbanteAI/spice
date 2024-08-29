import asyncio
import os
import sys
from timeit import default_timer as timer

from spice.models import SONNET_3_5, GPT_4o_2024_08_06

# Modify sys.path to ensure the script can run even when it's not part of the installed library.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spice import Spice


async def speed_compare():
    client = Spice()

    messages = (
        client.new_messages().add_system_text("You are a helpful assistant.").add_user_text("list 100 random words")
    )

    models = [GPT_4o_2024_08_06, SONNET_3_5]
    runs = 3

    for model in models:
        print(f"Timing {model.name}:")
        for i in range(runs):
            start = timer()
            response = await client.get_response(messages=messages, model=model)
            end = timer()
            print(f"    Run {i + 1}: {end - start:.1f}s, {response.characters_per_second:.0f} characters/s")


if __name__ == "__main__":
    asyncio.run(speed_compare())
