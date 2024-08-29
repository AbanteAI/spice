import asyncio
import os
import random
import sys

import requests
from termcolor import cprint

from spice import Spice
from spice.models import SONNET_3_5
from spice.utils import print_stream

# Modify sys.path to ensure the script can run even when it's not part of the installed library.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def display_response_stats(response):
    cprint(f"Input tokens: {response.input_tokens}", "cyan")
    cprint(f"Cache Creation Input Tokens: {response.cache_creation_input_tokens}", "cyan")
    cprint(f"Cache Read Input Tokens: {response.cache_read_input_tokens}", "cyan")
    cprint(f"Output tokens: {response.output_tokens}", "cyan")
    cprint(f"Total time: {response.total_time:.2f}s", "green")
    cprint(f"Cost: ${response.cost / 100:.2f}", "green")


async def run(cache: bool):
    cprint(f"Caching on: {cache}", "cyan")
    client = Spice()
    model = SONNET_3_5
    book_text = requests.get("https://www.gutenberg.org/cache/epub/42671/pg42671.txt").text

    messages = (
        client.new_messages()
        .add_system_text(f"Answer questions about a book. Seed:{random.random()}")
        .add_user_text(f"<book>{book_text}</book>", cache=cache)
    )

    cprint("First model response:", "green")
    response = await client.get_response(
        messages=messages.copy().add_user_text("how many chapters are there? no elaboration."),
        model=model,
        streaming_callback=print_stream,
    )
    display_response_stats(response)

    cprint("Second model response:", "green")
    response = await client.get_response(
        messages=messages.copy().add_user_text("how many volumes are there? no elaboration."),
        model=model,
        streaming_callback=print_stream,
    )
    display_response_stats(response)


if __name__ == "__main__":
    cache = any(arg in sys.argv for arg in ["--cache", "-c"])
    asyncio.run(run(cache))
