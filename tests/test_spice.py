import pytest

from spice import Spice
from spice.models import HAIKU
from tests.conftest import WrappedTestClient


@pytest.mark.asyncio
async def test_get_response_validator():
    client = WrappedTestClient(iter(["Hello, world!", "test"]))

    def equals_test(response):
        return response == "test"

    validator = equals_test

    spice = Spice()

    def return_wrapped_client(model, provider):
        return client

    spice._get_client = return_wrapped_client

    response = await spice.get_response(messages=[], model=HAIKU, validator=validator, retries=2)

    assert response.text == "test"


@pytest.mark.asyncio
async def test_streaming_callback():
    client = WrappedTestClient(iter(["Hello, world!"]))

    spice = Spice()

    def return_wrapped_client(model, provider):
        return client

    spice._get_client = return_wrapped_client
    cache = ""

    def accumulator(text: str):
        nonlocal cache
        cache += text

    response = await spice.get_response(messages=[], model=HAIKU, streaming_callback=accumulator)

    assert response.text == "Hello, world!"
    assert cache == "Hello, world!"


@pytest.mark.asyncio
async def test_get_response_converter():
    client = WrappedTestClient(iter(["Not an int", "42"]))

    spice = Spice()

    def return_wrapped_client(model, provider):
        return client

    spice._get_client = return_wrapped_client

    response = await spice.get_response(messages=[], model=HAIKU, converter=int, retries=2)

    assert response.text == "42"
    assert response.result == 42
