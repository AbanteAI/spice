import pytest

from spice import Spice
from spice.models import HAIKU
from spice.custom_retry_strategy import AddModelResponseRetryStrategy
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
    retry_strategy = DefaultRetryStrategy(validator=validator, retries=2)
    response = await spice.get_response(messages=[], model=HAIKU, retry_strategy=retry_strategy)

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
    retry_strategy = DefaultRetryStrategy(converter=int, retries=2)
    response = await spice.get_response(messages=[], model=HAIKU, retry_strategy=retry_strategy)

    assert response.text == "42"
    assert response.result == 42

@pytest.mark.asyncio
async def test_add_model_response_retry_strategy():
    client = WrappedTestClient(iter(["Invalid response", "Valid response"]))
    
    def always_invalid(response):
        return False

    retry_strategy = AddModelResponseRetryStrategy(validator=always_invalid, retries=1)
    spice = Spice()

    def return_wrapped_client(model, provider):
        return client

    spice._get_client = return_wrapped_client
    response = await spice.get_response(messages=[], model=HAIKU, retry_strategy=retry_strategy)

    assert "Previous response: Invalid response" in response.text
