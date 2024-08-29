import pytest

from spice import Spice
from spice.models import HAIKU
from spice.retry_strategy.converter_strategy import ConverterStrategy
from spice.retry_strategy.default_strategy import DefaultRetryStrategy
from spice.retry_strategy.validator_strategy import ValidatorStrategy
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
async def test_validator_retry_strategy():
    client = WrappedTestClient(iter(["Invalid response", "Valid response"]))

    def must_equal_valid(response):
        if response == "Valid response":
            return True, "You pass!"
        else:
            return False, "You fail!"

    retry_strategy = ValidatorStrategy(validator=must_equal_valid, retries=1)
    spice = Spice()

    def return_wrapped_client(model, provider):
        return client

    spice._get_client = return_wrapped_client
    response = await spice.get_response(messages=[], model=HAIKU, retry_strategy=retry_strategy)

    content = list(client.calls[-1].messages)[-1].content
    assert content.type == "text"
    last_message = content.text
    assert isinstance(last_message, str)
    assert "Failed to validate response for the following reason: You fail!" in last_message
    assert response.text == "Valid response"


@pytest.mark.asyncio
async def test_converter_retry_strategy():
    client = WrappedTestClient(iter(["Not an int", "42"]))

    spice = Spice()

    def return_wrapped_client(model, provider):
        return client

    spice._get_client = return_wrapped_client
    retry_strategy = ConverterStrategy(converter=int, retries=1)
    response = await spice.get_response(messages=[], model=HAIKU, retry_strategy=retry_strategy)

    content = list(client.calls[-1].messages)[-1].content
    assert content.type == "text"
    last_message = content.text
    assert isinstance(last_message, str)
    assert (
        "Failed to convert response for the following reason: invalid literal for int() with base 10: 'Not an int'"
        in last_message
    )
    assert response.text == "42"
    assert response.result == 42
