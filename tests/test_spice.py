import pytest

from spice import Spice
from spice.models import HAIKU
from tests.conftest import WrappedTestClient


@pytest.mark.asyncio
async def test_get_response():
    client = WrappedTestClient(iter(["Hello, world!", "test"]))

    def equals_test(response):
        return response == "test"

    validator = equals_test

    spice = Spice()

    def return_wrapped_client(model, provider):
        return client

    spice._get_client = return_wrapped_client
    cache = ""

    def accumulator(text: str):
        nonlocal cache
        cache += text

    response = await spice.get_response(
        messages=[], model=HAIKU, validator=validator, streaming_callback=accumulator, retries=2
    )

    assert response.text == "test"
    assert response.retries == 1
    assert cache == "Hello, world!test"
