import pytest

from spice import Spice
from spice.models import HAIKU
from tests.conftest import WrappedTestClient
from pydantic import BaseModel

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
    assert cache == "Hello, world!test"

class ResponseModel(BaseModel):
    message: str

@pytest.mark.asyncio
async def test_response_conversion():
    async def mock_api_call():
        return '{"message": "test"}'

    converter = lambda x: ResponseModel.parse_raw(x)
    response = await spice.get_response(messages=[], model=HAIKU, converter=converter)
    assert isinstance(response, ResponseModel)
    assert response.message == "test"
