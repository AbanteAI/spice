import os
from dataclasses import dataclass
from typing import Callable, List

from dotenv import load_dotenv

from spice.errors import InvalidProviderError, NoAPIKeyError
from spice.wrapped_clients import WrappedAnthropicClient, WrappedAzureClient, WrappedClient, WrappedOpenAIClient

# TODO: Is there a reason to have providers? We could just have a client that lazily checks for keys


@dataclass
class Provider:
    name: str
    get_client: Callable[[], WrappedClient]


def get_openai_client(cache=[]):
    if cache:
        return cache[0]

    base_url = os.getenv("OPENAI_API_BASE")
    key = os.getenv("OPENAI_API_KEY")
    if key is None:
        if base_url:
            # Using a dummy key for LiteLLM proxy setup where OPENAI_API_BASE is set but OPENAI_API_KEY is not required.
            key = "dummy_key"
        else:
            raise NoAPIKeyError("OPENAI_API_KEY not set")

    client = WrappedOpenAIClient(key, base_url)
    cache.append(client)
    return client


def get_azure_client(cache=[]):
    if cache:
        return cache[0]

    key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if key is None:
        raise NoAPIKeyError("AZURE_OPENAI_KEY not set")
    if endpoint is None:
        raise NoAPIKeyError("AZURE_OPENAI_ENDPOINT not set")

    client = WrappedAzureClient(key, endpoint)
    cache.append(client)
    return client


def get_anthropic_client(cache=[]):
    if cache:
        return cache[0]

    key = os.getenv("ANTHROPIC_API_KEY")
    if key is None:
        raise NoAPIKeyError("ANTHROPIC_API_KEY not set")

    client = WrappedAnthropicClient(key)
    cache.append(client)
    return client


load_dotenv()

OPEN_AI = Provider("open_ai", get_openai_client)
AZURE = Provider("azure", get_azure_client)
ANTHROPIC = Provider("anthropic", get_anthropic_client)

# Used to fetch a provider by name
providers: List[Provider] = [OPEN_AI, AZURE, ANTHROPIC]


def get_provider_from_name(provider_name: str) -> Provider:
    for provider in providers:
        if provider.name == provider_name:
            return provider

    raise InvalidProviderError(f"Unknown provider {provider_name}")