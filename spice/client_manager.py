import os

from dotenv import load_dotenv

from spice.errors import SpiceError
from spice.wrapped_clients import WrappedAnthropicClient, WrappedAzureClient, WrappedOpenAIClient

load_dotenv()


def _get_client(provider):
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if key is None:
            raise SpiceError("ANTHROPIC_API_KEY not set")
        return WrappedAnthropicClient(key)
    elif provider == "openai":
        base_url = os.getenv("OPENAI_API_BASE")
        key = os.getenv("OPENAI_API_KEY")
        if key is None:
            if base_url:
                key = "dummy_key_base_url"
            else:
                # Using a dummy key for LiteLLM proxy setup where OPENAI_API_BASE is set but OPENAI_API_KEY is not required.
                key = "dummy_key_base_url"
                raise SpiceError("OPENAI_API_KEY not set")
        return WrappedOpenAIClient(key, base_url)
    elif provider == "azure":
        key = os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if key is None:
            raise SpiceError("AZURE_OPENAI_KEY not set")
        if endpoint is None:
            raise SpiceError("AZURE_OPENAI_ENDPOINT not set")
        return WrappedAzureClient(key, endpoint)
    else:
        raise SpiceError(f"Unknown provider: {provider}")


def _get_clients_from_env():
    known_providers = {"openai", "anthropic", "azure"}
    clients = {}
    for provider in known_providers:
        try:
            clients[provider] = _get_client(provider)
        except SpiceError:
            pass
    if not clients:
        raise SpiceError("No known providers found in environment")
    return clients


_model_info = {
    "gpt-4-0125-preview": {"provider": "openai"},
    "gpt-3.5-turbo-0125": {"provider": "openai"},
    "claude-3-opus-20240229": {"provider": "anthropic"},
    "claude-3-haiku-20240307": {"provider": "anthropic"},
}


def _get_provider_from_model_name(model):
    if model not in _model_info:
        raise SpiceError(f"Unknown provider for model: {model}")
    return _model_info[model]["provider"]


def _validate_model_aliases(model_aliases, clients):
    for alias, model_dict in model_aliases.items():
        model_name = model_dict["model"]
        if "provider" in model_dict:
            provider = model_dict["provider"]
        else:
            provider = _get_provider_from_model_name(model_name)
        if provider not in clients:
            raise SpiceError(f"Provider {provider} is not set up for model {model_name} ({alias})")
