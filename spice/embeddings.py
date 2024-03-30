from __future__ import annotations

import os
from typing import TYPE_CHECKING

from openai import AzureOpenAI, OpenAI

from spice.errors import SpiceError

if TYPE_CHECKING:
    # This import is slow
    from chromadb.api.types import Embeddings


class SpiceEmbeddings:
    def __init__(self, provider: str = "openai"):
        if provider == "openai":
            base_url = os.getenv("OPENAI_API_BASE")
            key = os.getenv("OPENAI_API_KEY")
            if key is None:
                if base_url:
                    # Using a dummy key for LiteLLM proxy setup where OPENAI_API_BASE is set but OPENAI_API_KEY is not required.
                    key = "dummy_key_base_url"
                else:
                    raise SpiceError("OPENAI_API_KEY not set")
            self.client = OpenAI(api_key=key, base_url=base_url)
        elif provider == "azure":
            key = os.getenv("AZURE_OPENAI_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if key is None:
                raise SpiceError("AZURE_OPENAI_KEY not set")
            if endpoint is None:
                raise SpiceError("AZURE_OPENAI_ENDPOINT not set")
            self.client = AzureOpenAI(api_key=key, api_version="2023-12-01-preview", azure_endpoint=endpoint)
        else:
            raise SpiceError(f"Unknown embeddings provider: {provider}")

    def get_embeddings(self, input_texts: list[str], model: str = "text-embedding-ada-002") -> Embeddings:
        assert False, "yo"
        embeddings = self.client.embeddings.create(input=input_texts, model=model).data
        sorted_embeddings = sorted(embeddings, key=lambda e: e.index)
        return [result.embedding for result in sorted_embeddings]
