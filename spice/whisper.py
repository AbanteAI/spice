from __future__ import annotations

import os
from pathlib import Path

from openai import AsyncAzureOpenAI, AsyncOpenAI

from spice.errors import SpiceError


class SpiceWhisper:
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
            self._client = AsyncOpenAI(api_key=key, base_url=base_url)
        elif provider == "azure":
            key = os.getenv("AZURE_OPENAI_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if key is None:
                raise SpiceError("AZURE_OPENAI_KEY not set")
            if endpoint is None:
                raise SpiceError("AZURE_OPENAI_ENDPOINT not set")
            self._client = AsyncAzureOpenAI(api_key=key, api_version="2023-12-01-preview", azure_endpoint=endpoint)
        else:
            raise SpiceError(f"Unknown whisper provider: {provider}")

    async def get_whisper_transcription(self, audio_path: Path) -> str:
        async with open(audio_path, "rb") as audio_file:
            transcript = await self._client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        return transcript.text
