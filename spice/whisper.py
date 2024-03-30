from __future__ import annotations
import os
from pathlib import Path
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from azure.ai.textanalytics.aio import TextAnalyticsClient

if TYPE_CHECKING:
    from openai.api_resources.async_resources import AsyncAudio

class SpiceWhisper:
    def __init__(self, provider: str = "openai"):
        if provider == "openai":
            base_url = os.getenv("OPENAI_API_BASE")
            key = os.getenv("OPENAI_API_KEY")
            if key is None:
                raise ValueError("OPENAI_API_KEY not set")
            self.async_client = AsyncOpenAI(api_key=key, base_url=base_url)
        elif provider == "azure":
            key = os.getenv("AZURE_OPENAI_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if key is None or endpoint is None:
                raise ValueError("AZURE_OPENAI_KEY or AZURE_OPENAI_ENDPOINT not set")
            self.async_client = TextAnalyticsClient(endpoint=endpoint, credential=key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def call_whisper_api(self, audio_path: Path) -> str:
        async with open(audio_path, "rb") as audio_file:
            transcript = await self.async_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        return transcript.text