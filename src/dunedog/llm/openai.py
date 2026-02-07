"""OpenAI LLM provider."""
from __future__ import annotations

import httpx

from .provider import DEFAULT_TIMEOUT, LLMError, LLMProvider


class OpenAIProvider(LLMProvider):
    """Provider for the OpenAI chat completions API."""

    API_URL = "https://api.openai.com/v1/chat/completions"

    async def complete(self, messages: list[dict], **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get(
                "max_tokens", self.kwargs.get("max_tokens", 4096)
            ),
            "temperature": kwargs.get(
                "temperature", self.kwargs.get("temperature", 0.8)
            ),
        }
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                resp = await client.post(self.API_URL, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            raise LLMError(
                f"OpenAI API returned {exc.response.status_code}: {body}"
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
