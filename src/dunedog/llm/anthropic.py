"""Anthropic LLM provider."""
from __future__ import annotations

import httpx

from .provider import LLMError, LLMProvider


class AnthropicProvider(LLMProvider):
    """Provider for the Anthropic messages API."""

    API_URL = "https://api.anthropic.com/v1/messages"

    async def complete(self, messages: list[dict], **kwargs) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        system_content = ""
        non_system: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                non_system.append(msg)

        body: dict = {
            "model": self.model,
            "messages": non_system,
            "max_tokens": kwargs.get(
                "max_tokens", self.kwargs.get("max_tokens", 4096)
            ),
        }
        if system_content:
            body["system"] = system_content

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(self.API_URL, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
                return data["content"][0]["text"]
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            raise LLMError(
                f"Anthropic API returned {exc.response.status_code}: {body}"
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc
