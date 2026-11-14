"""Anthropic LLM provider."""
from __future__ import annotations

import logging

import httpx

from .provider import LLMError, LLMProvider

log = logging.getLogger(__name__)

# Thinking models can take minutes on one request at high effort.  The
# request is not streamed, so nothing arrives until the whole answer is done.
ANTHROPIC_TIMEOUT = httpx.Timeout(connect=10.0, read=1800.0, write=30.0, pool=10.0)

# Thinking plus three stories at high effort.  Without streaming, caps are
# kept well short of the 128K model limit (the CLI allows up to 64000).
DEFAULT_MAX_TOKENS = 32000


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
            # max_tokens covers thinking as well as the stories: at high
            # effort 16000 left too little room and cut the last story off.
            "max_tokens": kwargs.get("max_tokens")
            or self.kwargs.get("max_tokens")
            or DEFAULT_MAX_TOKENS,
        }
        if system_content:
            body["system"] = system_content
        # Current models take no sampling parameters (temperature is
        # rejected); depth is set with effort instead.
        effort = kwargs.get("effort") or self.kwargs.get("effort")
        if effort:
            body["output_config"] = {"effort": effort}

        try:
            async with httpx.AsyncClient(timeout=ANTHROPIC_TIMEOUT) as client:
                resp = await client.post(self.API_URL, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise LLMError(
                f"Anthropic API returned {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc

        try:
            stop_reason = data.get("stop_reason")
            if stop_reason == "refusal":
                category = (data.get("stop_details") or {}).get("category")
                raise LLMError(f"Anthropic model declined the request (category: {category})")
            # The response is a list of blocks; thinking blocks come first on
            # models that think, so take the text blocks rather than block 0.
            text = "".join(
                block.get("text", "") for block in data["content"]
                if block.get("type", "text") == "text"
            )
            if not text:
                raise LLMError(f"Anthropic response had no text (stop_reason: {stop_reason})")
            self.last_truncated = stop_reason == "max_tokens"
            if self.last_truncated:
                log.warning("Anthropic response was cut off at max_tokens; the last story may be incomplete")
            return text
        except (KeyError, TypeError, AttributeError) as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc
