"""ChatGPT unofficial backend provider (best-effort)."""
from __future__ import annotations

import httpx

from .provider import DEFAULT_TIMEOUT, LLMError, LLMProvider


class ChatGPTProvider(LLMProvider):
    """Best-effort provider using the unofficial ChatGPT backend API."""

    API_URL = "https://chatgpt.com/backend-api/conversation"

    async def complete(self, messages: list[dict], **kwargs) -> str:
        headers = {
            "Cookie": f"__Secure-next-auth.session-token={self.api_key}",
            "Content-Type": "application/json",
        }

        # Extract the last user message for the ChatGPT backend format.
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        body = {
            "action": "next",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "content_type": "text",
                        "parts": [user_message],
                    },
                }
            ],
            "model": self.model,
        }
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                resp = await client.post(self.API_URL, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"]["parts"][0]
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            raise LLMError(
                f"ChatGPT API returned {exc.response.status_code}: {body}"
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            raise LLMError(f"ChatGPT request failed: {exc}") from exc
