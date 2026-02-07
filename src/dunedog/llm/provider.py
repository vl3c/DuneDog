"""Abstract LLM provider and factory."""
from __future__ import annotations
from abc import ABC, abstractmethod

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


class LLMError(Exception):
    """Error from LLM provider."""
    pass


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    def __init__(self, api_key: str = "", model: str = "", **kwargs):
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs

    def __repr__(self) -> str:
        key_display = "***" if self.api_key else ""
        return f"{self.__class__.__name__}(api_key='{key_display}', model='{self.model}')"

    @abstractmethod
    async def complete(self, messages: list[dict], **kwargs) -> str:
        """Send messages to LLM and return response text."""
        ...


DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5-20250929",
    "openrouter": "anthropic/claude-sonnet-4.5",
    "chatgpt": "gpt-4o",
}


def create_provider(
    provider_name: str, api_key: str = "", model: str = "", **kwargs
) -> LLMProvider:
    """Factory function to create the right provider."""
    if not model:
        model = DEFAULT_MODELS.get(provider_name, "")

    if provider_name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(api_key=api_key, model=model, **kwargs)
    elif provider_name == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(api_key=api_key, model=model, **kwargs)
    elif provider_name == "openrouter":
        from .openrouter import OpenRouterProvider

        return OpenRouterProvider(api_key=api_key, model=model, **kwargs)
    elif provider_name == "chatgpt":
        from .chatgpt import ChatGPTProvider

        return ChatGPTProvider(api_key=api_key, model=model, **kwargs)
    else:
        raise LLMError(f"Unknown provider: {provider_name}")
