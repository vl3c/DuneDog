"""Tests for LLM HTTP provider complete() methods using mocked httpx."""

import asyncio

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dunedog.llm.openai import OpenAIProvider
from dunedog.llm.anthropic import AnthropicProvider
from dunedog.llm.openrouter import OpenRouterProvider
from dunedog.llm.chatgpt import ChatGPTProvider
from dunedog.llm.provider import LLMError, LLMProvider, create_provider


def _mock_response(json_data, status_code=200):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = ""
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


def _make_mock_client(mock_resp):
    """Create a mock httpx.AsyncClient with context manager support."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


# ------------------------------------------------------------------ #
# OpenAI
# ------------------------------------------------------------------ #


class TestOpenAIProvider:
    def test_complete_success(self):
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        mock_resp = _mock_response({
            "choices": [{"message": {"content": "Hello from OpenAI"}}]
        })

        with patch("dunedog.llm.openai.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_mock_client(mock_resp)
            result = asyncio.run(provider.complete([{"role": "user", "content": "Hi"}]))
            assert result == "Hello from OpenAI"

    def test_complete_error_401(self):
        provider = OpenAIProvider(api_key="bad-key", model="gpt-4o")
        mock_resp = _mock_response({}, status_code=401)

        with patch("dunedog.llm.openai.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_mock_client(mock_resp)
            with pytest.raises(LLMError, match="OpenAI API returned 401"):
                asyncio.run(provider.complete([{"role": "user", "content": "Hi"}]))

    def test_complete_sends_correct_body(self):
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        mock_resp = _mock_response({
            "choices": [{"message": {"content": "ok"}}]
        })
        mock_client = _make_mock_client(mock_resp)

        with patch("dunedog.llm.openai.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            asyncio.run(provider.complete(
                [{"role": "user", "content": "Hello"}],
                max_tokens=100,
                temperature=0.5,
            ))
            call_args = mock_client.post.call_args
            body = call_args.kwargs.get("json") or call_args[1].get("json")
            assert body["model"] == "gpt-4o"
            assert body["max_tokens"] == 100
            assert body["temperature"] == 0.5
            assert body["messages"] == [{"role": "user", "content": "Hello"}]

    def test_complete_uses_default_max_tokens(self):
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        mock_resp = _mock_response({
            "choices": [{"message": {"content": "ok"}}]
        })
        mock_client = _make_mock_client(mock_resp)

        with patch("dunedog.llm.openai.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            asyncio.run(provider.complete([{"role": "user", "content": "Hi"}]))
            call_args = mock_client.post.call_args
            body = call_args.kwargs.get("json") or call_args[1].get("json")
            assert body["max_tokens"] == 4096

    def test_complete_key_error_raises_llm_error(self):
        """If the response JSON is missing expected keys, LLMError is raised."""
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        mock_resp = _mock_response({"unexpected": "format"})

        with patch("dunedog.llm.openai.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_mock_client(mock_resp)
            with pytest.raises(LLMError, match="OpenAI request failed"):
                asyncio.run(provider.complete([{"role": "user", "content": "Hi"}]))

    def test_complete_sends_auth_header(self):
        provider = OpenAIProvider(api_key="sk-test-123", model="gpt-4o")
        mock_resp = _mock_response({
            "choices": [{"message": {"content": "ok"}}]
        })
        mock_client = _make_mock_client(mock_resp)

        with patch("dunedog.llm.openai.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            asyncio.run(provider.complete([{"role": "user", "content": "Hi"}]))
            call_args = mock_client.post.call_args
            headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
            assert headers["Authorization"] == "Bearer sk-test-123"


# ------------------------------------------------------------------ #
# Anthropic
# ------------------------------------------------------------------ #


class TestAnthropicProvider:
    def test_complete_success(self):
        provider = AnthropicProvider(api_key="test-key", model="claude-sonnet-4-5-20250929")
        mock_resp = _mock_response({
            "content": [{"text": "Hello from Anthropic"}]
        })

        with patch("dunedog.llm.anthropic.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_mock_client(mock_resp)
            result = asyncio.run(provider.complete([
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ]))
            assert result == "Hello from Anthropic"

    def test_system_message_extraction(self):
        """Anthropic provider should extract system messages separately."""
        provider = AnthropicProvider(api_key="test-key", model="claude-sonnet-4-5-20250929")
        mock_resp = _mock_response({"content": [{"text": "response"}]})
        mock_client = _make_mock_client(mock_resp)

        with patch("dunedog.llm.anthropic.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            asyncio.run(provider.complete([
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Hello"},
            ]))

            call_args = mock_client.post.call_args
            body = call_args.kwargs.get("json") or call_args[1].get("json")
            assert body["system"] == "System prompt"
            # Messages should not contain system role
            assert all(m["role"] != "system" for m in body["messages"])

    def test_no_system_message_omits_system_field(self):
        """When there is no system message, the body should not have a system field."""
        provider = AnthropicProvider(api_key="test-key", model="claude-sonnet-4-5-20250929")
        mock_resp = _mock_response({"content": [{"text": "response"}]})
        mock_client = _make_mock_client(mock_resp)

        with patch("dunedog.llm.anthropic.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            asyncio.run(provider.complete([
                {"role": "user", "content": "Hello"},
            ]))

            call_args = mock_client.post.call_args
            body = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "system" not in body

    def test_complete_error_500(self):
        provider = AnthropicProvider(api_key="test-key", model="claude-sonnet-4-5-20250929")
        mock_resp = _mock_response({}, status_code=500)

        with patch("dunedog.llm.anthropic.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_mock_client(mock_resp)
            with pytest.raises(LLMError, match="Anthropic API returned 500"):
                asyncio.run(provider.complete([{"role": "user", "content": "Hi"}]))

    def test_sends_api_key_header(self):
        provider = AnthropicProvider(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929")
        mock_resp = _mock_response({"content": [{"text": "ok"}]})
        mock_client = _make_mock_client(mock_resp)

        with patch("dunedog.llm.anthropic.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            asyncio.run(provider.complete([{"role": "user", "content": "Hi"}]))
            call_args = mock_client.post.call_args
            headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
            assert headers["x-api-key"] == "sk-ant-test"
            assert "anthropic-version" in headers


# ------------------------------------------------------------------ #
# OpenRouter
# ------------------------------------------------------------------ #


class TestOpenRouterProvider:
    def test_complete_success(self):
        provider = OpenRouterProvider(api_key="test-key", model="anthropic/claude-sonnet-4.5")
        mock_resp = _mock_response({
            "choices": [{"message": {"content": "Hello from OpenRouter"}}]
        })

        with patch("dunedog.llm.openrouter.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_mock_client(mock_resp)
            result = asyncio.run(provider.complete([{"role": "user", "content": "Hi"}]))
            assert result == "Hello from OpenRouter"

    def test_complete_error_429(self):
        provider = OpenRouterProvider(api_key="test-key", model="anthropic/claude-sonnet-4.5")
        mock_resp = _mock_response({}, status_code=429)

        with patch("dunedog.llm.openrouter.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_mock_client(mock_resp)
            with pytest.raises(LLMError, match="OpenRouter API returned 429"):
                asyncio.run(provider.complete([{"role": "user", "content": "Hi"}]))

    def test_sends_referer_header(self):
        provider = OpenRouterProvider(api_key="test-key", model="anthropic/claude-sonnet-4.5")
        mock_resp = _mock_response({
            "choices": [{"message": {"content": "ok"}}]
        })
        mock_client = _make_mock_client(mock_resp)

        with patch("dunedog.llm.openrouter.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            asyncio.run(provider.complete([{"role": "user", "content": "Hi"}]))
            call_args = mock_client.post.call_args
            headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
            assert "HTTP-Referer" in headers

    def test_complete_malformed_response(self):
        provider = OpenRouterProvider(api_key="test-key", model="anthropic/claude-sonnet-4.5")
        mock_resp = _mock_response({"bad": "data"})

        with patch("dunedog.llm.openrouter.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_mock_client(mock_resp)
            with pytest.raises(LLMError, match="OpenRouter request failed"):
                asyncio.run(provider.complete([{"role": "user", "content": "Hi"}]))


# ------------------------------------------------------------------ #
# ChatGPT
# ------------------------------------------------------------------ #


class TestChatGPTProvider:
    def test_complete_success(self):
        provider = ChatGPTProvider(api_key="session-token", model="gpt-4o")
        mock_resp = _mock_response({
            "message": {"content": {"parts": ["Hello from ChatGPT"]}}
        })

        with patch("dunedog.llm.chatgpt.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_mock_client(mock_resp)
            result = asyncio.run(provider.complete([
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Hi"},
            ]))
            assert result == "Hello from ChatGPT"

    def test_complete_extracts_last_user_message(self):
        """ChatGPT provider should use the last user message."""
        provider = ChatGPTProvider(api_key="session-token", model="gpt-4o")
        mock_resp = _mock_response({
            "message": {"content": {"parts": ["ok"]}}
        })
        mock_client = _make_mock_client(mock_resp)

        with patch("dunedog.llm.chatgpt.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            asyncio.run(provider.complete([
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "first message"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second message"},
            ]))
            call_args = mock_client.post.call_args
            body = call_args.kwargs.get("json") or call_args[1].get("json")
            # The body should contain the last user message
            parts = body["messages"][0]["content"]["parts"]
            assert parts == ["second message"]

    def test_complete_error_403(self):
        provider = ChatGPTProvider(api_key="bad-token", model="gpt-4o")
        mock_resp = _mock_response({}, status_code=403)

        with patch("dunedog.llm.chatgpt.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = _make_mock_client(mock_resp)
            with pytest.raises(LLMError, match="ChatGPT API returned 403"):
                asyncio.run(provider.complete([{"role": "user", "content": "Hi"}]))

    def test_sends_session_cookie(self):
        provider = ChatGPTProvider(api_key="my-session-tok", model="gpt-4o")
        mock_resp = _mock_response({
            "message": {"content": {"parts": ["ok"]}}
        })
        mock_client = _make_mock_client(mock_resp)

        with patch("dunedog.llm.chatgpt.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = mock_client
            asyncio.run(provider.complete([{"role": "user", "content": "Hi"}]))
            call_args = mock_client.post.call_args
            headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
            assert "my-session-tok" in headers["Cookie"]


# ------------------------------------------------------------------ #
# Provider repr and factory
# ------------------------------------------------------------------ #


class TestProviderRepr:
    def test_repr_masks_api_key(self):
        provider = OpenAIProvider(api_key="sk-secret-key", model="gpt-4o")
        r = repr(provider)
        assert "sk-secret-key" not in r
        assert "***" in r

    def test_repr_empty_key(self):
        provider = OpenAIProvider(api_key="", model="gpt-4o")
        r = repr(provider)
        assert "***" not in r

    def test_repr_shows_model(self):
        provider = AnthropicProvider(api_key="key", model="claude-sonnet-4-5-20250929")
        r = repr(provider)
        assert "claude-sonnet-4-5-20250929" in r

    def test_repr_shows_class_name(self):
        provider = OpenRouterProvider(api_key="key", model="some-model")
        r = repr(provider)
        assert "OpenRouterProvider" in r


class TestCreateProvider:
    def test_create_openai(self):
        p = create_provider("openai", api_key="key")
        assert isinstance(p, OpenAIProvider)

    def test_create_anthropic(self):
        p = create_provider("anthropic", api_key="key")
        assert isinstance(p, AnthropicProvider)

    def test_create_openrouter(self):
        p = create_provider("openrouter", api_key="key")
        assert isinstance(p, OpenRouterProvider)

    def test_create_chatgpt(self):
        p = create_provider("chatgpt", api_key="key")
        assert isinstance(p, ChatGPTProvider)

    def test_create_unknown_raises(self):
        with pytest.raises(LLMError, match="Unknown provider"):
            create_provider("nonexistent", api_key="key")

    def test_default_model_applied(self):
        p = create_provider("openai", api_key="key")
        assert p.model == "gpt-4o"

    def test_custom_model_overrides_default(self):
        p = create_provider("openai", api_key="key", model="gpt-3.5-turbo")
        assert p.model == "gpt-3.5-turbo"
