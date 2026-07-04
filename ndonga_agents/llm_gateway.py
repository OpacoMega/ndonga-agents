"""Provider-agnostic LLM gateway for Ndonga."""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("ndonga.llm_gateway")


class LLMGatewayError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class UnifiedMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class UnifiedToolCall(BaseModel):
    id: str
    name: str
    arguments: str = ""


class UnifiedLLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[UnifiedToolCall] = Field(default_factory=list)
    model: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ToolCallDelta:
    index: int = 0
    id: str | None = None
    function: Any | None = None


@dataclass
class DeltaFunction:
    name: str | None = None
    arguments: str | None = None


@dataclass
class ChoiceDelta:
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


@dataclass
class StreamChoice:
    delta: ChoiceDelta
    finish_reason: str | None = None


@dataclass
class StreamChunk:
    choices: list[StreamChoice] = field(default_factory=list)


class BaseLLMProvider(ABC):
    @abstractmethod
    async def achat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> UnifiedLLMResponse:
        raise NotImplementedError

    @abstractmethod
    async def achat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        raise NotImplementedError


class OpenRouterProvider(BaseLLMProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "HTTP-Referer": "https://ndonga.yapahub.com",
            "X-Title": "Ndonga",
        }

    async def achat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> UnifiedLLMResponse:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload)
        if response.status_code >= 400:
            raise LLMGatewayError(response.text, status_code=response.status_code)
        data = response.json()
        message = data.get("choices", [{}])[0].get("message", {})
        tool_calls = [
            UnifiedToolCall(
                id=call.get("id", ""),
                name=call.get("function", {}).get("name", ""),
                arguments=call.get("function", {}).get("arguments", ""),
            )
            for call in message.get("tool_calls") or []
        ]
        return UnifiedLLMResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            model=data.get("model") or model,
            raw=data,
        )

    async def achat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        logger.info(
            "LLM gateway stream start | provider=openrouter | model=%s | messages=%s | tools=%s | max_tokens=%s",
            model,
            len(messages),
            bool(tools),
            max_tokens,
        )
        # read_timeout acts as TTFT guard: if the model holds the connection
        # open with no bytes for this many seconds (free-tier queue wait or
        # mid-stream stall), httpx raises ReadTimeout → LLMGatewayError →
        # the caller's fallback chain moves to the next model.
        _timeout = httpx.Timeout(connect=5.0, read=8.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=_timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as response:
                logger.info(
                    "LLM gateway stream connected | provider=openrouter | model=%s | status=%s",
                    model,
                    response.status_code,
                )
                if response.status_code >= 400:
                    body = await response.aread()
                    logger.warning(
                        "LLM gateway stream rejected | provider=openrouter | model=%s | status=%s | body=%s",
                        model,
                        response.status_code,
                        body.decode("utf-8", errors="ignore")[:1000],
                    )
                    raise LLMGatewayError(body.decode("utf-8", errors="ignore"), status_code=response.status_code)
                chunk_count = 0
                try:
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            logger.debug("LLM gateway ignored non-data SSE line | model=%s | line=%s", model, line[:200])
                            continue
                        data_text = line.removeprefix("data:").strip()
                        if not data_text or data_text == "[DONE]":
                            if data_text == "[DONE]":
                                logger.info(
                                    "LLM gateway stream done | provider=openrouter | model=%s | chunks=%s",
                                    model,
                                    chunk_count,
                                )
                            continue
                        try:
                            chunk = _chunk_from_openai_delta(json.loads(data_text))
                        except json.JSONDecodeError as exc:
                            logger.warning(
                                "LLM gateway skipped malformed SSE JSON | provider=openrouter | model=%s | error=%s | data=%s",
                                model,
                                exc,
                                data_text[:500],
                            )
                            continue
                        chunk_count += 1
                        yield chunk
                except httpx.TimeoutException as exc:
                    # ReadTimeout fires when no bytes arrive for read_timeout seconds
                    # (free-tier queue wait or mid-stream stall). Convert to
                    # LLMGatewayError so callers' fallback chains handle it uniformly.
                    logger.warning(
                        "LLM gateway stream timed out | provider=openrouter | model=%s | chunks_before_timeout=%s | error=%s",
                        model,
                        chunk_count,
                        exc,
                    )
                    raise LLMGatewayError(f"Stream timed out after {chunk_count} chunks", status_code=None) from exc


class GroqProvider(OpenRouterProvider):
    """
    Groq LPU inference — same OpenAI-compatible API, 800+ tok/s on Llama 70B.
    Free tier: 14,400 req/day, 500k tokens/day per model.
    Model IDs (no org prefix): llama-3.3-70b-versatile, deepseek-r1-distill-llama-70b, qwen-qwq-32b
    """

    def __init__(self, *, api_key: str, timeout: float = 45.0) -> None:
        super().__init__(api_key=api_key, base_url="https://api.groq.com/openai/v1", timeout=timeout)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


class DeepSeekProvider(OpenRouterProvider):
    """
    DeepSeek direct API — OpenAI-compatible, generous free tier.
    Model IDs: deepseek-chat (V3), deepseek-reasoner (R1)
    """

    def __init__(self, *, api_key: str, timeout: float = 45.0) -> None:
        super().__init__(api_key=api_key, base_url="https://api.deepseek.com/v1", timeout=timeout)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


class MultiProviderGateway:
    """
    Routes model requests across multiple providers in priority order.

    Model ID format:
      "groq:llama-3.3-70b-versatile"       → Groq provider, model "llama-3.3-70b-versatile"
      "deepseek:deepseek-chat"              → DeepSeek direct, model "deepseek-chat"
      "meta-llama/llama-3.3-70b-instruct:free" → OpenRouter (no prefix = openrouter)

    Providers whose keys are not configured are silently skipped.
    """

    def __init__(self, providers: dict[str, BaseLLMProvider | None]) -> None:
        self._providers: dict[str, BaseLLMProvider] = {
            name: p for name, p in providers.items() if p is not None
        }

    def _resolve(self, model_id: str) -> tuple[BaseLLMProvider | None, str]:
        """Parse 'provider:model' or plain model_id → (provider, clean_model_id)."""
        parts = model_id.split(":", 1)
        if len(parts) == 2 and parts[0] in self._providers:
            return self._providers[parts[0]], parts[1]
        openrouter = self._providers.get("openrouter")
        return openrouter, model_id

    def is_available(self, model_id: str) -> bool:
        provider, _ = self._resolve(model_id)
        return provider is not None

    async def achat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> UnifiedLLMResponse:
        provider, clean_model = self._resolve(model)
        if provider is None:
            raise LLMGatewayError(f"No provider available for model: {model}", status_code=None)
        return await provider.achat(
            model=clean_model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def achat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        provider, clean_model = self._resolve(model)
        if provider is None:
            raise LLMGatewayError(f"No provider available for model: {model}", status_code=None)
        async for chunk in provider.achat_stream(
            model=clean_model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk


def build_gateway() -> MultiProviderGateway:
    """
    Construct the shared MultiProviderGateway from environment keys.
    Providers whose key is absent or empty are silently excluded so the
    gateway degrades gracefully on environments that only have OpenRouter.
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    return MultiProviderGateway({
        "openrouter": OpenRouterProvider(api_key=openrouter_key) if openrouter_key else None,
        "groq": GroqProvider(api_key=groq_key) if groq_key else None,
        "deepseek": DeepSeekProvider(api_key=deepseek_key) if deepseek_key else None,
    })


def _chunk_from_openai_delta(data: dict[str, Any]) -> StreamChunk:
    choices: list[StreamChoice] = []
    for choice in data.get("choices", []):
        delta = choice.get("delta") or {}
        tool_calls: list[ToolCallDelta] = []
        for call in delta.get("tool_calls") or []:
            function = call.get("function") or {}
            tool_calls.append(
                ToolCallDelta(
                    index=int(call.get("index") or 0),
                    id=call.get("id"),
                    function=DeltaFunction(
                        name=function.get("name"),
                        arguments=function.get("arguments"),
                    ),
                )
            )
        choices.append(
            StreamChoice(
                delta=ChoiceDelta(
                    content=delta.get("content"),
                    tool_calls=tool_calls or None,
                ),
                finish_reason=choice.get("finish_reason"),
            )
        )
    return StreamChunk(choices=choices)
