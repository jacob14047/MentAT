"""
Client concreto per LM Studio.

LM Studio, quando avvii il "Local Server" (tab Developer), espone un'API
REST compatibile con lo standard OpenAI chat-completions su
`http://localhost:1234/v1/chat/completions`. Ci basta quindi un client HTTP
minimale: nessuna dipendenza dall'SDK di OpenAI.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from llm.base_client import BaseLLMClient, Message


class LMStudioClient(BaseLLMClient):
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model_name: str = "qwen/qwen3.6-35b-a3b",
        api_key: str = "lm-studio",
        temperature: float = 0.2,
        max_tokens: int = 16384,
        max_reasoning_tokens: int = 2048,
        request_timeout_seconds: int = 500,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_reasoning_tokens = max_reasoning_tokens
        self.timeout = request_timeout_seconds
        self._last_usage: Optional[Dict[str, Any]] = None

    @property
    def last_usage(self) -> Optional[Dict[str, Any]]:
        return self._last_usage

    def chat(self, messages: List[Message], **kwargs: Any) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model_name),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "max_reasoning_tokens": kwargs.get("max_reasoning_tokens", self.max_reasoning_tokens),
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ConnectionError(
                f"Impossibile contattare LM Studio su {url}. "
                f"Verifica che il Local Server sia avviato in LM Studio. "
                f"Dettaglio: {exc}"
            ) from exc

        data = response.json()
        self._last_usage = data.get("usage")
        return self._extract_content(data)

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"Risposta inattesa da LM Studio: {data}"
            ) from exc

    @classmethod
    def from_settings(cls, llm_settings) -> "LMStudioClient":
        """Factory di comodo a partire da un oggetto LLMSettings (config/settings.py)."""
        return cls(
            base_url=llm_settings.base_url,
            model_name=llm_settings.model_name,
            api_key=llm_settings.api_key,
            temperature=llm_settings.temperature,
            max_tokens=llm_settings.max_tokens,
            max_reasoning_tokens=llm_settings.max_reasoning_tokens,
            request_timeout_seconds=llm_settings.request_timeout_seconds,
        )
