from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class LLMClientConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_s: float = 120.0
    trust_env: bool = True

    @classmethod
    def from_env(cls) -> "LLMClientConfig":
        base_url = (
            os.getenv("STAGE1_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("X2_AGENT_LLM_BASE_URL")
            or os.getenv("AXONHUB_BASE_URL")
            or "https://ai.zxcoding.top/v1"
        )
        api_key = (
            os.getenv("STAGE1_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("X2_AGENT_LLM_API_KEY")
            or os.getenv("AXONHUB_API_KEY")
        )
        model = (
            os.getenv("STAGE1_LLM_MODEL")
            or os.getenv("OPENAI_MODEL")
            or os.getenv("X2_AGENT_LLM_MODEL")
            or "gpt-5.5"
        )
        if not base_url:
            raise RuntimeError("Missing STAGE1_LLM_BASE_URL or OPENAI_BASE_URL.")
        if not api_key:
            raise RuntimeError("Missing STAGE1_LLM_API_KEY or OPENAI_API_KEY.")
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=float(os.getenv("STAGE1_LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("STAGE1_LLM_MAX_TOKENS", "2048")),
            timeout_s=float(os.getenv("STAGE1_LLM_TIMEOUT_S", "120")),
            trust_env=os.getenv("STAGE1_LLM_TRUST_ENV", "1").lower() not in {"0", "false", "no"},
        )


class OpenAICompatibleClient:
    def __init__(self, config: LLMClientConfig):
        self.config = config

    def chat(self, messages: list[dict[str, str]]) -> str:
        url = self._chat_url(self.config.base_url)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        start = time.time()
        session = requests.Session()
        session.trust_env = self.config.trust_env
        response = session.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.config.timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response format: {body}") from exc
        finally:
            _ = time.time() - start

    @staticmethod
    def _chat_url(base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"


def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()
