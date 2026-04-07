"""
API-agnostic LLM client for synthetic data generation.
Supports OpenAI, Anthropic, and local vLLM endpoints.
"""

import os
import json
import time
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.7


class LLMClient:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = self._init_client()
        return self._client

    def _init_client(self):
        if self.config.provider in ("openai", "local"):
            from openai import OpenAI
            kwargs = {}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            return OpenAI(**kwargs)

        elif self.config.provider == "anthropic":
            from anthropic import Anthropic
            kwargs = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            return Anthropic(**kwargs)

        raise ValueError(f"Unknown provider: {self.config.provider}")

    def generate(self, system: str, user: str, retries: int = 3) -> str:
        """Generate a response with retry logic."""
        for attempt in range(retries):
            try:
                return self._call(system, user)
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/{retries} after {wait}s: {e}")
                time.sleep(wait)

    def _call(self, system: str, user: str) -> str:
        if self.config.provider in ("openai", "local"):
            resp = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            return resp.choices[0].message.content

        elif self.config.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.config.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            return resp.content[0].text

    def generate_json(self, system: str, user: str, retries: int = 3) -> dict:
        """Generate and parse a JSON response."""
        raw = self.generate(system, user, retries)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        return json.loads(cleaned)
