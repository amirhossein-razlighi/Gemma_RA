from __future__ import annotations

import json
from typing import Any

import httpx

from gemma_ra.core.config import OllamaConfig
from gemma_ra.core.exceptions import ModelError


class OllamaClient:
    def __init__(self, config: OllamaConfig) -> None:
        self.config = config

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_chunk: callable | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": 0.2},
        }
        if tools:
            payload["tools"] = tools
        if not stream:
            try:
                response = httpx.post(
                    f"{self.config.host}/api/chat",
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ModelError(f"Failed to call Ollama at {self.config.host}: {exc}") from exc
            return response.json()

        final_message: dict[str, Any] = {"role": "assistant", "content": "", "thinking": "", "tool_calls": []}
        try:
            with httpx.stream(
                "POST",
                f"{self.config.host}/api/chat",
                json=payload,
                timeout=self.config.timeout_seconds,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    message = chunk.get("message", {})
                    if not final_message.get("role") and message.get("role"):
                        final_message["role"] = message["role"]
                    content_delta = message.get("content", "")
                    thinking_delta = message.get("thinking", "")
                    if content_delta:
                        final_message["content"] += content_delta
                        if on_chunk is not None:
                            on_chunk("content", content_delta)
                    if thinking_delta:
                        final_message["thinking"] += thinking_delta
                        if on_chunk is not None:
                            on_chunk("thinking", thinking_delta)
                    tool_calls = message.get("tool_calls") or []
                    if tool_calls:
                        final_message["tool_calls"] = tool_calls
                    if chunk.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise ModelError(f"Failed to call Ollama at {self.config.host}: {exc}") from exc
        return {"message": final_message}

    def generate_structured(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0.2},
        }
        try:
            response = httpx.post(
                f"{self.config.host}/api/generate",
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelError(f"Failed to call Ollama at {self.config.host}: {exc}") from exc

        raw = response.json().get("response", "")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelError("Ollama returned non-JSON structured output.") from exc
