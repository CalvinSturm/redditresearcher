from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .config import LLMConfig
from .models import LLMGenerationResult


class LLMClientError(RuntimeError):
    """Raised when LLM provider access fails."""


@dataclass(frozen=True)
class LLMModelInfo:
    id: str
    owned_by: str | None = None


class LMStudioClient:
    def __init__(
        self,
        config: LLMConfig,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if config.provider != "lmstudio":
            raise ValueError("LMStudioClient requires provider='lmstudio'")

        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        self._config = config
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(config.request_timeout_seconds),
            transport=transport,
            headers=headers,
        )

    async def __aenter__(self) -> "LMStudioClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    async def list_models(self) -> list[LLMModelInfo]:
        response = await self._http.get(f"{self._config.base_url}/models")
        response.raise_for_status()
        payload = response.json()
        items = payload.get("data")
        if not isinstance(items, list):
            raise LLMClientError("LM Studio /models response missing data list")

        models: list[LLMModelInfo] = []
        for item in items:
            model_id = item.get("id")
            if not model_id:
                continue
            models.append(
                LLMModelInfo(
                    id=str(model_id),
                    owned_by=str(item.get("owned_by")) if item.get("owned_by") is not None else None,
                )
            )
        return models

    async def generate_text(self, prompt: str, model: str | None = None) -> str:
        result = await self.generate_response(prompt, model=model)
        return result.output_text

    async def generate_response(
        self,
        prompt: str,
        model: str | None = None,
    ) -> LLMGenerationResult:
        selected_model = model or self._config.model
        if not selected_model:
            raise LLMClientError("No model configured. Set LLM_MODEL or pass --model.")

        response = await self._http.post(
            f"{self._config.base_url}/responses",
            json={"model": selected_model, "input": prompt},
        )
        response.raise_for_status()
        payload = response.json()
        text = extract_response_text(payload)
        if not text:
            raise LLMClientError("LM Studio response did not contain output text")
        return LLMGenerationResult(
            provider=self._config.provider,
            model=selected_model,
            prompt=prompt,
            output_text=text,
            raw_response=payload,
        )


def extract_response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output")
    if not isinstance(output, list):
        return ""

    texts: list[str] = []
    for item in output:
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "output_text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
    return "\n".join(texts).strip()
