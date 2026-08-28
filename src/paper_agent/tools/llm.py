from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterator

from .http_client import post_json, post_sse_json

try:
    from .. import local_config
except ImportError:  # pragma: no cover - local config is optional
    local_config = None


class KimiClient:
    """Small OpenAI-compatible client for Moonshot/Kimi chat completions."""

    _call_log: list[dict[str, Any]] = []
    _max_call_log = 200

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        api_base: str | None = None,
    ) -> None:
        local_kimi_key = getattr(local_config, "KIMI_API_KEY", "") if local_config else ""
        local_moonshot_key = getattr(local_config, "MOONSHOT_API_KEY", "") if local_config else ""
        local_model = getattr(local_config, "KIMI_MODEL", "kimi-k2.6") if local_config else "kimi-k2.6"
        local_api_base = getattr(local_config, "KIMI_API_BASE", "https://api.moonshot.cn/v1") if local_config else "https://api.moonshot.cn/v1"
        self.api_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY") or local_kimi_key or local_moonshot_key
        self.model = model or os.getenv("KIMI_MODEL") or local_model or "kimi-k2.6"
        self.api_base = (api_base or os.getenv("KIMI_API_BASE") or local_api_base or "https://api.moonshot.cn/v1").rstrip("/")
        self.temperature = _allowed_temperature(self.model)

    @classmethod
    def log_size(cls) -> int:
        return len(cls._call_log)

    @classmethod
    def calls_since(cls, index: int) -> list[dict[str, Any]]:
        return [dict(item) for item in cls._call_log[index:]]

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 2500,
        timeout: int = 300,
        stream: bool = True,
        label: str = "",
    ) -> str:
        if not self.api_key:
            raise RuntimeError("Kimi API key is not configured. Set KIMI_API_KEY or MOONSHOT_API_KEY.")
        entry = self._start_call(label=label or "chat_text", stream=stream, max_tokens=max_tokens)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if stream:
            payload["stream"] = True
            chunks: list[str] = []
            try:
                for event in post_sse_json(
                    f"{self.api_base}/chat/completions",
                    payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=timeout,
                ):
                    usage = event.get("usage")
                    if usage is not None:
                        entry["usage"] = usage
                    choice = (event.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        chunks.append(content)
            except Exception as exc:
                self._fail_call(entry, exc)
                raise
            result = "".join(chunks)
            self._finish_call(entry, response_text=result)
            return result

        try:
            response = post_json(
                f"{self.api_base}/chat/completions",
                payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=timeout,
            )
        except Exception as exc:
            self._fail_call(entry, exc)
            raise
        entry["usage"] = response.get("usage")
        result = response["choices"][0]["message"]["content"]
        self._finish_call(entry, response_text=result)
        return result

    def stream_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 2500,
        timeout: int = 300,
        label: str = "",
    ) -> Iterator[str]:
        """Yield completion text deltas as they arrive from Kimi.

        ``chat_text(stream=True)`` is intentionally kept as the convenient
        collect-then-return API used by existing tools.  The web Agent needs a
        real iterator so it can forward each delta to the browser immediately.
        """
        if not self.api_key:
            raise RuntimeError("Kimi API key is not configured. Set KIMI_API_KEY or MOONSHOT_API_KEY.")
        entry = self._start_call(label=label or "stream_text", stream=True, max_tokens=max_tokens)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        chunks: list[str] = []
        try:
            for event in post_sse_json(
                f"{self.api_base}/chat/completions",
                payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=timeout,
            ):
                usage = event.get("usage")
                if usage is not None:
                    entry["usage"] = usage
                choice = (event.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    text = str(content)
                    chunks.append(text)
                    yield text
        except Exception as exc:
            self._fail_call(entry, exc)
            raise
        else:
            self._finish_call(entry, response_text="".join(chunks))

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 1200,
        timeout: int = 120,
        stream: bool = False,
        label: str = "",
    ) -> dict[str, Any]:
        content = self.chat_text(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            stream=stream,
            label=label or "chat_json",
        )
        return _extract_json_object(content)

    def _start_call(self, *, label: str, stream: bool, max_tokens: int) -> dict[str, Any]:
        entry = {
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "label": label,
            "stream": stream,
            "model": self.model,
            "api_base": self.api_base,
            "max_tokens": max_tokens,
            "success": False,
            "error": "",
            "response_chars": 0,
            "usage": None,
        }
        self._call_log.append(entry)
        if len(self._call_log) > self._max_call_log:
            del self._call_log[:-self._max_call_log]
        return entry

    def _finish_call(self, entry: dict[str, Any], *, response_text: str) -> None:
        entry["success"] = True
        entry["response_chars"] = len(response_text or "")

    def _fail_call(self, entry: dict[str, Any], exc: Exception) -> None:
        entry["success"] = False
        entry["error"] = str(exc)


def _extract_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM response is not a JSON object.")
    return value


def _allowed_temperature(model: str) -> float:
    """Kimi K2.x currently rejects arbitrary temperatures and only allows 1."""
    normalized = model.lower()
    if "kimi-k2" in normalized:
        return 1
    return 1
