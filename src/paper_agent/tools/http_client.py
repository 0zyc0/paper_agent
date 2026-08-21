from __future__ import annotations

import http.client
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterator


class HttpError(RuntimeError):
    pass


def get_text(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 25,
    retries: int = 2,
) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "paper-related-work-agent/0.1",
            **(headers or {}),
        },
    )
    last_transient_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise HttpError(f"HTTP {exc.code} for {url}: {body[:300]}") from exc
        except urllib.error.URLError as exc:
            raise HttpError(f"Could not fetch {url}: {exc.reason}") from exc
        except (http.client.IncompleteRead, ssl.SSLError, TimeoutError, OSError) as exc:
            last_transient_error = exc
            if attempt >= retries:
                break
            time.sleep(0.8 * (attempt + 1))
    raise HttpError(f"Could not fetch {url}: {last_transient_error}") from last_transient_error


def get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 25,
    retries: int = 2,
) -> dict:
    return json.loads(get_text(url, headers=headers, timeout=timeout, retries=retries))


def post_json(
    url: str,
    payload: dict,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "User-Agent": "paper-related-work-agent/0.1",
            "Content-Type": "application/json",
            **(headers or {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise HttpError(f"HTTP {exc.code} for {url}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise HttpError(f"Could not post to {url}: {exc.reason}") from exc
    except (ssl.SSLError, TimeoutError, OSError) as exc:
        raise HttpError(f"Could not post to {url}: {exc}") from exc


def post_sse_json(
    url: str,
    payload: dict,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 300,
) -> Iterator[dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "User-Agent": "paper-related-work-agent/0.1",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **(headers or {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data_lines: list[str] = []
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    if data_lines:
                        data = "\n".join(data_lines)
                        data_lines = []
                        if data == "[DONE]":
                            break
                        yield json.loads(data)
                    continue
                if line.startswith("data:"):
                    data_lines.append(line.removeprefix("data:").strip())
            if data_lines:
                data = "\n".join(data_lines)
                if data != "[DONE]":
                    yield json.loads(data)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise HttpError(f"HTTP {exc.code} for {url}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise HttpError(f"Could not stream from {url}: {exc.reason}") from exc
    except (ssl.SSLError, TimeoutError, OSError) as exc:
        raise HttpError(f"Could not stream from {url}: {exc}") from exc


def with_query(base_url: str, params: dict[str, object]) -> str:
    clean_params = {key: value for key, value in params.items() if value is not None}
    return f"{base_url}?{urllib.parse.urlencode(clean_params)}"
