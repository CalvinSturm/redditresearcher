from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any

import httpx

from .config import RuntimeConfig
from .models import RateLimitSnapshot, RequestLogEntry, SearchRequestSpec


class RedditClientError(RuntimeError):
    """Raised when Reddit API access fails."""


class RedditClient:
    def __init__(
        self,
        config: RuntimeConfig,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(config.request_timeout_seconds),
            transport=transport,
            headers={"User-Agent": config.reddit_user_agent},
        )
        self._token: str | None = None
        self._token_expires_at: datetime | None = None

    async def __aenter__(self) -> "RedditClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    async def search_subreddit(
        self,
        spec: SearchRequestSpec,
    ) -> tuple[dict[str, Any], RequestLogEntry]:
        params = {
            "q": spec.query,
            "sort": spec.sort,
            "t": spec.time_filter,
            "limit": str(spec.limit),
            "restrict_sr": "true",
            "raw_json": "1",
            "type": "link",
        }
        if spec.after:
            params["after"] = spec.after

        return await self._request(
            method="GET",
            url=f"{self._config.api_base_url}/r/{spec.subreddit}/search",
            params=params,
            request_name=f"search:{spec.subreddit}",
        )

    async def _request(
        self,
        method: str,
        url: str,
        params: dict[str, str],
        request_name: str,
    ) -> tuple[dict[str, Any], RequestLogEntry]:
        last_error: Exception | None = None
        for attempt in range(1, self._config.max_retries + 1):
            started_at = datetime.now(UTC)
            started_perf = perf_counter()
            try:
                token = await self._get_access_token()
                response = await self._http.request(
                    method,
                    url,
                    params=params,
                    headers={"Authorization": f"Bearer {token}"},
                )
                duration_ms = round((perf_counter() - started_perf) * 1000, 2)
                rate_limit = _rate_limit_from_headers(response.headers)
                log_entry = RequestLogEntry(
                    requested_at=started_at,
                    request_name=request_name,
                    method=method,
                    url=str(response.request.url),
                    params=params,
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                    attempt=attempt,
                    rate_limit=rate_limit,
                )

                if response.status_code == 401 and attempt < self._config.max_retries:
                    self._token = None
                    self._token_expires_at = None
                    await asyncio.sleep(_retry_delay_seconds(attempt))
                    continue

                if response.status_code in {429, 500, 502, 503, 504}:
                    if attempt < self._config.max_retries:
                        await asyncio.sleep(_retry_delay_seconds(attempt, rate_limit))
                        continue
                    raise RedditClientError(
                        f"{request_name} failed with status {response.status_code}"
                    )

                response.raise_for_status()
                return response.json(), log_entry
            except (httpx.HTTPError, RedditClientError) as exc:
                last_error = exc
                if attempt >= self._config.max_retries:
                    break
                await asyncio.sleep(_retry_delay_seconds(attempt))

        raise RedditClientError(f"{request_name} failed after retries: {last_error}")

    async def _get_access_token(self) -> str:
        now = datetime.now(UTC)
        if self._token and self._token_expires_at and now < self._token_expires_at:
            return self._token

        response = await self._http.post(
            f"{self._config.oauth_base_url}/api/v1/access_token",
            data={"grant_type": "client_credentials"},
            auth=(self._config.reddit_client_id, self._config.reddit_client_secret),
        )
        response.raise_for_status()
        payload = response.json()
        access_token = payload.get("access_token")
        expires_in = payload.get("expires_in")
        if not access_token or not expires_in:
            raise RedditClientError("OAuth token response missing access_token or expires_in")

        self._token = str(access_token)
        self._token_expires_at = now + timedelta(seconds=max(int(expires_in) - 60, 0))
        return self._token


def _rate_limit_from_headers(headers: httpx.Headers) -> RateLimitSnapshot | None:
    used = _maybe_float(headers.get("x-ratelimit-used"))
    remaining = _maybe_float(headers.get("x-ratelimit-remaining"))
    reset_seconds = _maybe_float(headers.get("x-ratelimit-reset"))
    if used is None and remaining is None and reset_seconds is None:
        return None
    return RateLimitSnapshot(used=used, remaining=remaining, reset_seconds=reset_seconds)


def _maybe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _retry_delay_seconds(
    attempt: int,
    rate_limit: RateLimitSnapshot | None = None,
) -> float:
    if rate_limit and rate_limit.remaining is not None and rate_limit.remaining <= 0:
        if rate_limit.reset_seconds:
            return min(rate_limit.reset_seconds, 30.0)
    return min(0.5 * (2 ** (attempt - 1)), 8.0)
