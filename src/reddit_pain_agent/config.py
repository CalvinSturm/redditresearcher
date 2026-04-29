from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
import re


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "outputs" / "runs"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_CONCURRENT_REQUESTS = 4
DEFAULT_SORT = "relevance"
DEFAULT_TIME_FILTER = "all"
DEFAULT_LIMIT = 25
DEFAULT_PAGES_PER_QUERY = 2
DEFAULT_EXPAND_QUERIES = True
ALLOWED_SORTS = {"comments", "hot", "new", "relevance", "top"}
ALLOWED_TIME_FILTERS = {"all", "day", "hour", "month", "week", "year"}
DEFAULT_LLM_PROVIDER = "lmstudio"
DEFAULT_LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 60.0


class ConfigurationError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-")


def repo_paths(root: Path | None = None) -> dict[str, Path]:
    repo_root = root or DEFAULT_ROOT
    research_dir = repo_root / "research"
    outputs_dir = repo_root / "outputs"
    return {
        "repo_root": repo_root,
        "research_dir": research_dir,
        "briefs_dir": research_dir / "briefs",
        "queries_dir": research_dir / "queries",
        "templates_dir": repo_root / "templates",
        "outputs_dir": outputs_dir,
        "runs_dir": outputs_dir / "runs",
    }


def ensure_repo_layout(root: Path | None = None) -> dict[str, Path]:
    paths = repo_paths(root)
    for key, path in paths.items():
        if key == "repo_root":
            continue
        path.mkdir(parents=True, exist_ok=True)
    return paths


@dataclass(frozen=True)
class RunPaths:
    slug: str
    brief_path: Path
    output_dir: Path
    candidate_posts_path: Path
    selected_posts_path: Path
    theme_summary_path: Path
    final_memo_path: Path


def build_run_paths(run_name: str, root: Path | None = None) -> RunPaths:
    slug = slugify(run_name)
    if not slug:
        raise ValueError("run name must contain at least one letter or number")

    paths = repo_paths(root)
    output_dir = paths["runs_dir"] / slug
    return RunPaths(
        slug=slug,
        brief_path=paths["briefs_dir"] / f"{slug}.md",
        output_dir=output_dir,
        candidate_posts_path=output_dir / "candidate_posts.json",
        selected_posts_path=output_dir / "selected_posts.json",
        theme_summary_path=output_dir / "theme_summary.json",
        final_memo_path=output_dir / "final_memo.md",
    )


@dataclass(frozen=True)
class RuntimeConfig:
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str
    output_root: Path
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS
    oauth_base_url: str = "https://www.reddit.com"
    api_base_url: str = "https://oauth.reddit.com"

    def public_settings(self) -> dict[str, object]:
        return {
            "reddit_client_id": self.reddit_client_id,
            "reddit_user_agent": self.reddit_user_agent,
            "output_root": str(self.output_root),
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_retries": self.max_retries,
            "max_concurrent_requests": self.max_concurrent_requests,
            "oauth_base_url": self.oauth_base_url,
            "api_base_url": self.api_base_url,
        }


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model: str | None
    api_key: str | None
    request_timeout_seconds: float = DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS

    def public_settings(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "request_timeout_seconds": self.request_timeout_seconds,
            "api_key_configured": bool(self.api_key),
        }


def _read_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


def _parse_positive_float(raw: str | None, env_name: str, default: float) -> float:
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{env_name} must be a number") from exc
    if value <= 0:
        raise ConfigurationError(f"{env_name} must be greater than 0")
    return value


def _parse_positive_int(raw: str | None, env_name: str, default: int) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{env_name} must be an integer") from exc
    if value <= 0:
        raise ConfigurationError(f"{env_name} must be greater than 0")
    return value


def load_runtime_config(output_root_override: str | Path | None = None) -> RuntimeConfig:
    client_id = _read_env("REDDIT_CLIENT_ID")
    user_agent = _read_env("REDDIT_USER_AGENT")
    missing = [
        name
        for name, value in (
            ("REDDIT_CLIENT_ID", client_id),
            ("REDDIT_USER_AGENT", user_agent),
        )
        if value is None
    ]
    if missing:
        raise ConfigurationError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    output_root = Path(output_root_override) if output_root_override else Path(
        _read_env("REDDIT_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)) or str(DEFAULT_OUTPUT_ROOT)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    return RuntimeConfig(
        reddit_client_id=client_id,
        reddit_client_secret=_read_env("REDDIT_CLIENT_SECRET", "") or "",
        reddit_user_agent=user_agent,
        output_root=output_root,
        request_timeout_seconds=_parse_positive_float(
            _read_env("REDDIT_REQUEST_TIMEOUT_SECONDS"),
            "REDDIT_REQUEST_TIMEOUT_SECONDS",
            DEFAULT_REQUEST_TIMEOUT_SECONDS,
        ),
        max_retries=_parse_positive_int(
            _read_env("REDDIT_MAX_RETRIES"),
            "REDDIT_MAX_RETRIES",
            DEFAULT_MAX_RETRIES,
        ),
        max_concurrent_requests=_parse_positive_int(
            _read_env("REDDIT_MAX_CONCURRENT_REQUESTS"),
            "REDDIT_MAX_CONCURRENT_REQUESTS",
            DEFAULT_MAX_CONCURRENT_REQUESTS,
        ),
    )


def load_llm_config(require_model: bool = False) -> LLMConfig:
    provider = (_read_env("LLM_PROVIDER", DEFAULT_LLM_PROVIDER) or DEFAULT_LLM_PROVIDER).lower()
    if provider not in {"lmstudio", "openai"}:
        raise ConfigurationError(
            "Unsupported LLM_PROVIDER. Supported values: lmstudio, openai"
        )

    model = _read_env("LLM_MODEL")
    if require_model and model is None:
        raise ConfigurationError("Missing required environment variable: LLM_MODEL")

    default_base_url = (
        DEFAULT_LMSTUDIO_BASE_URL if provider == "lmstudio" else DEFAULT_OPENAI_BASE_URL
    )
    base_url = _read_env("LLM_BASE_URL", default_base_url) or default_base_url
    api_key = _read_env("LLM_API_KEY")
    if provider == "openai" and api_key is None:
        api_key = _read_env("OPENAI_API_KEY")
    if provider == "openai" and api_key is None:
        raise ConfigurationError(
            "Missing required environment variable: OPENAI_API_KEY or LLM_API_KEY"
        )
    return LLMConfig(
        provider=provider,
        base_url=base_url.rstrip("/"),
        model=model,
        api_key=api_key,
        request_timeout_seconds=_parse_positive_float(
            _read_env("LLM_REQUEST_TIMEOUT_SECONDS"),
            "LLM_REQUEST_TIMEOUT_SECONDS",
            DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
        ),
    )


def build_search_run_slug(
    subreddits: list[str],
    queries: list[str],
    now: datetime | None = None,
) -> str:
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    subreddit_part = slugify("-".join(subreddits[:2])) or "reddit"
    query_part = slugify(queries[0])[:32] if queries else "search"
    return "-".join(piece for piece in ["search", timestamp, subreddit_part, query_part] if piece)
