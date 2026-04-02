from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifact_store import ArtifactStore, build_artifact_store
from .config import (
    ALLOWED_SORTS,
    ALLOWED_TIME_FILTERS,
    DEFAULT_LIMIT,
    DEFAULT_SORT,
    DEFAULT_TIME_FILTER,
    RuntimeConfig,
    build_search_run_slug,
)
from .models import CandidatePost, RunManifest, SearchRequestSpec, Submission
from .reddit_client import RedditClient


@dataclass(frozen=True)
class SearchRunResult:
    run_slug: str
    run_dir: Path
    request_count: int
    candidate_count: int
    filtered_counts: dict[str, int]
    raw_search_artifacts: list[str]


def build_search_specs(
    subreddits: list[str],
    queries: list[str],
    sort: str = DEFAULT_SORT,
    time_filter: str = DEFAULT_TIME_FILTER,
    limit: int = DEFAULT_LIMIT,
) -> list[SearchRequestSpec]:
    if not subreddits:
        raise ValueError("at least one subreddit is required")
    if not queries:
        raise ValueError("at least one query is required")
    if sort not in ALLOWED_SORTS:
        raise ValueError(f"sort must be one of: {', '.join(sorted(ALLOWED_SORTS))}")
    if time_filter not in ALLOWED_TIME_FILTERS:
        raise ValueError(
            f"time filter must be one of: {', '.join(sorted(ALLOWED_TIME_FILTERS))}"
        )
    if limit <= 0:
        raise ValueError("limit must be greater than 0")

    return [
        SearchRequestSpec(
            subreddit=subreddit,
            query=query,
            sort=sort,
            time_filter=time_filter,
            limit=min(limit, 100),
        )
        for subreddit in subreddits
        for query in queries
    ]


async def run_search_command(
    config: RuntimeConfig,
    subreddits: list[str],
    queries: list[str],
    sort: str = DEFAULT_SORT,
    time_filter: str = DEFAULT_TIME_FILTER,
    limit: int = DEFAULT_LIMIT,
    output_dir: Path | None = None,
    client: RedditClient | None = None,
) -> SearchRunResult:
    specs = build_search_specs(subreddits, queries, sort=sort, time_filter=time_filter, limit=limit)
    run_slug = build_search_run_slug(subreddits, queries)
    store = build_artifact_store(config.output_root, run_slug, output_dir)
    manifest = RunManifest(
        run_slug=run_slug,
        status="running",
        started_at=datetime.now(UTC),
        output_dir=str(store.run_dir),
        subreddits=subreddits,
        queries=queries,
        sort=sort,
        time_filter=time_filter,
        limit=limit,
        request_timeout_seconds=config.request_timeout_seconds,
        max_retries=config.max_retries,
        max_concurrent_requests=config.max_concurrent_requests,
        warnings=[
            "Comment retrieval, ranking, clustering, and memo synthesis are not implemented yet.",
            "Stored post text should be treated as inspectable run data, not permanent archival data.",
        ],
    )
    store.write_manifest(manifest)

    owns_client = client is None
    reddit_client = client or RedditClient(config)
    try:
        if owns_client:
            await reddit_client.__aenter__()
        result = await _execute_specs(reddit_client, specs, store, manifest, config)
    except Exception:
        manifest.status = "failed"
        manifest.completed_at = datetime.now(UTC)
        store.write_manifest(manifest)
        raise
    finally:
        if owns_client:
            await reddit_client.__aexit__(None, None, None)
    return result


async def _execute_specs(
    client: RedditClient,
    specs: list[SearchRequestSpec],
    store: ArtifactStore,
    manifest: RunManifest,
    config: RuntimeConfig,
) -> SearchRunResult:
    semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def fetch(spec: SearchRequestSpec):
        async with semaphore:
            payload, log_entry = await client.search_subreddit(spec)
            return spec, payload, log_entry

    responses = await asyncio.gather(*(fetch(spec) for spec in specs))
    artifacts: list[str] = []
    filtered = Counter()
    candidates_by_id: dict[str, CandidatePost] = {}

    for index, (spec, payload, log_entry) in enumerate(responses, start=1):
        artifact_path = store.write_raw_search_payload(index, spec, payload)
        artifacts.append(artifact_path)
        log_entry.raw_artifact_path = artifact_path
        store.append_request_log(log_entry)

        children = (payload.get("data") or {}).get("children") or []
        for child in children:
            candidate, reason = normalize_candidate(child, spec)
            if candidate is None:
                filtered[reason] += 1
                continue
            existing = candidates_by_id.get(candidate.id)
            if existing is None:
                candidates_by_id[candidate.id] = candidate
                continue
            filtered["duplicate"] += 1
            _merge_candidate(existing, candidate)

    candidates = sorted(
        candidates_by_id.values(),
        key=lambda item: ((item.num_comments or 0), (item.score or 0), (item.created_utc or 0)),
        reverse=True,
    )
    store.write_candidate_posts(candidates)

    manifest.status = "completed"
    manifest.completed_at = datetime.now(UTC)
    manifest.request_count = len(responses)
    manifest.raw_search_artifacts = artifacts
    manifest.candidate_count = len(candidates)
    manifest.filtered_counts = dict(filtered)
    store.write_manifest(manifest)

    return SearchRunResult(
        run_slug=manifest.run_slug,
        run_dir=store.run_dir,
        request_count=len(responses),
        candidate_count=len(candidates),
        filtered_counts=dict(filtered),
        raw_search_artifacts=artifacts,
    )


def normalize_candidate(
    child: dict[str, Any],
    spec: SearchRequestSpec,
) -> tuple[CandidatePost | None, str]:
    if child.get("kind") != "t3":
        return None, "non_submission"

    data = child.get("data") or {}
    submission_id = data.get("id")
    if submission_id is None:
        return None, "missing_id"

    submission = Submission(
        id=str(submission_id),
        title=str(data.get("title") or ""),
        subreddit=str(data.get("subreddit") or spec.subreddit),
        url=str(data.get("url") or ""),
        permalink=_normalize_permalink(data.get("permalink")),
        score=_to_int(data.get("score")),
        num_comments=_to_int(data.get("num_comments")),
        created_utc=_to_float(data.get("created_utc")),
        selftext=str(data.get("selftext") or ""),
        author=str(data.get("author")) if data.get("author") is not None else None,
        is_self=_to_bool(data.get("is_self")),
        over_18=_to_bool(data.get("over_18")),
        removed_by_category=str(data.get("removed_by_category"))
        if data.get("removed_by_category") is not None
        else None,
    )
    if _is_deleted_submission(submission):
        return None, "deleted"
    if _is_empty_submission(submission):
        return None, "empty"

    return CandidatePost(
        id=submission.id,
        title=submission.title,
        subreddit=submission.subreddit,
        url=submission.url,
        permalink=submission.permalink,
        score=submission.score,
        num_comments=submission.num_comments,
        created_utc=submission.created_utc,
        selftext=submission.selftext,
        author=submission.author,
        source_queries=[spec.query],
        source_subreddits=[spec.subreddit],
        retrieval_requests=[spec.request_key],
    ), ""


def _merge_candidate(existing: CandidatePost, candidate: CandidatePost) -> None:
    for value in candidate.source_queries:
        if value not in existing.source_queries:
            existing.source_queries.append(value)
    for value in candidate.source_subreddits:
        if value not in existing.source_subreddits:
            existing.source_subreddits.append(value)
    for value in candidate.retrieval_requests:
        if value not in existing.retrieval_requests:
            existing.retrieval_requests.append(value)


def _normalize_permalink(value: Any) -> str | None:
    if value is None:
        return None
    permalink = str(value).strip()
    if not permalink:
        return None
    if permalink.startswith("http://") or permalink.startswith("https://"):
        return permalink
    return f"https://www.reddit.com{permalink}"


def _is_deleted_submission(submission: Submission) -> bool:
    deleted_markers = {"[deleted]", "[removed]", "[deleted by user]"}
    return submission.title.strip().lower() in deleted_markers or submission.url.strip().lower() in deleted_markers


def _is_empty_submission(submission: Submission) -> bool:
    return not submission.title.strip() and not submission.selftext.strip() and not submission.url.strip()


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    return bool(value)
