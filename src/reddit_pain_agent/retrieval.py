from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any

from .artifact_store import ArtifactStore, build_artifact_store
from .config import (
    ALLOWED_SORTS,
    ALLOWED_TIME_FILTERS,
    DEFAULT_LIMIT,
    DEFAULT_PAGES_PER_QUERY,
    DEFAULT_EXPAND_QUERIES,
    DEFAULT_SORT,
    DEFAULT_TIME_FILTER,
    RuntimeConfig,
    build_search_run_slug,
)
from .models import (
    CandidatePost,
    Comment,
    CommentEnrichmentArtifact,
    RunManifest,
    SearchRequestSpec,
    Submission,
    SubmissionCommentsArtifact,
)
from .reddit_client import RedditClient


@dataclass(frozen=True)
class SearchRunResult:
    run_slug: str
    run_dir: Path
    request_count: int
    candidate_count: int
    query_variant_count: int
    search_spec_count: int
    sort_count: int
    time_filter_count: int
    pages_per_query: int
    filtered_counts: dict[str, int]
    raw_search_artifacts: list[str]


@dataclass(frozen=True)
class CommentEnrichmentResult:
    run_dir: Path
    requested_submission_count: int
    fetched_submission_count: int
    comment_count: int
    morechildren_request_count: int
    raw_comment_artifacts: list[str]
    normalized_comment_artifacts: list[str]


@dataclass(frozen=True)
class RetrievalQualityFilters:
    min_score: int = 0
    min_comments: int = 0
    filter_nsfw: bool = False
    allowed_subreddits: tuple[str, ...] = ()
    denied_subreddits: tuple[str, ...] = ()


QUERY_EXPANSION_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def build_search_specs(
    subreddits: list[str],
    queries: list[str],
    sort: str = DEFAULT_SORT,
    time_filter: str = DEFAULT_TIME_FILTER,
    limit: int = DEFAULT_LIMIT,
    pages_per_query: int = DEFAULT_PAGES_PER_QUERY,
    expand_queries: bool = DEFAULT_EXPAND_QUERIES,
    additional_sorts: list[str] | None = None,
    additional_time_filters: list[str] | None = None,
) -> list[SearchRequestSpec]:
    if not subreddits:
        raise ValueError("at least one subreddit is required")
    if not queries:
        raise ValueError("at least one query is required")
    if limit <= 0:
        raise ValueError("limit must be greater than 0")
    if pages_per_query <= 0:
        raise ValueError("pages_per_query must be greater than 0")

    search_sorts = _resolve_search_values(
        primary=sort,
        additional=additional_sorts or [],
        allowed=ALLOWED_SORTS,
        label="sort",
    )
    search_time_filters = _resolve_search_values(
        primary=time_filter,
        additional=additional_time_filters or [],
        allowed=ALLOWED_TIME_FILTERS,
        label="time filter",
    )

    specs: list[SearchRequestSpec] = []
    for subreddit in subreddits:
        for query in queries:
            seed_query = _normalize_query_text(query)
            query_variants = (
                expand_query_variants(seed_query)
                if expand_queries
                else [seed_query]
            )
            for query_variant in query_variants:
                for search_sort in search_sorts:
                    for search_time_filter in search_time_filters:
                        specs.append(
                            SearchRequestSpec(
                                subreddit=subreddit,
                                query=query_variant,
                                seed_query=seed_query,
                                sort=search_sort,
                                time_filter=search_time_filter,
                                limit=min(limit, 100),
                            )
                        )
    return specs


def resolve_search_plan(
    *,
    sort: str,
    time_filter: str,
    additional_sorts: list[str] | None = None,
    additional_time_filters: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    return (
        _resolve_search_values(
            primary=sort,
            additional=additional_sorts or [],
            allowed=ALLOWED_SORTS,
            label="sort",
        ),
        _resolve_search_values(
            primary=time_filter,
            additional=additional_time_filters or [],
            allowed=ALLOWED_TIME_FILTERS,
            label="time filter",
        ),
    )


def build_retrieval_quality_filters(
    *,
    min_score: int = 0,
    min_comments: int = 0,
    filter_nsfw: bool = False,
    allowed_subreddits: list[str] | None = None,
    denied_subreddits: list[str] | None = None,
) -> RetrievalQualityFilters:
    if min_score < 0:
        raise ValueError("min_score must be 0 or greater")
    if min_comments < 0:
        raise ValueError("min_comments must be 0 or greater")

    normalized_allowed = _normalize_subreddit_filters(allowed_subreddits or [])
    normalized_denied = _normalize_subreddit_filters(denied_subreddits or [])
    overlap = sorted(set(normalized_allowed).intersection(normalized_denied))
    if overlap:
        raise ValueError(
            "allowed_subreddits and denied_subreddits must not overlap: "
            + ", ".join(overlap)
        )

    return RetrievalQualityFilters(
        min_score=min_score,
        min_comments=min_comments,
        filter_nsfw=filter_nsfw,
        allowed_subreddits=tuple(normalized_allowed),
        denied_subreddits=tuple(normalized_denied),
    )


def apply_candidate_quality_filters(
    candidate: CandidatePost,
    quality_filters: RetrievalQualityFilters,
) -> str:
    subreddit_key = candidate.subreddit.strip().lower()
    if quality_filters.filter_nsfw and bool(candidate.over_18):
        return "nsfw"
    if quality_filters.allowed_subreddits and subreddit_key not in quality_filters.allowed_subreddits:
        return "non_allowed_subreddit"
    if quality_filters.denied_subreddits and subreddit_key in quality_filters.denied_subreddits:
        return "denied_subreddit"
    if (candidate.score or 0) < quality_filters.min_score:
        return "low_score"
    if (candidate.num_comments or 0) < quality_filters.min_comments:
        return "low_comments"
    return ""


async def enrich_run_with_comments(
    config: RuntimeConfig,
    run_dir: Path,
    *,
    max_posts: int = 5,
    comment_limit: int = 20,
    comment_depth: int = 3,
    comment_sort: str = "best",
    max_morechildren_requests: int = 3,
    morechildren_batch_size: int = 20,
    client: RedditClient | None = None,
) -> CommentEnrichmentResult:
    if max_posts <= 0:
        raise ValueError("max_posts must be greater than 0")
    if comment_limit <= 0:
        raise ValueError("comment_limit must be greater than 0")
    if comment_depth <= 0:
        raise ValueError("comment_depth must be greater than 0")
    if max_morechildren_requests < 0:
        raise ValueError("max_morechildren_requests must be 0 or greater")
    if morechildren_batch_size <= 0:
        raise ValueError("morechildren_batch_size must be greater than 0")

    store = ArtifactStore(run_dir)
    candidates = _load_candidate_posts(run_dir)[:max_posts]

    owns_client = client is None
    reddit_client = client or RedditClient(config)
    try:
        if owns_client:
            await reddit_client.__aenter__()
        semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        async def fetch(post: CandidatePost):
            if not post.permalink:
                return post, None, None
            async with semaphore:
                payload, log_entry = await reddit_client.fetch_submission_comments(
                    post.permalink,
                    sort=comment_sort,
                    limit=comment_limit,
                    depth=comment_depth,
                )
                return post, payload, log_entry

        responses = await asyncio.gather(*(fetch(post) for post in candidates))
    finally:
        if owns_client:
            await reddit_client.__aexit__(None, None, None)

    raw_artifacts: list[str] = []
    normalized_artifacts: list[str] = []
    fetched_submission_count = 0
    total_comment_count = 0
    morechildren_request_count = 0

    for post, payload, log_entry in responses:
        if payload is None or log_entry is None:
            continue
        raw_artifact_path = store.write_raw_comment_payload(post.id, payload)
        raw_artifacts.append(raw_artifact_path)
        log_entry.raw_artifact_path = raw_artifact_path
        store.append_request_log(log_entry)

        comments, morechildren_ids = normalize_comments_payload(payload)
        comment_index: dict[str, Comment] = {comment.id: comment for comment in comments if comment.id}
        queued_morechildren = list(dict.fromkeys(morechildren_ids))
        while queued_morechildren and morechildren_request_count < max_morechildren_requests:
            batch = queued_morechildren[:morechildren_batch_size]
            queued_morechildren = queued_morechildren[morechildren_batch_size:]
            more_payload, more_log_entry = await reddit_client.fetch_more_children(
                link_id=f"t3_{post.id}",
                children=batch,
                sort=comment_sort,
                depth=comment_depth,
            )
            morechildren_request_count += 1
            more_raw_artifact_path = store.write_raw_comment_payload(
                f"{post.id}-more-{morechildren_request_count:03d}",
                more_payload,
            )
            raw_artifacts.append(more_raw_artifact_path)
            more_log_entry.raw_artifact_path = more_raw_artifact_path
            store.append_request_log(more_log_entry)

            expanded_comments, nested_morechildren = normalize_morechildren_payload(more_payload)
            for comment in expanded_comments:
                if comment.id and comment.id not in comment_index:
                    comment_index[comment.id] = comment
            for child_id in nested_morechildren:
                if child_id not in queued_morechildren:
                    queued_morechildren.append(child_id)

        comments = sorted(
            comment_index.values(),
            key=lambda item: ((item.score or 0), -(item.depth or 0)),
            reverse=True,
        )
        total_comment_count += len(comments)
        submission_artifact = SubmissionCommentsArtifact(
            submission_id=post.id,
            subreddit=post.subreddit,
            permalink=post.permalink,
            title=post.title,
            fetched_comment_count=len(comments),
            comments=comments,
        )
        normalized_artifact_path = store.write_submission_comments(submission_artifact)
        normalized_artifacts.append(normalized_artifact_path)
        fetched_submission_count += 1

    enrichment_artifact = CommentEnrichmentArtifact(
        run_dir=str(run_dir),
        generated_at=datetime.now(UTC),
        requested_submission_count=len(candidates),
        fetched_submission_count=fetched_submission_count,
        comment_count=total_comment_count,
        morechildren_request_count=morechildren_request_count,
        raw_comment_artifacts=raw_artifacts,
        normalized_comment_artifacts=normalized_artifacts,
    )
    store.write_comment_enrichment_json(enrichment_artifact.model_dump(mode="json"))

    return CommentEnrichmentResult(
        run_dir=run_dir,
        requested_submission_count=len(candidates),
        fetched_submission_count=fetched_submission_count,
        comment_count=total_comment_count,
        morechildren_request_count=morechildren_request_count,
        raw_comment_artifacts=raw_artifacts,
        normalized_comment_artifacts=normalized_artifacts,
    )


async def run_search_command(
    config: RuntimeConfig,
    subreddits: list[str],
    queries: list[str],
    sort: str = DEFAULT_SORT,
    time_filter: str = DEFAULT_TIME_FILTER,
    limit: int = DEFAULT_LIMIT,
    pages_per_query: int = DEFAULT_PAGES_PER_QUERY,
    expand_queries: bool = DEFAULT_EXPAND_QUERIES,
    additional_sorts: list[str] | None = None,
    additional_time_filters: list[str] | None = None,
    min_score: int = 0,
    min_comments: int = 0,
    filter_nsfw: bool = False,
    allowed_subreddits: list[str] | None = None,
    denied_subreddits: list[str] | None = None,
    output_dir: Path | None = None,
    client: RedditClient | None = None,
) -> SearchRunResult:
    search_sorts, search_time_filters = resolve_search_plan(
        sort=sort,
        time_filter=time_filter,
        additional_sorts=additional_sorts,
        additional_time_filters=additional_time_filters,
    )
    specs = build_search_specs(
        subreddits,
        queries,
        sort=sort,
        time_filter=time_filter,
        limit=limit,
        pages_per_query=pages_per_query,
        expand_queries=expand_queries,
        additional_sorts=additional_sorts,
        additional_time_filters=additional_time_filters,
    )
    quality_filters = build_retrieval_quality_filters(
        min_score=min_score,
        min_comments=min_comments,
        filter_nsfw=filter_nsfw,
        allowed_subreddits=allowed_subreddits,
        denied_subreddits=denied_subreddits,
    )
    run_slug = build_search_run_slug(subreddits, queries)
    store = build_artifact_store(config.output_root, run_slug, output_dir)
    manifest = RunManifest(
        run_slug=run_slug,
        status="running",
        started_at=datetime.now(UTC),
        output_dir=str(store.run_dir),
        subreddits=subreddits,
        queries=queries,
        query_variants=_dedupe_preserve_order([spec.query for spec in specs]),
        search_sorts=search_sorts,
        search_time_filters=search_time_filters,
        min_score=quality_filters.min_score,
        min_comments=quality_filters.min_comments,
        filter_nsfw=quality_filters.filter_nsfw,
        allowed_subreddits=list(quality_filters.allowed_subreddits),
        denied_subreddits=list(quality_filters.denied_subreddits),
        sort=sort,
        time_filter=time_filter,
        limit=limit,
        pages_per_query=pages_per_query,
        request_timeout_seconds=config.request_timeout_seconds,
        max_retries=config.max_retries,
        max_concurrent_requests=config.max_concurrent_requests,
        warnings=[
            "Comment retrieval is a separate explicit stage and is not run automatically by search.",
            "Ranking is a separate explicit stage and is not run automatically by search.",
            "Clustering is a separate explicit stage and is not run automatically by search.",
            "Summary and memo synthesis are separate explicit stages and are not run automatically by search.",
            "Stored post text should be treated as inspectable run data, not permanent archival data.",
        ],
    )
    store.write_manifest(manifest)

    owns_client = client is None
    reddit_client = client or RedditClient(config)
    try:
        if owns_client:
            await reddit_client.__aenter__()
        result = await _execute_specs(
            reddit_client,
            specs,
            store,
            manifest,
            config,
            quality_filters=quality_filters,
        )
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
    *,
    quality_filters: RetrievalQualityFilters,
) -> SearchRunResult:
    semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def fetch(spec: SearchRequestSpec):
        async with semaphore:
            payload, log_entry = await client.search_subreddit(spec)
            return spec, payload, log_entry

    async def fetch_chain(base_spec: SearchRequestSpec):
        responses: list[tuple[SearchRequestSpec, dict[str, Any], Any]] = []
        after: str | None = None
        for _ in range(manifest.pages_per_query):
            page_spec = base_spec.model_copy(update={"after": after})
            spec, payload, log_entry = await fetch(page_spec)
            responses.append((spec, payload, log_entry))
            after = ((payload.get("data") or {}).get("after")) or None
            if not after:
                break
        return responses

    response_groups = await asyncio.gather(*(fetch_chain(spec) for spec in specs))
    responses = [item for group in response_groups for item in group]
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
            reason = apply_candidate_quality_filters(candidate, quality_filters)
            if reason:
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
        query_variant_count=len(manifest.query_variants),
        search_spec_count=len(specs),
        sort_count=len(manifest.search_sorts),
        time_filter_count=len(manifest.search_time_filters),
        pages_per_query=manifest.pages_per_query,
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
        over_18=submission.over_18,
        source_queries=[spec.seed_query or spec.query],
        source_subreddits=[spec.subreddit],
        source_sorts=[spec.sort],
        source_time_filters=[spec.time_filter],
        retrieval_requests=[spec.request_key],
    ), ""


def expand_query_variants(query: str) -> list[str]:
    normalized = _normalize_query_text(query)
    if not normalized:
        return []

    variants = [normalized]
    if " " in normalized and '"' not in normalized:
        variants.append(f'"{normalized}"')

    keyword_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", normalized.lower())
        if token not in QUERY_EXPANSION_STOPWORDS
    ]
    if len(keyword_tokens) >= 2:
        variants.append(" ".join(keyword_tokens))

    return _dedupe_preserve_order(variants)


def normalize_comments_payload(payload: Any) -> tuple[list[Comment], list[str]]:
    if not isinstance(payload, list) or len(payload) < 2:
        return [], []
    listing = payload[1]
    children = ((listing or {}).get("data") or {}).get("children") or []
    comments: list[Comment] = []
    morechildren_ids: list[str] = []
    for child in children:
        extracted_comments, extracted_morechildren = _extract_comments_from_child(child)
        comments.extend(extracted_comments)
        morechildren_ids.extend(extracted_morechildren)
    return comments, _dedupe_preserve_order(morechildren_ids)


def normalize_morechildren_payload(payload: Any) -> tuple[list[Comment], list[str]]:
    things = (((payload or {}).get("json") or {}).get("data") or {}).get("things") or []
    comments: list[Comment] = []
    morechildren_ids: list[str] = []
    for thing in things:
        extracted_comments, extracted_morechildren = _extract_comments_from_child(thing)
        comments.extend(extracted_comments)
        morechildren_ids.extend(extracted_morechildren)
    return comments, _dedupe_preserve_order(morechildren_ids)


def _merge_candidate(existing: CandidatePost, candidate: CandidatePost) -> None:
    for value in candidate.source_queries:
        if value not in existing.source_queries:
            existing.source_queries.append(value)
    for value in candidate.source_subreddits:
        if value not in existing.source_subreddits:
            existing.source_subreddits.append(value)
    for value in candidate.source_sorts:
        if value not in existing.source_sorts:
            existing.source_sorts.append(value)
    for value in candidate.source_time_filters:
        if value not in existing.source_time_filters:
            existing.source_time_filters.append(value)
    for value in candidate.retrieval_requests:
        if value not in existing.retrieval_requests:
            existing.retrieval_requests.append(value)


def _extract_comments_from_child(child: Any) -> tuple[list[Comment], list[str]]:
    if not isinstance(child, dict):
        return [], []
    kind = child.get("kind")
    if kind == "more":
        data = child.get("data") or {}
        children = data.get("children") or []
        if not isinstance(children, list):
            return [], []
        return [], [str(child_id) for child_id in children if str(child_id).strip()]
    if kind != "t1":
        return [], []

    data = child.get("data") or {}
    body = str(data.get("body") or "").strip()
    if not body or body.lower() in {"[deleted]", "[removed]"}:
        return [], []

    comment = Comment(
        id=str(data.get("id") or ""),
        body=body,
        author=str(data.get("author")) if data.get("author") is not None else None,
        score=_to_int(data.get("score")),
        created_utc=_to_float(data.get("created_utc")),
        permalink=_normalize_permalink(data.get("permalink")),
        parent_id=str(data.get("parent_id")) if data.get("parent_id") is not None else None,
        link_id=str(data.get("link_id")) if data.get("link_id") is not None else None,
        depth=_to_int(data.get("depth")),
    )
    comments = [comment]
    morechildren_ids: list[str] = []

    replies = data.get("replies")
    if isinstance(replies, dict):
        reply_children = ((replies.get("data") or {}).get("children")) or []
        for reply in reply_children:
            nested_comments, nested_morechildren = _extract_comments_from_child(reply)
            comments.extend(nested_comments)
            morechildren_ids.extend(nested_morechildren)
    return comments, morechildren_ids


def _load_candidate_posts(run_dir: Path) -> list[CandidatePost]:
    candidate_posts_path = run_dir / "candidate_posts.json"
    if not candidate_posts_path.exists():
        raise FileNotFoundError(f"candidate posts not found: {candidate_posts_path}")
    payload = json.loads(candidate_posts_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("candidate_posts.json must contain a list")
    return [CandidatePost.model_validate(item) for item in payload]


def _normalize_query_text(query: str) -> str:
    return " ".join(query.split()).strip()


def _normalize_subreddit_filters(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        normalized_value = str(value).strip().lower()
        if not normalized_value:
            continue
        if normalized_value not in normalized:
            normalized.append(normalized_value)
    return normalized


def _resolve_search_values(
    *,
    primary: str,
    additional: list[str],
    allowed: set[str],
    label: str,
) -> list[str]:
    requested = [primary, *additional]
    normalized: list[str] = []
    for value in requested:
        normalized_value = str(value).strip().lower()
        if normalized_value not in allowed:
            raise ValueError(f"{label} must be one of: {', '.join(sorted(allowed))}")
        if normalized_value not in normalized:
            normalized.append(normalized_value)
    return normalized


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


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
