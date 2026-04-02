from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RateLimitSnapshot(BaseModel):
    used: float | None = None
    remaining: float | None = None
    reset_seconds: float | None = None


class SearchRequestSpec(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    subreddit: str
    query: str
    sort: str = "relevance"
    time_filter: str = "all"
    limit: int = Field(default=25, ge=1, le=100)
    after: str | None = None

    @property
    def request_key(self) -> str:
        return "|".join(
            [
                self.subreddit.lower(),
                self.query.lower(),
                self.sort.lower(),
                self.time_filter.lower(),
                self.after or "",
            ]
        )


class Submission(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    subreddit: str
    url: str
    permalink: str | None = None
    score: int | None = None
    num_comments: int | None = None
    created_utc: float | None = None
    selftext: str = ""
    author: str | None = None
    is_self: bool | None = None
    over_18: bool | None = None
    removed_by_category: str | None = None


class Comment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    body: str
    author: str | None = None
    score: int | None = None
    created_utc: float | None = None
    permalink: str | None = None
    parent_id: str | None = None
    link_id: str | None = None


class CandidatePost(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    subreddit: str
    url: str
    permalink: str | None = None
    score: int | None = None
    num_comments: int | None = None
    created_utc: float | None = None
    selftext: str = ""
    author: str | None = None
    source_queries: list[str] = Field(default_factory=list)
    source_subreddits: list[str] = Field(default_factory=list)
    retrieval_requests: list[str] = Field(default_factory=list)


class RequestLogEntry(BaseModel):
    requested_at: datetime
    request_name: str
    method: str
    url: str
    params: dict[str, str]
    status_code: int | None = None
    duration_ms: float | None = None
    attempt: int = 1
    rate_limit: RateLimitSnapshot | None = None
    raw_artifact_path: str | None = None
    error: str | None = None


class RunManifest(BaseModel):
    run_slug: str
    status: Literal["running", "completed", "failed"]
    started_at: datetime
    completed_at: datetime | None = None
    output_dir: str
    subreddits: list[str]
    queries: list[str]
    sort: str
    time_filter: str
    limit: int
    request_timeout_seconds: float
    max_retries: int
    max_concurrent_requests: int
    request_count: int = 0
    raw_search_artifacts: list[str] = Field(default_factory=list)
    candidate_count: int = 0
    filtered_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
