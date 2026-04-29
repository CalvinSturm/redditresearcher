from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


def _coerce_created_utc(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


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
    seed_query: str | None = None
    after: str | None = None

    @property
    def request_key(self) -> str:
        query_key = self.query.lower()
        seed_query = (self.seed_query or "").lower()
        if seed_query and seed_query != query_key:
            query_key = f"{seed_query}=>{query_key}"
        return "|".join(
            [
                self.subreddit.lower(),
                query_key,
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

    id: str = Field(validation_alias=AliasChoices("id", "comment_id"))
    body: str
    author: str | None = None
    score: int | None = None
    created_utc: float | None = Field(
        default=None,
        validation_alias=AliasChoices("created_utc", "created"),
    )
    permalink: str | None = None
    parent_id: str | None = None
    link_id: str | None = None
    depth: int | None = None

    @field_validator("created_utc", mode="before")
    @classmethod
    def parse_created_utc(cls, value: object) -> float | None:
        return _coerce_created_utc(value)


class ManualImportPost(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(validation_alias=AliasChoices("id", "post_id"))
    title: str
    subreddit: str
    url: str = Field(validation_alias=AliasChoices("url", "permalink"))
    permalink: str | None = None
    score: int | None = None
    num_comments: int | None = Field(
        default=None,
        validation_alias=AliasChoices("num_comments", "comments", "comments_full_count"),
    )
    created_utc: float | None = Field(
        default=None,
        validation_alias=AliasChoices("created_utc", "created"),
    )
    selftext: str = Field(
        default="",
        validation_alias=AliasChoices("selftext", "body_full", "body"),
    )
    author: str | None = None
    over_18: bool | None = None
    source_queries: list[str] = Field(default_factory=list)
    source_subreddits: list[str] = Field(default_factory=list)
    source_sorts: list[str] = Field(default_factory=list)
    source_time_filters: list[str] = Field(default_factory=list)
    retrieval_requests: list[str] = Field(default_factory=list)
    comments: list[Comment] = Field(
        default_factory=list,
        validation_alias=AliasChoices("comments", "comments_full"),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_userscript_payload(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "id" not in normalized and normalized.get("post_id") is not None:
            normalized["id"] = normalized.get("post_id")
        if "url" not in normalized and normalized.get("permalink") is not None:
            normalized["url"] = normalized.get("permalink")
        if "num_comments" not in normalized:
            comments_value = normalized.get("comments")
            if isinstance(comments_value, int):
                normalized["num_comments"] = comments_value
            elif normalized.get("comments_full_count") is not None:
                normalized["num_comments"] = normalized.get("comments_full_count")
        if "selftext" not in normalized:
            if normalized.get("body_full") is not None:
                normalized["selftext"] = normalized.get("body_full")
            elif normalized.get("body") is not None:
                normalized["selftext"] = normalized.get("body")
        if not isinstance(normalized.get("comments"), list) and isinstance(
            normalized.get("comments_full"), list
        ):
            normalized["comments"] = normalized.get("comments_full")
        return normalized

    @field_validator("created_utc", mode="before")
    @classmethod
    def parse_created_utc(cls, value: object) -> float | None:
        return _coerce_created_utc(value)


class ManualImportBundle(BaseModel):
    model_config = ConfigDict(extra="ignore")

    posts: list[ManualImportPost] = Field(default_factory=list)


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
    over_18: bool | None = None
    source_queries: list[str] = Field(default_factory=list)
    source_subreddits: list[str] = Field(default_factory=list)
    source_sorts: list[str] = Field(default_factory=list)
    source_time_filters: list[str] = Field(default_factory=list)
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
    retrieval_mode: Literal["api", "manual"] = "api"
    manual_input_path: str | None = None
    topic: str | None = None
    target_audience: str | None = None
    category: Literal["software", "business", "ergonomics"] | None = None
    time_horizon: str | None = None
    subreddits: list[str]
    queries: list[str]
    query_variants: list[str] = Field(default_factory=list)
    search_sorts: list[str] = Field(default_factory=list)
    search_time_filters: list[str] = Field(default_factory=list)
    min_score: int = 0
    min_comments: int = 0
    filter_nsfw: bool = False
    allowed_subreddits: list[str] = Field(default_factory=list)
    denied_subreddits: list[str] = Field(default_factory=list)
    sort: str
    time_filter: str
    limit: int
    pages_per_query: int = 1
    request_timeout_seconds: float
    max_retries: int
    max_concurrent_requests: int
    request_count: int = 0
    raw_search_artifacts: list[str] = Field(default_factory=list)
    raw_manual_artifacts: list[str] = Field(default_factory=list)
    candidate_count: int = 0
    filtered_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    brief_path: str | None = None


class LLMGenerationResult(BaseModel):
    provider: str
    model: str
    prompt: str
    output_text: str
    raw_response: dict


class AssetGenerationProvenance(BaseModel):
    provider: str | None = None
    model: str | None = None
    prompt_artifact_path: str | None = None
    raw_response_artifact_path: str | None = None


class RegisteredAsset(BaseModel):
    artifact_path: str
    artifact_type: str
    created_at: datetime
    generation: AssetGenerationProvenance | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class EvidenceSummaryArtifact(BaseModel):
    run_dir: str
    generated_at: datetime
    provider: str
    model: str
    candidate_count: int
    comment_count: int = 0
    selected_comment_count: int = 0
    max_posts_used: int
    prompt_artifact_path: str
    raw_response_artifact_path: str
    summary_markdown_artifact_path: str
    summary_text: str


class FinalMemoArtifact(BaseModel):
    run_dir: str
    generated_at: datetime
    provider: str
    model: str
    strongest_cluster_id: str
    strongest_cluster_size: int
    included_post_count: int
    topic: str | None = None
    target_audience: str | None = None
    category: Literal["software", "business", "ergonomics"] | None = None
    time_horizon: str | None = None
    source_thread_urls: list[str] = Field(default_factory=list)
    passed_validation: bool = True
    validation_issues: list[str] = Field(default_factory=list)
    prompt_artifact_path: str
    raw_response_artifact_path: str
    final_memo_artifact_path: str
    memo_text: str


class ReplyDraft(BaseModel):
    post_id: str
    title: str
    subreddit: str
    url: str
    rank: int | None = None
    comment_opportunity_score: float | None = None
    comment_opportunity_bucket: Literal["high_value", "watchlist", "ignore"] | None = None
    reply_text: str
    relevance_score: float | None = None
    conversation_value_score: float | None = None
    voice_match_score: float | None = None
    reddit_friendliness_score: float | None = None
    average_score: float | None = None
    passed_threshold: bool | None = None
    evaluation_feedback: str | None = None


class ReplyDraftArtifact(BaseModel):
    run_dir: str
    generated_at: datetime
    provider: str
    model: str
    selected_post_count: int
    voice: str
    improvement_rounds: int = 0
    score_threshold: float = 4.0
    minimum_dimension_score: float = 3.0
    passed_threshold: bool = False
    passed_validation: bool = True
    output_validation_issues: list[str] = Field(default_factory=list)
    prompt_artifact_path: str
    raw_response_artifact_path: str
    reply_markdown_artifact_path: str
    drafts: list[ReplyDraft] = Field(default_factory=list)


class CommentOpportunity(BaseModel):
    post_id: str
    title: str
    subreddit: str
    url: str
    rank: int | None = None
    total_score: float
    bucket: Literal["high_value", "watchlist", "ignore"]
    breakdown: ThreadCommentOpportunityBreakdown


class CommentOpportunityArtifact(BaseModel):
    run_dir: str
    generated_at: datetime
    scored_post_count: int
    opportunities: list[CommentOpportunity] = Field(default_factory=list)


class ReviewCheckpointArtifact(BaseModel):
    run_dir: str
    review_type: Literal["memo", "reply"]
    status: Literal["pending", "approved", "rejected"] = "pending"
    created_at: datetime
    updated_at: datetime
    artifact_path: str | None = None
    notes: str | None = None
    context: dict[str, object] = Field(default_factory=dict)


class ResearchBrief(BaseModel):
    path: str | None = None
    topic: str | None = None
    target_audience: str | None = None
    preferred_subreddits: list[str] = Field(default_factory=list)
    avoid_subreddits: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    category: Literal["software", "business", "ergonomics"] | None = None
    time_horizon: str | None = None
    success_criteria: list[str] = Field(default_factory=list)
    notes: str | None = None


class RunStageReport(BaseModel):
    stage: str
    status: Literal["completed", "failed", "skipped", "stopped"]
    duration_ms: float | None = None
    details: dict[str, object] = Field(default_factory=dict)
    artifact_fingerprints: dict[str, str] = Field(default_factory=dict)


class RunReportArtifact(BaseModel):
    run_slug: str
    run_dir: str
    status: Literal["completed", "failed", "stopped"]
    started_at: datetime
    completed_at: datetime
    subreddits: list[str]
    queries: list[str]
    sort: str
    time_filter: str
    limit: int
    provider: str | None = None
    model: str | None = None
    stop_reason: str | None = None
    error: str | None = None
    stage_reports: list[RunStageReport] = Field(default_factory=list)
    output_paths: dict[str, str] = Field(default_factory=dict)


class SubmissionCommentsArtifact(BaseModel):
    submission_id: str
    subreddit: str
    permalink: str | None = None
    title: str
    fetched_comment_count: int
    comments: list[Comment] = Field(default_factory=list)


class CommentEnrichmentArtifact(BaseModel):
    run_dir: str
    generated_at: datetime
    requested_submission_count: int
    fetched_submission_count: int
    comment_count: int
    morechildren_request_count: int = 0
    raw_comment_artifacts: list[str] = Field(default_factory=list)
    normalized_comment_artifacts: list[str] = Field(default_factory=list)


class CommentSelectionBreakdown(BaseModel):
    length_score: float = 0.0
    engagement_score: float = 0.0
    depth_score: float = 0.0
    detail_score: float = 0.0
    penalty_score: float = 0.0
    total_score: float = 0.0


class SelectedCommentEvidence(BaseModel):
    submission_id: str
    comment_id: str
    body: str
    score: int | None = None
    depth: int | None = None
    permalink: str | None = None
    breakdown: CommentSelectionBreakdown


class CommentSelectionArtifact(BaseModel):
    run_dir: str
    generated_at: datetime
    total_saved_comments: int
    selected_comment_count: int
    max_comments_per_post: int
    selections: list[SelectedCommentEvidence] = Field(default_factory=list)


class CommentScreeningBreakdown(BaseModel):
    saved_comment_count: int = 0
    non_trivial_comment_count: int = 0
    complaint_signal_comment_count: int = 0


class ThreadCommentOpportunityBreakdown(BaseModel):
    strong_pain_score: float = 0.0
    first_person_urgency_score: float = 0.0
    discussion_confirmation_score: float = 0.0
    freshness_score: float = 0.0
    icp_fit_score: float = 0.0
    source_quality_score: float = 0.0
    engagement_safety_score: float = 0.0
    total_score: float = 0.0
    recommendation: Literal["high_value", "watchlist", "ignore"] = "ignore"
    safe_to_engage: bool = False


class CandidateScreeningDecision(BaseModel):
    candidate: CandidatePost
    kept: bool
    rejection_reason: str | None = None
    breakdown: CommentScreeningBreakdown


class CandidateScreeningArtifact(BaseModel):
    run_dir: str
    generated_at: datetime
    candidate_count: int
    kept_count: int
    rejected_count: int
    min_non_trivial_comments: int = 0
    min_complaint_signal_comments: int = 0
    rejection_counts: dict[str, int] = Field(default_factory=dict)
    decisions: list[CandidateScreeningDecision] = Field(default_factory=list)


class PostScoreBreakdown(BaseModel):
    query_relevance_score: float = 0.0
    engagement_score: float = 0.0
    comment_richness_score: float = 0.0
    text_richness_score: float = 0.0
    recency_score: float = 0.0
    source_quality_penalty: float = 0.0
    penalty_score: float = 0.0
    total_score: float = 0.0


class RankedCandidatePost(BaseModel):
    candidate: CandidatePost
    saved_comment_count: int = 0
    breakdown: PostScoreBreakdown
    rank: int


class PostRankingArtifact(BaseModel):
    run_dir: str
    generated_at: datetime
    candidate_count: int
    screened_candidate_count: int = 0
    rejected_candidate_count: int = 0
    selected_count: int
    rejection_counts: dict[str, int] = Field(default_factory=dict)
    ranked_posts: list[RankedCandidatePost] = Field(default_factory=list)


class ThemeCluster(BaseModel):
    cluster_id: str
    label: str
    post_ids: list[str] = Field(default_factory=list)
    size: int
    average_post_score: float = 0.0
    total_comment_count: int = 0
    top_terms: list[str] = Field(default_factory=list)
    member_ranks: list[int] = Field(default_factory=list)
    cohesion_score: float = 0.0
    source_thread_urls: list[str] = Field(default_factory=list)
    source_subreddits: list[str] = Field(default_factory=list)
    cross_subreddit_count: int = 0
    minimum_theme_size_met: bool | None = None
    opportunity_score: float = 0.0
    opportunity_recommendation: Literal["strong", "moderate", "weak"] = "weak"

    @model_validator(mode="after")
    def _default_minimum_theme_size_met(self) -> "ThemeCluster":
        if self.minimum_theme_size_met is None:
            self.minimum_theme_size_met = self.size >= 5
        return self


class ClusterEvidenceMember(BaseModel):
    post_id: str
    kept: bool = False
    non_trivial_comment_count: int = 0
    complaint_signal_comment_count: int = 0


class ClusterEvidenceValidationArtifact(BaseModel):
    run_dir: str
    generated_at: datetime
    strongest_cluster_id: str | None = None
    strongest_cluster_post_count: int = 0
    screening_available: bool = False
    screened_cluster_post_count: int = 0
    complaint_signal_post_count: int = 0
    min_cluster_complaint_posts: int = 0
    passes: bool = True
    failure_reason: str | None = None
    members: list[ClusterEvidenceMember] = Field(default_factory=list)


class ThemeSummaryArtifact(BaseModel):
    run_dir: str
    generated_at: datetime
    source_post_count: int
    cluster_count: int
    strongest_cluster_id: str | None = None
    strongest_post_ids: list[str] = Field(default_factory=list)
    valid_cluster_count: int = 0
    invalid_cluster_count: int = 0
    clusters: list[ThemeCluster] = Field(default_factory=list)


class RunMemoryEntry(BaseModel):
    run_slug: str
    run_dir: str
    status: str | None = None
    topic: str | None = None
    target_audience: str | None = None
    category: str | None = None
    time_horizon: str | None = None
    strongest_cluster_id: str | None = None
    strongest_cluster_label: str | None = None
    strongest_cluster_size: int = 0
    strongest_cluster_opportunity_score: float = 0.0
    strongest_cluster_opportunity_recommendation: str | None = None
    source_thread_count: int = 0
    memo_validation_passed: bool | None = None
    reply_review_status: str | None = None
    memo_review_status: str | None = None
    completed_at: datetime | None = None


class RunMemoryArtifact(BaseModel):
    generated_at: datetime
    runs_root: str
    run_count: int
    entries: list[RunMemoryEntry] = Field(default_factory=list)
