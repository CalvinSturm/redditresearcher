from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re

from .artifact_store import ArtifactStore
from .models import (
    CandidatePost,
    CandidateScreeningArtifact,
    CandidateScreeningDecision,
    Comment,
    CommentScreeningBreakdown,
    PostRankingArtifact,
    PostScoreBreakdown,
    RankedCandidatePost,
    SubmissionCommentsArtifact,
)


@dataclass(frozen=True)
class PostRankingResult:
    run_dir: Path
    candidate_count: int
    screened_candidate_count: int
    rejected_candidate_count: int
    selected_count: int
    rejection_counts: dict[str, int]
    ranked_posts: list[RankedCandidatePost]


LOW_SIGNAL_COMMENT_BODIES = {"same", "same here", "this", "me too", "+1", "following"}
COMPLAINT_SIGNAL_PATTERN = re.compile(
    r"\b("
    r"because|still|manual|manually|problem|issue|pain|painful|frustrat\w*|"
    r"annoy\w*|stuck|waste\w*|broken|breaks|hard|heavy|spreadsheet|crm|"
    r"follow-?up|miss\w*|copy|paste|workaround|hack|reminder|workflow"
    r")\b",
    re.IGNORECASE,
)
FIRST_PERSON_PATTERN = re.compile(r"\b(i|we|my|our|me|us)\b", re.IGNORECASE)


def rank_run_candidates(
    run_dir: Path,
    *,
    max_selected_posts: int = 10,
    min_non_trivial_comments: int = 0,
    min_complaint_signal_comments: int = 0,
    now: datetime | None = None,
) -> PostRankingResult:
    if max_selected_posts <= 0:
        raise ValueError("max_selected_posts must be greater than 0")
    if min_non_trivial_comments < 0:
        raise ValueError("min_non_trivial_comments must be 0 or greater")
    if min_complaint_signal_comments < 0:
        raise ValueError("min_complaint_signal_comments must be 0 or greater")

    store = ArtifactStore(run_dir)
    candidates = load_candidate_posts(run_dir)
    comments_by_submission = load_submission_comments(run_dir)
    screening_artifact = screen_candidates_for_ranking(
        run_dir=str(run_dir),
        candidates=candidates,
        comments_by_submission=comments_by_submission,
        min_non_trivial_comments=min_non_trivial_comments,
        min_complaint_signal_comments=min_complaint_signal_comments,
        generated_at=now or datetime.now(UTC),
    )
    store.write_candidate_screening_json(screening_artifact.model_dump(mode="json"))
    screened_candidates = [
        decision.candidate
        for decision in screening_artifact.decisions
        if decision.kept
    ]

    ranked_posts = rank_candidates(
        screened_candidates,
        comments_by_submission=comments_by_submission,
        now=now,
    )
    selected_posts = ranked_posts[:max_selected_posts]

    ranking_artifact = PostRankingArtifact(
        run_dir=str(run_dir),
        generated_at=now or datetime.now(UTC),
        candidate_count=len(candidates),
        screened_candidate_count=screening_artifact.kept_count,
        rejected_candidate_count=screening_artifact.rejected_count,
        selected_count=len(selected_posts),
        rejection_counts=screening_artifact.rejection_counts,
        ranked_posts=ranked_posts,
    )
    store.write_post_ranking_json(ranking_artifact.model_dump(mode="json"))
    store.write_selected_posts_json(
        [ranked_post.model_dump(mode="json") for ranked_post in selected_posts]
    )

    return PostRankingResult(
        run_dir=run_dir,
        candidate_count=len(candidates),
        screened_candidate_count=screening_artifact.kept_count,
        rejected_candidate_count=screening_artifact.rejected_count,
        selected_count=len(selected_posts),
        rejection_counts=screening_artifact.rejection_counts,
        ranked_posts=ranked_posts,
    )


def screen_candidates_for_ranking(
    *,
    run_dir: str,
    candidates: list[CandidatePost],
    comments_by_submission: dict[str, list[Comment]],
    min_non_trivial_comments: int = 0,
    min_complaint_signal_comments: int = 0,
    generated_at: datetime | None = None,
) -> CandidateScreeningArtifact:
    rejection_counts: Counter[str] = Counter()
    decisions: list[CandidateScreeningDecision] = []

    for candidate in candidates:
        breakdown = analyze_comment_screening(
            comments_by_submission.get(candidate.id, [])
        )
        rejection_reason: str | None = None
        if breakdown.non_trivial_comment_count < min_non_trivial_comments:
            rejection_reason = "low_non_trivial_comments"
        elif breakdown.complaint_signal_comment_count < min_complaint_signal_comments:
            rejection_reason = "low_complaint_signal_comments"

        kept = rejection_reason is None
        if rejection_reason:
            rejection_counts[rejection_reason] += 1
        decisions.append(
            CandidateScreeningDecision(
                candidate=candidate,
                kept=kept,
                rejection_reason=rejection_reason,
                breakdown=breakdown,
            )
        )

    return CandidateScreeningArtifact(
        run_dir=run_dir,
        generated_at=generated_at or datetime.now(UTC),
        candidate_count=len(candidates),
        kept_count=sum(1 for decision in decisions if decision.kept),
        rejected_count=sum(1 for decision in decisions if not decision.kept),
        min_non_trivial_comments=min_non_trivial_comments,
        min_complaint_signal_comments=min_complaint_signal_comments,
        rejection_counts=dict(sorted(rejection_counts.items())),
        decisions=decisions,
    )


def analyze_comment_screening(comments: list[Comment]) -> CommentScreeningBreakdown:
    non_trivial_comment_count = 0
    complaint_signal_comment_count = 0
    for comment in comments:
        if is_non_trivial_comment(comment):
            non_trivial_comment_count += 1
            if has_complaint_signal(comment):
                complaint_signal_comment_count += 1
    return CommentScreeningBreakdown(
        saved_comment_count=len(comments),
        non_trivial_comment_count=non_trivial_comment_count,
        complaint_signal_comment_count=complaint_signal_comment_count,
    )


def is_non_trivial_comment(comment: Comment) -> bool:
    body = comment.body.strip()
    if not body:
        return False
    normalized = body.lower()
    if normalized in LOW_SIGNAL_COMMENT_BODIES:
        return False
    word_count = len(body.split())
    if word_count < 6:
        return False
    if word_count >= 12:
        return True
    return bool(re.search(r"[.,;:]", body) or FIRST_PERSON_PATTERN.search(body))


def has_complaint_signal(comment: Comment) -> bool:
    body = comment.body.strip()
    if not body or not is_non_trivial_comment(comment):
        return False
    return bool(
        COMPLAINT_SIGNAL_PATTERN.search(body)
        and FIRST_PERSON_PATTERN.search(body)
    )


def rank_candidates(
    candidates: list[CandidatePost],
    *,
    comments_by_submission: dict[str, list[Comment]] | None = None,
    now: datetime | None = None,
) -> list[RankedCandidatePost]:
    comments_by_submission = comments_by_submission or {}
    ranked: list[RankedCandidatePost] = []
    current_time = now or datetime.now(UTC)

    for candidate in candidates:
        saved_comments = comments_by_submission.get(candidate.id, [])
        breakdown = score_candidate_post(
            candidate,
            saved_comments=saved_comments,
            now=current_time,
        )
        ranked.append(
            RankedCandidatePost(
                candidate=candidate,
                saved_comment_count=len(saved_comments),
                breakdown=breakdown,
                rank=0,
            )
        )

    ranked.sort(
        key=lambda item: (
            item.breakdown.total_score,
            item.saved_comment_count,
            item.candidate.num_comments or 0,
            item.candidate.score or 0,
            item.candidate.created_utc or 0,
        ),
        reverse=True,
    )

    for index, item in enumerate(ranked, start=1):
        item.rank = index  # type: ignore[misc]
    return ranked


def score_candidate_post(
    candidate: CandidatePost,
    *,
    saved_comments: list[Comment],
    now: datetime,
) -> PostScoreBreakdown:
    text_blob = " ".join([candidate.title, candidate.selftext]).lower()
    text_tokens = _tokenize(text_blob)

    query_relevance_score = 0.0
    for query in candidate.source_queries:
        query_tokens = _tokenize(query)
        if not query_tokens:
            continue
        overlap = len(text_tokens.intersection(query_tokens))
        ratio = overlap / len(query_tokens)
        query_relevance_score = max(query_relevance_score, min(ratio * 3.0, 3.0))

    score_part = min(max(candidate.score or 0, 0) / 25.0, 3.0)
    comments_part = min(max(candidate.num_comments or 0, 0) / 15.0, 3.0)
    engagement_score = score_part + comments_part

    saved_comment_count = len(saved_comments)
    avg_comment_score = (
        sum(max(comment.score or 0, 0) for comment in saved_comments) / saved_comment_count
        if saved_comment_count
        else 0.0
    )
    max_comment_depth = max((comment.depth or 0) for comment in saved_comments) if saved_comments else 0
    comment_richness_score = min(saved_comment_count / 4.0, 2.5) + min(avg_comment_score / 8.0, 1.5) + min(max_comment_depth * 0.3, 1.0)

    title_words = len(candidate.title.split())
    body_words = len(candidate.selftext.split())
    text_richness_score = min(title_words / 8.0, 1.5) + min(body_words / 40.0, 2.0)

    recency_score = 0.0
    if candidate.created_utc is not None:
        created_at = datetime.fromtimestamp(candidate.created_utc, tz=UTC)
        age_days = max((now - created_at).total_seconds() / 86400.0, 0.0)
        recency_score = max(0.2, 1.5 / (1.0 + age_days / 30.0))

    penalty_score = 0.0
    if not candidate.selftext.strip() and (candidate.num_comments or 0) < 5:
        penalty_score -= 0.75
    if len(candidate.title.split()) < 4:
        penalty_score -= 0.4
    if (candidate.score or 0) <= 1 and (candidate.num_comments or 0) <= 1:
        penalty_score -= 0.75

    total_score = (
        query_relevance_score
        + engagement_score
        + comment_richness_score
        + text_richness_score
        + recency_score
        + penalty_score
    )

    return PostScoreBreakdown(
        query_relevance_score=round(query_relevance_score, 3),
        engagement_score=round(engagement_score, 3),
        comment_richness_score=round(comment_richness_score, 3),
        text_richness_score=round(text_richness_score, 3),
        recency_score=round(recency_score, 3),
        penalty_score=round(penalty_score, 3),
        total_score=round(total_score, 3),
    )


def load_candidate_posts(run_dir: Path) -> list[CandidatePost]:
    candidate_posts_path = run_dir / "candidate_posts.json"
    if not candidate_posts_path.exists():
        raise FileNotFoundError(f"candidate posts not found: {candidate_posts_path}")

    payload = json.loads(candidate_posts_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("candidate_posts.json must contain a list")
    return [CandidatePost.model_validate(item) for item in payload]


def load_submission_comments(run_dir: Path) -> dict[str, list[Comment]]:
    comments_dir = run_dir / "comments"
    if not comments_dir.exists():
        return {}

    comments_by_submission: dict[str, list[Comment]] = {}
    for path in sorted(comments_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        artifact = SubmissionCommentsArtifact.model_validate(payload)
        comments_by_submission[artifact.submission_id] = artifact.comments
    return comments_by_submission


def load_selected_posts(run_dir: Path) -> list[CandidatePost]:
    ranked_posts = load_selected_ranked_posts(run_dir)
    return [item.candidate for item in ranked_posts]


def load_selected_ranked_posts(run_dir: Path) -> list[RankedCandidatePost]:
    selected_posts_path = run_dir / "selected_posts.json"
    if not selected_posts_path.exists():
        return []
    payload = json.loads(selected_posts_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("selected_posts.json must contain a list")
    return [RankedCandidatePost.model_validate(item) for item in payload]


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"\b[a-z0-9]{3,}\b", text.lower())}
