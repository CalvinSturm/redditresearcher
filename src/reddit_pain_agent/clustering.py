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
    ClusterEvidenceMember,
    ClusterEvidenceValidationArtifact,
    RankedCandidatePost,
    ThemeCluster,
    ThemeSummaryArtifact,
)
from .ranking import load_selected_ranked_posts


STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "been", "before", "being",
    "but", "can", "could", "every", "feel", "from", "have", "into", "just",
    "like", "manual", "more", "need", "over", "same", "still", "than", "that",
    "the", "their", "them", "they", "this", "using", "very", "what", "when",
    "with", "work", "workflow", "your",
    "pain",
}


@dataclass(frozen=True)
class ThemeClusteringResult:
    run_dir: Path
    source_post_count: int
    cluster_count: int
    strongest_cluster_id: str | None
    strongest_post_ids: list[str]
    strongest_cluster_complaint_signal_post_count: int
    strongest_cluster_screened_post_count: int
    evidence_validation_passed: bool
    evidence_failure_reason: str | None
    clusters: list[ThemeCluster]


def cluster_run_posts(
    run_dir: Path,
    *,
    similarity_threshold: float = 0.22,
    min_shared_terms: int = 2,
    min_cluster_complaint_posts: int = 0,
    now: datetime | None = None,
) -> ThemeClusteringResult:
    if min_cluster_complaint_posts < 0:
        raise ValueError("min_cluster_complaint_posts must be 0 or greater")

    ranked_posts = load_cluster_source_posts(run_dir)
    clusters = cluster_ranked_posts(
        ranked_posts,
        similarity_threshold=similarity_threshold,
        min_shared_terms=min_shared_terms,
    )
    strongest_cluster = clusters[0] if clusters else None

    artifact = ThemeSummaryArtifact(
        run_dir=str(run_dir),
        generated_at=now or datetime.now(UTC),
        source_post_count=len(ranked_posts),
        cluster_count=len(clusters),
        strongest_cluster_id=strongest_cluster.cluster_id if strongest_cluster else None,
        strongest_post_ids=strongest_cluster.post_ids if strongest_cluster else [],
        clusters=clusters,
    )
    store = ArtifactStore(run_dir)
    store.write_theme_summary_json(artifact.model_dump(mode="json"))
    evidence_validation = validate_cluster_evidence(
        run_dir=run_dir,
        theme_summary=artifact,
        min_cluster_complaint_posts=min_cluster_complaint_posts,
        generated_at=now or datetime.now(UTC),
    )
    store.write_cluster_evidence_validation_json(
        evidence_validation.model_dump(mode="json")
    )

    return ThemeClusteringResult(
        run_dir=run_dir,
        source_post_count=len(ranked_posts),
        cluster_count=len(clusters),
        strongest_cluster_id=artifact.strongest_cluster_id,
        strongest_post_ids=artifact.strongest_post_ids,
        strongest_cluster_complaint_signal_post_count=evidence_validation.complaint_signal_post_count,
        strongest_cluster_screened_post_count=evidence_validation.screened_cluster_post_count,
        evidence_validation_passed=evidence_validation.passes,
        evidence_failure_reason=evidence_validation.failure_reason,
        clusters=clusters,
    )


def cluster_ranked_posts(
    ranked_posts: list[RankedCandidatePost],
    *,
    similarity_threshold: float = 0.22,
    min_shared_terms: int = 2,
) -> list[ThemeCluster]:
    if similarity_threshold < 0:
        raise ValueError("similarity_threshold must be non-negative")
    if min_shared_terms < 1:
        raise ValueError("min_shared_terms must be at least 1")

    working_clusters: list[dict] = []
    for ranked_post in ranked_posts:
        tokens = _post_tokens(ranked_post.candidate)
        best_match: dict | None = None
        best_similarity = 0.0
        for cluster in working_clusters:
            similarity = _jaccard(tokens, cluster["token_set"])
            shared_terms = len(tokens.intersection(cluster["token_set"]))
            if similarity >= similarity_threshold or shared_terms >= min_shared_terms:
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cluster

        if best_match is None:
            working_clusters.append(
                {
                    "members": [ranked_post],
                    "token_set": set(tokens),
                    "term_counter": Counter(tokens),
                }
            )
            continue

        best_match["members"].append(ranked_post)
        best_match["token_set"].update(tokens)
        best_match["term_counter"].update(tokens)

    built_clusters = [_build_theme_cluster(index + 1, cluster["members"], cluster["term_counter"]) for index, cluster in enumerate(working_clusters)]
    built_clusters.sort(
        key=lambda cluster: (
            cluster.size,
            cluster.average_post_score,
            cluster.total_comment_count,
        ),
        reverse=True,
    )
    return built_clusters


def load_cluster_source_posts(run_dir: Path) -> list[RankedCandidatePost]:
    ranked_posts = load_selected_ranked_posts(run_dir)
    if ranked_posts:
        return ranked_posts

    candidate_posts_path = run_dir / "candidate_posts.json"
    if not candidate_posts_path.exists():
        raise FileNotFoundError(f"candidate posts not found: {candidate_posts_path}")
    payload = json.loads(candidate_posts_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("candidate_posts.json must contain a list")
    candidates = [CandidatePost.model_validate(item) for item in payload]
    return [
        RankedCandidatePost(
            candidate=candidate,
            saved_comment_count=0,
            breakdown={"total_score": 0.0},
            rank=index,
        )
        for index, candidate in enumerate(candidates, start=1)
    ]


def load_strongest_cluster_posts(run_dir: Path) -> list[CandidatePost]:
    theme_summary_path = run_dir / "theme_summary.json"
    if not theme_summary_path.exists():
        return []

    payload = json.loads(theme_summary_path.read_text(encoding="utf-8"))
    artifact = ThemeSummaryArtifact.model_validate(payload)
    if not artifact.strongest_post_ids:
        return []

    ranked_posts = load_cluster_source_posts(run_dir)
    by_id = {item.candidate.id: item.candidate for item in ranked_posts}
    return [by_id[post_id] for post_id in artifact.strongest_post_ids if post_id in by_id]


def load_cluster_evidence_validation(
    run_dir: Path,
) -> ClusterEvidenceValidationArtifact | None:
    path = run_dir / "cluster_evidence_validation.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ClusterEvidenceValidationArtifact.model_validate(payload)


def validate_cluster_evidence(
    *,
    run_dir: Path,
    theme_summary: ThemeSummaryArtifact | None = None,
    min_cluster_complaint_posts: int = 0,
    generated_at: datetime | None = None,
) -> ClusterEvidenceValidationArtifact:
    theme_summary = theme_summary or _load_theme_summary(run_dir)
    strongest_post_ids = list(theme_summary.strongest_post_ids)
    screening_artifact = _load_candidate_screening(run_dir)

    screening_available = screening_artifact is not None
    screening_by_post_id = {
        item.candidate.id: item for item in screening_artifact.decisions
    } if screening_artifact else {}

    members: list[ClusterEvidenceMember] = []
    for post_id in strongest_post_ids:
        decision = screening_by_post_id.get(post_id)
        members.append(
            ClusterEvidenceMember(
                post_id=post_id,
                kept=decision.kept if decision is not None else False,
                non_trivial_comment_count=(
                    decision.breakdown.non_trivial_comment_count
                    if decision is not None
                    else 0
                ),
                complaint_signal_comment_count=(
                    decision.breakdown.complaint_signal_comment_count
                    if decision is not None
                    else 0
                ),
            )
        )

    screened_cluster_post_count = sum(1 for item in members if item.kept)
    complaint_signal_post_count = sum(
        1
        for item in members
        if item.kept and item.complaint_signal_comment_count > 0
    )

    failure_reason: str | None = None
    if strongest_post_ids and min_cluster_complaint_posts > 0 and not screening_available:
        failure_reason = "missing_candidate_screening"
    elif complaint_signal_post_count < min_cluster_complaint_posts:
        failure_reason = "insufficient_cluster_complaint_signal_posts"

    return ClusterEvidenceValidationArtifact(
        run_dir=str(run_dir),
        generated_at=generated_at or datetime.now(UTC),
        strongest_cluster_id=theme_summary.strongest_cluster_id,
        strongest_cluster_post_count=len(strongest_post_ids),
        screening_available=screening_available,
        screened_cluster_post_count=screened_cluster_post_count,
        complaint_signal_post_count=complaint_signal_post_count,
        min_cluster_complaint_posts=min_cluster_complaint_posts,
        passes=failure_reason is None,
        failure_reason=failure_reason,
        members=members,
    )


def _build_theme_cluster(
    cluster_index: int,
    members: list[RankedCandidatePost],
    term_counter: Counter[str],
) -> ThemeCluster:
    member_ids = [member.candidate.id for member in members]
    top_terms = [term for term, _ in term_counter.most_common(5)]
    label = " / ".join(top_terms[:3]) if top_terms else f"cluster-{cluster_index}"
    average_score = (
        sum(member.breakdown.total_score for member in members) / len(members)
        if members
        else 0.0
    )
    total_comments = sum(member.candidate.num_comments or 0 for member in members)
    cohesion = _cluster_cohesion(members)
    return ThemeCluster(
        cluster_id=f"cluster-{cluster_index}",
        label=label,
        post_ids=member_ids,
        size=len(members),
        average_post_score=round(average_score, 3),
        total_comment_count=total_comments,
        top_terms=top_terms,
        member_ranks=[member.rank for member in members],
        cohesion_score=round(cohesion, 3),
    )


def _cluster_cohesion(members: list[RankedCandidatePost]) -> float:
    if len(members) <= 1:
        return 1.0
    similarities: list[float] = []
    for index, left in enumerate(members):
        left_tokens = _post_tokens(left.candidate)
        for right in members[index + 1:]:
            right_tokens = _post_tokens(right.candidate)
            similarities.append(_jaccard(left_tokens, right_tokens))
    return sum(similarities) / len(similarities) if similarities else 0.0


def _post_tokens(candidate: CandidatePost) -> set[str]:
    text = " ".join([candidate.title, candidate.selftext, " ".join(candidate.source_queries)])
    tokens = {
        _normalize_token(token)
        for token in re.findall(r"\b[a-z0-9]{3,}\b", text.lower())
        if _normalize_token(token) not in STOPWORDS
    }
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left.union(right)
    if not union:
        return 0.0
    return len(left.intersection(right)) / len(union)


def _normalize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _load_theme_summary(run_dir: Path) -> ThemeSummaryArtifact:
    theme_summary_path = run_dir / "theme_summary.json"
    if not theme_summary_path.exists():
        raise FileNotFoundError(f"theme summary not found: {theme_summary_path}")
    payload = json.loads(theme_summary_path.read_text(encoding="utf-8"))
    return ThemeSummaryArtifact.model_validate(payload)


def _load_candidate_screening(
    run_dir: Path,
) -> CandidateScreeningArtifact | None:
    path = run_dir / "candidate_screening.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CandidateScreeningArtifact.model_validate(payload)
