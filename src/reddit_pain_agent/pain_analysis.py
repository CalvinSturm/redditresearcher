from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re

from .artifact_store import ArtifactStore
from .clustering import load_strongest_cluster_posts
from .llm import LLMClient
from .models import (
    AssetGenerationProvenance,
    CandidatePost,
    Comment,
    CommentSelectionArtifact,
    CommentSelectionBreakdown,
    EvidenceSummaryArtifact,
    RunManifest,
    SelectedCommentEvidence,
    SubmissionCommentsArtifact,
)
from .prompts import build_candidate_evidence_prompt
from .ranking import load_selected_posts


def load_candidate_posts(run_dir: Path) -> list[CandidatePost]:
    candidate_posts_path = run_dir / "candidate_posts.json"
    if not candidate_posts_path.exists():
        raise FileNotFoundError(f"candidate posts not found: {candidate_posts_path}")

    payload = json.loads(candidate_posts_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("candidate_posts.json must contain a list")
    return [CandidatePost.model_validate(item) for item in payload]


def load_summary_posts(run_dir: Path) -> list[CandidatePost]:
    strongest_cluster_posts = load_strongest_cluster_posts(run_dir)
    if strongest_cluster_posts:
        return strongest_cluster_posts
    selected_posts = load_selected_posts(run_dir)
    if selected_posts:
        return selected_posts
    return load_candidate_posts(run_dir)


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


def load_run_manifest(run_dir: Path) -> RunManifest | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return RunManifest.model_validate(payload)


def select_comment_evidence(
    posts: list[CandidatePost],
    comments_by_submission: dict[str, list[Comment]],
    *,
    max_posts: int,
    max_comments_per_post: int = 3,
) -> dict[str, list[SelectedCommentEvidence]]:
    selected_by_submission: dict[str, list[SelectedCommentEvidence]] = {}
    for post in posts[:max_posts]:
        scored_comments = [
            SelectedCommentEvidence(
                submission_id=post.id,
                comment_id=comment.id,
                body=comment.body,
                score=comment.score,
                depth=comment.depth,
                permalink=comment.permalink,
                breakdown=score_comment_for_evidence(comment),
            )
            for comment in comments_by_submission.get(post.id, [])
            if comment.id and comment.body.strip()
        ]
        scored_comments.sort(
            key=lambda item: (
                item.breakdown.total_score,
                item.score or 0,
                -(item.depth or 0),
            ),
            reverse=True,
        )
        if scored_comments:
            selected_by_submission[post.id] = scored_comments[:max_comments_per_post]
    return selected_by_submission


def score_comment_for_evidence(comment: Comment) -> CommentSelectionBreakdown:
    body = comment.body.strip()
    lower_body = body.lower()
    word_count = len(body.split())

    length_score = min(max(word_count - 4, 0) / 12.0, 3.0)
    engagement_score = min(max((comment.score or 0), 0) / 5.0, 3.0)
    depth_score = min(float(comment.depth or 0) * 0.35, 1.5)
    detail_score = 0.0
    if re.search(r"\b(i|we|my|our|me)\b", lower_body):
        detail_score += 0.8
    if re.search(r"\b(because|when|after|before|still|every|manually|workflow|spreadsheet|crm|follow-?up)\b", lower_body):
        detail_score += 0.8
    if re.search(r"\d", body):
        detail_score += 0.4
    if any(token in body for token in [".", ",", ";", ":"]):
        detail_score += 0.3

    penalty_score = 0.0
    if word_count < 5:
        penalty_score -= 1.5
    if lower_body in {"same", "same here", "this", "me too", "+1"}:
        penalty_score -= 1.5
    if len(body) > 600:
        penalty_score -= 0.5

    total_score = length_score + engagement_score + depth_score + detail_score + penalty_score
    return CommentSelectionBreakdown(
        length_score=round(length_score, 3),
        engagement_score=round(engagement_score, 3),
        depth_score=round(depth_score, 3),
        detail_score=round(detail_score, 3),
        penalty_score=round(penalty_score, 3),
        total_score=round(total_score, 3),
    )


async def summarize_candidate_posts(
    run_dir: Path,
    client: LLMClient,
    model: str | None = None,
    max_posts: int = 10,
    max_comments_per_post: int = 3,
) -> EvidenceSummaryArtifact:
    store = ArtifactStore(run_dir)
    posts = load_summary_posts(run_dir)
    comments_by_submission = load_submission_comments(run_dir)
    manifest = load_run_manifest(run_dir)
    selected_comments_by_submission = select_comment_evidence(
        posts,
        comments_by_submission,
        max_posts=max_posts,
        max_comments_per_post=max_comments_per_post,
    )
    prompt = build_candidate_evidence_prompt(
        posts,
        comments_by_submission={
            submission_id: [
                Comment(
                    id=item.comment_id,
                    body=item.body,
                    score=item.score,
                    depth=item.depth,
                    permalink=item.permalink,
                )
                for item in selected_items
            ]
            for submission_id, selected_items in selected_comments_by_submission.items()
        },
        research_context=manifest,
        max_posts=max_posts,
        max_comments_per_post=max_comments_per_post,
    )
    generation = await client.generate_response(prompt, model=model)
    generation_metadata = AssetGenerationProvenance(
        provider=generation.provider,
        model=generation.model,
    )

    prompt_artifact_path = store.write_prompt_text(
        "candidate_evidence_summary",
        prompt,
        generation=generation_metadata,
    )
    generation_metadata = generation_metadata.model_copy(
        update={"prompt_artifact_path": prompt_artifact_path}
    )
    raw_response_artifact_path = store.write_raw_llm_response(
        "candidate_evidence_summary",
        generation.raw_response,
        generation=generation_metadata,
    )
    generation_metadata = generation_metadata.model_copy(
        update={"raw_response_artifact_path": raw_response_artifact_path}
    )
    summary_markdown = build_evidence_summary_markdown(
        generation.output_text,
        provider=generation.provider,
        model=generation.model,
        candidate_count=min(len(posts), max_posts),
        comment_count=sum(
            len(comments_by_submission.get(post.id, []))
            for post in posts[:max_posts]
        ),
        selected_comment_count=sum(
            len(selected_comments_by_submission.get(post.id, []))
            for post in posts[:max_posts]
        ),
    )
    store.write_evidence_summary_markdown(
        summary_markdown,
        generation=generation_metadata,
    )
    comment_selection_artifact = CommentSelectionArtifact(
        run_dir=str(run_dir),
        generated_at=datetime.now(UTC),
        total_saved_comments=sum(
            len(comments_by_submission.get(post.id, []))
            for post in posts[:max_posts]
        ),
        selected_comment_count=sum(
            len(selected_comments_by_submission.get(post.id, []))
            for post in posts[:max_posts]
        ),
        max_comments_per_post=max_comments_per_post,
        selections=[
            item
            for post in posts[:max_posts]
            for item in selected_comments_by_submission.get(post.id, [])
        ],
    )
    store.write_comment_selection_json(comment_selection_artifact.model_dump(mode="json"))

    artifact = EvidenceSummaryArtifact(
        run_dir=str(run_dir),
        generated_at=datetime.now(UTC),
        provider=generation.provider,
        model=generation.model,
        candidate_count=len(posts),
        comment_count=sum(len(comments) for comments in comments_by_submission.values()),
        selected_comment_count=comment_selection_artifact.selected_comment_count,
        max_posts_used=min(len(posts), max_posts),
        prompt_artifact_path=prompt_artifact_path,
        raw_response_artifact_path=raw_response_artifact_path,
        summary_markdown_artifact_path=str(
            store.evidence_summary_markdown_path.relative_to(store.run_dir)
        ),
        summary_text=generation.output_text,
    )
    store.write_evidence_summary_json(
        artifact.model_dump(mode="json"),
        generation=generation_metadata,
    )
    return artifact


def build_evidence_summary_markdown(
    summary_text: str,
    provider: str,
    model: str,
    candidate_count: int,
    comment_count: int = 0,
    selected_comment_count: int = 0,
) -> str:
    return "\n".join(
        [
            "# Evidence Summary",
            "",
            f"- provider: {provider}",
            f"- model: {model}",
            f"- candidate_posts_used: {candidate_count}",
            f"- saved_comments_used: {comment_count}",
            f"- selected_comments_used: {selected_comment_count}",
            "",
            summary_text.strip(),
        ]
    )
