from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any

from .artifact_store import ArtifactStore
from .clustering import load_strongest_cluster_posts
from .llm import LMStudioClient
from .models import CandidatePost, RankedCandidatePost, ReplyDraft, ReplyDraftArtifact
from .prompts import (
    build_reply_drafts_prompt,
    build_reply_evaluation_prompt,
    build_reply_improvement_prompt,
)
from .ranking import load_selected_ranked_posts


DEFAULT_REPLY_SCORE_THRESHOLD = 4.0
DEFAULT_REPLY_MIN_DIMENSION_SCORE = 3.0
DEFAULT_REPLY_MAX_IMPROVEMENT_ROUNDS = 3


def load_reply_source_posts(run_dir: Path) -> list[RankedCandidatePost]:
    ranked_posts = load_selected_ranked_posts(run_dir)
    if ranked_posts:
        return ranked_posts

    strongest_cluster_posts = load_strongest_cluster_posts(run_dir)
    if strongest_cluster_posts:
        return [
            RankedCandidatePost(
                candidate=post,
                saved_comment_count=0,
                breakdown={"total_score": 0.0},
                rank=index,
            )
            for index, post in enumerate(strongest_cluster_posts, start=1)
        ]

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


async def draft_reply_suggestions(
    run_dir: Path,
    client: LMStudioClient,
    *,
    voice: str,
    model: str | None = None,
    max_posts: int = 3,
    score_threshold: float = DEFAULT_REPLY_SCORE_THRESHOLD,
    minimum_dimension_score: float = DEFAULT_REPLY_MIN_DIMENSION_SCORE,
    max_improvement_rounds: int = DEFAULT_REPLY_MAX_IMPROVEMENT_ROUNDS,
) -> ReplyDraftArtifact:
    if max_posts <= 0:
        raise ValueError("--max-posts must be greater than 0")
    if not voice.strip():
        raise ValueError("--voice is required")
    if score_threshold <= 0 or score_threshold > 5:
        raise ValueError("--score-threshold must be between 0 and 5")
    if minimum_dimension_score <= 0 or minimum_dimension_score > 5:
        raise ValueError("--minimum-dimension-score must be between 0 and 5")
    if max_improvement_rounds < 0:
        raise ValueError("--max-improvement-rounds must be 0 or greater")

    store = ArtifactStore(run_dir)
    source_posts = load_reply_source_posts(run_dir)
    initial_prompt = build_reply_drafts_prompt(source_posts, voice=voice, max_posts=max_posts)
    initial_generation = await client.generate_response(initial_prompt, model=model)
    selected_posts = source_posts[:max_posts]
    store.write_prompt_text("reply_drafts_initial", initial_prompt)
    store.write_raw_llm_response("reply_drafts_initial", initial_generation.raw_response)

    current_output = initial_generation.output_text
    generation = initial_generation
    final_prompt_text = initial_prompt
    evaluations: dict[str, _ReplyEvaluation] = {}
    passed_threshold = False
    completed_rounds = 0

    for round_number in range(0, max_improvement_rounds + 1):
        evaluation_prompt = build_reply_evaluation_prompt(
            selected_posts,
            current_output,
            voice=voice,
            max_posts=max_posts,
        )
        evaluation_generation = await client.generate_response(evaluation_prompt, model=model)
        store.write_prompt_text(f"reply_drafts_evaluation_round_{round_number:02d}", evaluation_prompt)
        store.write_raw_llm_response(
            f"reply_drafts_evaluation_round_{round_number:02d}",
            evaluation_generation.raw_response,
        )
        evaluations = _parse_reply_evaluations(evaluation_generation.output_text, selected_posts)
        if _evaluations_pass_threshold(
            evaluations,
            selected_posts,
            score_threshold=score_threshold,
            minimum_dimension_score=minimum_dimension_score,
        ):
            passed_threshold = True
            completed_rounds = round_number
            break
        if round_number >= max_improvement_rounds:
            completed_rounds = round_number
            break

        improvement_prompt = build_reply_improvement_prompt(
            selected_posts,
            current_output,
            voice=voice,
            max_posts=max_posts,
            evaluation_feedback={post_id: item.feedback for post_id, item in evaluations.items()},
            round_number=round_number + 1,
        )
        generation = await client.generate_response(improvement_prompt, model=model)
        store.write_prompt_text(f"reply_drafts_revision_round_{round_number + 1:02d}", improvement_prompt)
        store.write_raw_llm_response(
            f"reply_drafts_revision_round_{round_number + 1:02d}",
            generation.raw_response,
        )
        final_prompt_text = improvement_prompt
        current_output = generation.output_text

    drafts = _parse_reply_drafts(current_output, selected_posts, evaluations)
    prompt_artifact_path = store.write_prompt_text("reply_drafts", final_prompt_text)
    raw_response_artifact_path = store.write_raw_llm_response("reply_drafts", generation.raw_response)
    markdown = build_reply_drafts_markdown(
        drafts,
        provider=generation.provider,
        model=generation.model,
        voice=voice,
    )
    store.write_reply_drafts_markdown(markdown)

    artifact = ReplyDraftArtifact(
        run_dir=str(run_dir),
        generated_at=datetime.now(UTC),
        provider=generation.provider,
        model=generation.model,
        selected_post_count=len(drafts),
        voice=voice,
        improvement_rounds=completed_rounds,
        score_threshold=score_threshold,
        minimum_dimension_score=minimum_dimension_score,
        passed_threshold=passed_threshold,
        prompt_artifact_path=prompt_artifact_path,
        raw_response_artifact_path=raw_response_artifact_path,
        reply_markdown_artifact_path=str(store.reply_drafts_markdown_path.relative_to(store.run_dir)),
        drafts=drafts,
    )
    store.write_reply_drafts_json(artifact.model_dump(mode="json"))
    return artifact


def build_reply_drafts_markdown(
    drafts: list[ReplyDraft],
    *,
    provider: str,
    model: str,
    voice: str,
) -> str:
    lines = [
        "# Reply Drafts",
        "",
        f"- provider: {provider}",
        f"- model: {model}",
        f"- selected_post_count: {len(drafts)}",
        f"- voice: {voice.strip()}",
        "- manual_review_only: yes",
        "",
    ]
    for index, draft in enumerate(drafts, start=1):
        lines.extend(
            [
                f"## Post {index}",
                f"- post_id: {draft.post_id}",
                f"- subreddit: {draft.subreddit}",
                f"- title: {draft.title}",
                f"- url: {draft.url}",
                "",
                draft.reply_text.strip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _parse_reply_drafts(
    output_text: str,
    selected_posts: list[RankedCandidatePost],
    evaluations: dict[str, "_ReplyEvaluation"] | None = None,
) -> list[ReplyDraft]:
    blocks = re.split(r"(?m)^## Post \d+\s*$", output_text)
    parsed_by_post_id: dict[str, str] = {}
    for block in blocks:
        post_id_match = re.search(r"(?mi)^post_id:\s*(.+?)\s*$", block)
        reply_match = re.search(r"(?mis)^reply:\s*(.*)$", block)
        if not post_id_match or not reply_match:
            continue
        post_id = post_id_match.group(1).strip()
        reply_text = _normalize_reply_text(reply_match.group(1).strip())
        if post_id and reply_text:
            parsed_by_post_id[post_id] = reply_text

    drafts: list[ReplyDraft] = []
    for post in selected_posts:
        candidate = post.candidate
        reply_text = parsed_by_post_id.get(candidate.id)
        if not reply_text:
            reply_text = (
                "I can see why this is frustrating. When the process keeps falling back to manual effort, "
                "it usually means the system still is not dependable enough in the moments that matter.\n\n"
                "That part stands out to me more than the tool choice itself, because once people stop trusting the workflow they start building side habits around it."
            )
        evaluation = (evaluations or {}).get(candidate.id)
        drafts.append(
            ReplyDraft(
                post_id=candidate.id,
                title=candidate.title,
                subreddit=candidate.subreddit,
                url=candidate.url,
                rank=post.rank,
                reply_text=reply_text,
                relevance_score=evaluation.relevance_score if evaluation else None,
                conversation_value_score=evaluation.conversation_value_score if evaluation else None,
                voice_match_score=evaluation.voice_match_score if evaluation else None,
                reddit_friendliness_score=evaluation.reddit_friendliness_score if evaluation else None,
                average_score=evaluation.average_score if evaluation else None,
                passed_threshold=evaluation.passed_threshold if evaluation else None,
                evaluation_feedback=evaluation.feedback if evaluation else None,
            )
        )
    return drafts


def _normalize_reply_text(reply_text: str) -> str:
    cleaned = reply_text.replace("\r\n", "\n").strip()
    cleaned = re.sub(r"(?m)^\s{0,3}(?:[-*]|\d+\.)\s+", "", cleaned)
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", cleaned) if paragraph.strip()]
    normalized_paragraphs: list[str] = []
    for paragraph in paragraphs[:3]:
        paragraph = re.sub(r"(?m)^#{1,6}\s+", "", paragraph).strip()
        if paragraph:
            normalized_paragraphs.append(paragraph)
    if not normalized_paragraphs:
        return ""
    return "\n\n".join(normalized_paragraphs)


class _ReplyEvaluation:
    def __init__(
        self,
        *,
        relevance_score: float,
        conversation_value_score: float,
        voice_match_score: float,
        reddit_friendliness_score: float,
        feedback: str,
    ) -> None:
        self.relevance_score = relevance_score
        self.conversation_value_score = conversation_value_score
        self.voice_match_score = voice_match_score
        self.reddit_friendliness_score = reddit_friendliness_score
        self.feedback = feedback.strip()
        self.average_score = round(
            (
                self.relevance_score
                + self.conversation_value_score
                + self.voice_match_score
                + self.reddit_friendliness_score
            )
            / 4.0,
            3,
        )
        self.passed_threshold = False


def _parse_reply_evaluations(
    output_text: str,
    selected_posts: list[RankedCandidatePost],
) -> dict[str, _ReplyEvaluation]:
    payload = _extract_json_object(output_text)
    items = payload.get("evaluations") if isinstance(payload, dict) else None
    evaluations: dict[str, _ReplyEvaluation] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            post_id = str(item.get("post_id") or "").strip()
            if not post_id:
                continue
            evaluations[post_id] = _ReplyEvaluation(
                relevance_score=_coerce_score(item.get("relevance_score")),
                conversation_value_score=_coerce_score(item.get("conversation_value_score")),
                voice_match_score=_coerce_score(item.get("voice_match_score")),
                reddit_friendliness_score=_coerce_score(item.get("reddit_friendliness_score")),
                feedback=str(item.get("feedback") or "").strip() or "Make the reply more relevant, more useful, and more native to Reddit.",
            )

    for post in selected_posts:
        if post.candidate.id not in evaluations:
            evaluations[post.candidate.id] = _ReplyEvaluation(
                relevance_score=2.5,
                conversation_value_score=2.5,
                voice_match_score=2.5,
                reddit_friendliness_score=2.5,
                feedback="The reply needs a clearer take, stronger relevance to the post, and a more Reddit-native voice.",
            )
    return evaluations


def _evaluations_pass_threshold(
    evaluations: dict[str, _ReplyEvaluation],
    selected_posts: list[RankedCandidatePost],
    *,
    score_threshold: float,
    minimum_dimension_score: float,
) -> bool:
    if not selected_posts:
        return False
    passes = True
    for post in selected_posts:
        evaluation = evaluations.get(post.candidate.id)
        if evaluation is None:
            passes = False
            continue
        dimension_scores = [
            evaluation.relevance_score,
            evaluation.conversation_value_score,
            evaluation.voice_match_score,
            evaluation.reddit_friendliness_score,
        ]
        evaluation.passed_threshold = (
            evaluation.average_score >= score_threshold
            and min(dimension_scores) >= minimum_dimension_score
        )
        if not evaluation.passed_threshold:
            passes = False
    return passes


def _extract_json_object(output_text: str) -> dict[str, Any]:
    text = output_text.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"(?s)\{.*\}", text)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 2.5
    return max(1.0, min(score, 5.0))
