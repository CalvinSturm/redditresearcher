from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

from .artifact_store import ArtifactStore
from .clustering import load_strongest_cluster_posts
from .llm import LMStudioClient
from .models import EvidenceSummaryArtifact, FinalMemoArtifact, ThemeCluster, ThemeSummaryArtifact
from .prompts import build_final_memo_prompt


def load_evidence_summary(run_dir: Path) -> EvidenceSummaryArtifact:
    path = run_dir / "evidence_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"evidence summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return EvidenceSummaryArtifact.model_validate(payload)


def load_theme_summary(run_dir: Path) -> ThemeSummaryArtifact:
    path = run_dir / "theme_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"theme summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ThemeSummaryArtifact.model_validate(payload)


def load_strongest_cluster(run_dir: Path) -> ThemeCluster:
    artifact = load_theme_summary(run_dir)
    if not artifact.strongest_cluster_id:
        raise ValueError("theme_summary.json does not contain a strongest cluster")
    for cluster in artifact.clusters:
        if cluster.cluster_id == artifact.strongest_cluster_id:
            return cluster
    raise ValueError(
        f"strongest cluster {artifact.strongest_cluster_id!r} is missing from theme_summary.json"
    )


async def write_final_memo(
    run_dir: Path,
    client: LMStudioClient,
    *,
    model: str | None = None,
    min_cluster_posts: int = 5,
    max_posts: int = 8,
) -> FinalMemoArtifact:
    if min_cluster_posts <= 0:
        raise ValueError("min_cluster_posts must be greater than 0")
    if max_posts <= 0:
        raise ValueError("max_posts must be greater than 0")

    store = ArtifactStore(run_dir)
    strongest_cluster = load_strongest_cluster(run_dir)
    if strongest_cluster.size < min_cluster_posts:
        raise ValueError(
            "strongest cluster is too weak for memo generation: "
            f"{strongest_cluster.size} posts found, requires at least {min_cluster_posts}"
        )

    posts = load_strongest_cluster_posts(run_dir)
    if len(posts) < min_cluster_posts:
        raise ValueError(
            "strongest cluster post set is too small for memo generation: "
            f"{len(posts)} posts found, requires at least {min_cluster_posts}"
        )

    evidence_summary = load_evidence_summary(run_dir)
    prompt = build_final_memo_prompt(
        strongest_cluster,
        posts,
        evidence_summary.summary_text,
        max_posts=max_posts,
    )
    generation = await client.generate_response(prompt, model=model)

    prompt_artifact_path = store.write_prompt_text("final_memo", prompt)
    raw_response_artifact_path = store.write_raw_llm_response("final_memo", generation.raw_response)
    final_memo_markdown = build_final_memo_markdown(
        generation.output_text,
        provider=generation.provider,
        model=generation.model,
        strongest_cluster=strongest_cluster,
        included_post_count=min(len(posts), max_posts),
    )
    store.write_final_memo_markdown(final_memo_markdown)

    artifact = FinalMemoArtifact(
        run_dir=str(run_dir),
        generated_at=datetime.now(UTC),
        provider=generation.provider,
        model=generation.model,
        strongest_cluster_id=strongest_cluster.cluster_id,
        strongest_cluster_size=strongest_cluster.size,
        included_post_count=min(len(posts), max_posts),
        prompt_artifact_path=prompt_artifact_path,
        raw_response_artifact_path=raw_response_artifact_path,
        final_memo_artifact_path=str(store.final_memo_markdown_path.relative_to(store.run_dir)),
        memo_text=generation.output_text,
    )
    store.write_final_memo_json(artifact.model_dump(mode="json"))
    return artifact


def build_final_memo_markdown(
    memo_text: str,
    *,
    provider: str,
    model: str,
    strongest_cluster: ThemeCluster,
    included_post_count: int,
) -> str:
    return "\n".join(
        [
            "# Final Memo",
            "",
            f"- provider: {provider}",
            f"- model: {model}",
            f"- strongest_cluster_id: {strongest_cluster.cluster_id}",
            f"- strongest_cluster_label: {strongest_cluster.label}",
            f"- strongest_cluster_size: {strongest_cluster.size}",
            f"- strongest_cluster_top_terms: {', '.join(strongest_cluster.top_terms) if strongest_cluster.top_terms else 'unknown'}",
            f"- strongest_cluster_total_comments: {strongest_cluster.total_comment_count}",
            f"- posts_used: {included_post_count}",
            "",
            memo_text.strip(),
        ]
    )
