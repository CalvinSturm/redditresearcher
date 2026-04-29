from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

from .artifact_store import ArtifactStore
from .clustering import load_cluster_evidence_validation, load_strongest_cluster_posts
from .llm import LLMClient
from .models import EvidenceSummaryArtifact, FinalMemoArtifact, RunManifest, ThemeCluster, ThemeSummaryArtifact
from .models import AssetGenerationProvenance
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


def load_run_manifest(run_dir: Path) -> RunManifest | None:
    path = run_dir / "manifest.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunManifest.model_validate(payload)


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
    client: LLMClient,
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
    evidence_validation = load_cluster_evidence_validation(run_dir)
    if strongest_cluster.size < min_cluster_posts:
        raise ValueError(
            "strongest cluster is too weak for memo generation: "
            f"{strongest_cluster.size} posts found, requires at least {min_cluster_posts}"
        )
    if not strongest_cluster.minimum_theme_size_met:
        raise ValueError(
            "strongest cluster does not meet the minimum valid theme size of 5 posts"
        )
    if evidence_validation is not None and not evidence_validation.passes:
        raise ValueError(
            "strongest cluster failed evidence validation: "
            f"{evidence_validation.failure_reason or 'unknown'}"
        )

    posts = load_strongest_cluster_posts(run_dir)
    if len(posts) < min_cluster_posts:
        raise ValueError(
            "strongest cluster post set is too small for memo generation: "
            f"{len(posts)} posts found, requires at least {min_cluster_posts}"
        )

    evidence_summary = load_evidence_summary(run_dir)
    manifest = load_run_manifest(run_dir)
    prompt = build_final_memo_prompt(
        strongest_cluster,
        posts,
        evidence_summary.summary_text,
        research_context=manifest,
        max_posts=max_posts,
    )
    generation = await client.generate_response(prompt, model=model)
    generation_metadata = AssetGenerationProvenance(
        provider=generation.provider,
        model=generation.model,
    )

    prompt_artifact_path = store.write_prompt_text(
        "final_memo",
        prompt,
        generation=generation_metadata,
    )
    generation_metadata = generation_metadata.model_copy(
        update={"prompt_artifact_path": prompt_artifact_path}
    )
    raw_response_artifact_path = store.write_raw_llm_response(
        "final_memo",
        generation.raw_response,
        generation=generation_metadata,
    )
    generation_metadata = generation_metadata.model_copy(
        update={"raw_response_artifact_path": raw_response_artifact_path}
    )
    included_posts = posts[:max_posts]
    source_thread_urls = _unique_urls(included_posts)
    validation_issues = validate_final_memo_text(generation.output_text, source_thread_urls=source_thread_urls)
    final_memo_markdown = build_final_memo_markdown(
        generation.output_text,
        provider=generation.provider,
        model=generation.model,
        strongest_cluster=strongest_cluster,
        included_post_count=len(included_posts),
        topic=manifest.topic if manifest else None,
        target_audience=manifest.target_audience if manifest else None,
        category=manifest.category if manifest else None,
        time_horizon=(manifest.time_horizon if manifest else None),
        source_thread_urls=source_thread_urls,
    )
    store.write_final_memo_markdown(
        final_memo_markdown,
        generation=generation_metadata,
    )

    artifact = FinalMemoArtifact(
        run_dir=str(run_dir),
        generated_at=datetime.now(UTC),
        provider=generation.provider,
        model=generation.model,
        strongest_cluster_id=strongest_cluster.cluster_id,
        strongest_cluster_size=strongest_cluster.size,
        included_post_count=len(included_posts),
        topic=manifest.topic if manifest else None,
        target_audience=manifest.target_audience if manifest else None,
        category=manifest.category if manifest else None,
        time_horizon=(manifest.time_horizon if manifest else None),
        source_thread_urls=source_thread_urls,
        passed_validation=not validation_issues,
        validation_issues=validation_issues,
        prompt_artifact_path=prompt_artifact_path,
        raw_response_artifact_path=raw_response_artifact_path,
        final_memo_artifact_path=str(store.final_memo_markdown_path.relative_to(store.run_dir)),
        memo_text=generation.output_text,
    )
    store.write_final_memo_json(
        artifact.model_dump(mode="json"),
        generation=generation_metadata,
    )
    return artifact


def build_final_memo_markdown(
    memo_text: str,
    *,
    provider: str,
    model: str,
    strongest_cluster: ThemeCluster,
    included_post_count: int,
    topic: str | None = None,
    target_audience: str | None = None,
    category: str | None = None,
    time_horizon: str | None = None,
    source_thread_urls: list[str] | None = None,
) -> str:
    lines = [
        "# Final Memo",
        "",
        f"- provider: {provider}",
        f"- model: {model}",
        f"- topic: {topic or 'unspecified'}",
        f"- target_audience: {target_audience or 'unspecified'}",
        f"- category: {category or 'unspecified'}",
        f"- time_horizon: {time_horizon or 'unspecified'}",
        f"- strongest_cluster_id: {strongest_cluster.cluster_id}",
        f"- strongest_cluster_label: {strongest_cluster.label}",
        f"- strongest_cluster_size: {strongest_cluster.size}",
        f"- strongest_cluster_top_terms: {', '.join(strongest_cluster.top_terms) if strongest_cluster.top_terms else 'unknown'}",
        f"- strongest_cluster_total_comments: {strongest_cluster.total_comment_count}",
        f"- posts_used: {included_post_count}",
        "",
        memo_text.strip(),
    ]
    urls = [url for url in (source_thread_urls or []) if url.strip()]
    if urls:
        lines.extend(["", "## Source Threads", ""])
        lines.extend(f"- {url}" for url in urls)
    return "\n".join(lines)


def _unique_urls(posts: list[object]) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for post in posts:
        url = getattr(post, "url", None)
        normalized = str(url or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        urls.append(normalized)
    return urls


def validate_final_memo_text(
    memo_text: str,
    *,
    source_thread_urls: list[str] | None = None,
) -> list[str]:
    lower_text = memo_text.lower()
    required_sections = [
        "## topic overview",
        "## top repeated pain themes",
        "## product opportunities",
        "## best single opportunity",
        "## risks and caveats",
    ]
    issues = [
        f"missing_section:{section.replace('## ', '').replace(' ', '_')}"
        for section in required_sections
        if section not in lower_text
    ]
    if not (source_thread_urls or []):
        issues.append("missing_source_threads")
    return issues
