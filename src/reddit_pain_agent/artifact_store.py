from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any
from datetime import UTC, datetime

from .models import (
    AssetGenerationProvenance,
    CandidatePost,
    RegisteredAsset,
    RequestLogEntry,
    RunManifest,
    SearchRequestSpec,
    SubmissionCommentsArtifact,
)


class ArtifactStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.raw_search_dir = run_dir / "raw" / "search"
        self.raw_manual_dir = run_dir / "raw" / "manual"
        self.raw_comments_dir = run_dir / "raw" / "comments"
        self.raw_llm_dir = run_dir / "raw" / "llm"
        self.comments_dir = run_dir / "comments"
        self.prompts_dir = run_dir / "prompts"
        self.request_log_path = run_dir / "request_log.jsonl"
        self.manifest_path = run_dir / "manifest.json"
        self.candidate_posts_path = run_dir / "candidate_posts.json"
        self.post_ranking_json_path = run_dir / "post_ranking.json"
        self.selected_posts_json_path = run_dir / "selected_posts.json"
        self.theme_summary_json_path = run_dir / "theme_summary.json"
        self.cluster_evidence_validation_json_path = run_dir / "cluster_evidence_validation.json"
        self.comment_enrichment_json_path = run_dir / "comment_enrichment.json"
        self.candidate_screening_json_path = run_dir / "candidate_screening.json"
        self.comment_selection_json_path = run_dir / "comment_selection.json"
        self.evidence_summary_json_path = run_dir / "evidence_summary.json"
        self.evidence_summary_markdown_path = run_dir / "evidence_summary.md"
        self.final_memo_json_path = run_dir / "final_memo.json"
        self.final_memo_markdown_path = run_dir / "final_memo.md"
        self.comment_opportunities_json_path = run_dir / "comment_opportunities.json"
        self.reply_drafts_json_path = run_dir / "reply_drafts.json"
        self.reply_drafts_markdown_path = run_dir / "reply_drafts.md"
        self.memo_review_json_path = run_dir / "review_memo.json"
        self.reply_review_json_path = run_dir / "review_reply.json"
        self.asset_registry_path = run_dir / "asset_registry.json"
        self.run_report_json_path = run_dir / "run_report.json"
        self.raw_search_dir.mkdir(parents=True, exist_ok=True)
        self.raw_manual_dir.mkdir(parents=True, exist_ok=True)
        self.raw_comments_dir.mkdir(parents=True, exist_ok=True)
        self.raw_llm_dir.mkdir(parents=True, exist_ok=True)
        self.comments_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, manifest: RunManifest) -> None:
        _atomic_write_json(self.manifest_path, manifest.model_dump(mode="json"))
        self.register_asset("manifest.json", artifact_type="manifest")

    def append_request_log(self, entry: RequestLogEntry) -> None:
        with self.request_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")

    def write_raw_search_payload(
        self,
        index: int,
        spec: SearchRequestSpec,
        payload: dict[str, Any],
    ) -> str:
        filename = (
            f"{index:03d}-{_safe_name(spec.subreddit)}-{_safe_name(spec.query)[:40]}"
            f"-{spec.sort}-{spec.time_filter}.json"
        )
        path = self.raw_search_dir / filename
        _atomic_write_json(path, payload)
        relative_path = _relative_path(path, self.run_dir)
        self.register_asset(relative_path, artifact_type="raw_search_payload")
        return relative_path

    def write_raw_manual_payload(self, name: str, payload: Any) -> str:
        path = self.raw_manual_dir / f"{_safe_name(name)}.json"
        _atomic_write_json(path, payload)
        relative_path = _relative_path(path, self.run_dir)
        self.register_asset(relative_path, artifact_type="raw_manual_payload")
        return relative_path

    def write_candidate_posts(self, posts: list[CandidatePost]) -> None:
        _atomic_write_json(
            self.candidate_posts_path,
            [post.model_dump(mode="json") for post in posts],
        )
        self.register_asset("candidate_posts.json", artifact_type="candidate_posts")

    def write_post_ranking_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.post_ranking_json_path, payload)
        self.register_asset("post_ranking.json", artifact_type="post_ranking")

    def write_selected_posts_json(self, payload: list[dict[str, Any]]) -> None:
        _atomic_write_json(self.selected_posts_json_path, payload)
        self.register_asset("selected_posts.json", artifact_type="selected_posts")

    def write_theme_summary_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.theme_summary_json_path, payload)
        self.register_asset("theme_summary.json", artifact_type="theme_summary")

    def write_cluster_evidence_validation_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.cluster_evidence_validation_json_path, payload)
        self.register_asset(
            "cluster_evidence_validation.json",
            artifact_type="cluster_evidence_validation",
        )

    def write_raw_comment_payload(self, submission_id: str, payload: Any) -> str:
        path = self.raw_comments_dir / f"{_safe_name(submission_id)}.json"
        _atomic_write_json(path, payload)
        relative_path = _relative_path(path, self.run_dir)
        self.register_asset(relative_path, artifact_type="raw_comment_payload")
        return relative_path

    def write_submission_comments(self, artifact: SubmissionCommentsArtifact) -> str:
        path = self.comments_dir / f"{_safe_name(artifact.submission_id)}.json"
        _atomic_write_json(path, artifact.model_dump(mode="json"))
        relative_path = _relative_path(path, self.run_dir)
        self.register_asset(relative_path, artifact_type="submission_comments")
        return relative_path

    def write_comment_enrichment_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.comment_enrichment_json_path, payload)
        self.register_asset("comment_enrichment.json", artifact_type="comment_enrichment")

    def write_candidate_screening_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.candidate_screening_json_path, payload)
        self.register_asset("candidate_screening.json", artifact_type="candidate_screening")

    def write_comment_selection_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.comment_selection_json_path, payload)
        self.register_asset("comment_selection.json", artifact_type="comment_selection")

    def write_prompt_text(
        self,
        name: str,
        prompt: str,
        *,
        generation: AssetGenerationProvenance | None = None,
    ) -> str:
        path = self.prompts_dir / f"{_safe_name(name)}.txt"
        _atomic_write_text(path, prompt)
        relative_path = _relative_path(path, self.run_dir)
        resolved_generation = generation
        if resolved_generation is None:
            resolved_generation = AssetGenerationProvenance(
                prompt_artifact_path=relative_path,
            )
        elif not resolved_generation.prompt_artifact_path:
            resolved_generation = resolved_generation.model_copy(
                update={"prompt_artifact_path": relative_path}
            )
        self.register_asset(
            relative_path,
            artifact_type="prompt",
            generation=resolved_generation,
        )
        return relative_path

    def write_raw_llm_response(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        generation: AssetGenerationProvenance | None = None,
    ) -> str:
        path = self.raw_llm_dir / f"{_safe_name(name)}.json"
        _atomic_write_json(path, payload)
        relative_path = _relative_path(path, self.run_dir)
        resolved_generation = generation
        if resolved_generation is None:
            resolved_generation = AssetGenerationProvenance(
                raw_response_artifact_path=relative_path,
            )
        elif not resolved_generation.raw_response_artifact_path:
            resolved_generation = resolved_generation.model_copy(
                update={"raw_response_artifact_path": relative_path}
            )
        self.register_asset(
            relative_path,
            artifact_type="raw_llm_response",
            generation=resolved_generation,
        )
        return relative_path

    def write_evidence_summary_json(
        self,
        payload: dict[str, Any],
        *,
        generation: AssetGenerationProvenance | None = None,
    ) -> None:
        _atomic_write_json(self.evidence_summary_json_path, payload)
        self.register_asset(
            "evidence_summary.json",
            artifact_type="evidence_summary",
            generation=generation,
        )

    def write_evidence_summary_markdown(
        self,
        content: str,
        *,
        generation: AssetGenerationProvenance | None = None,
    ) -> None:
        _atomic_write_text(self.evidence_summary_markdown_path, content)
        self.register_asset(
            "evidence_summary.md",
            artifact_type="evidence_summary_markdown",
            generation=generation,
        )

    def write_final_memo_json(
        self,
        payload: dict[str, Any],
        *,
        generation: AssetGenerationProvenance | None = None,
    ) -> None:
        _atomic_write_json(self.final_memo_json_path, payload)
        self.register_asset(
            "final_memo.json",
            artifact_type="final_memo",
            generation=generation,
        )

    def write_final_memo_markdown(
        self,
        content: str,
        *,
        generation: AssetGenerationProvenance | None = None,
    ) -> None:
        _atomic_write_text(self.final_memo_markdown_path, content)
        self.register_asset(
            "final_memo.md",
            artifact_type="final_memo_markdown",
            generation=generation,
        )

    def write_comment_opportunities_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.comment_opportunities_json_path, payload)
        self.register_asset(
            "comment_opportunities.json",
            artifact_type="comment_opportunities",
        )

    def write_reply_drafts_json(
        self,
        payload: dict[str, Any],
        *,
        generation: AssetGenerationProvenance | None = None,
    ) -> None:
        _atomic_write_json(self.reply_drafts_json_path, payload)
        self.register_asset(
            "reply_drafts.json",
            artifact_type="reply_drafts",
            generation=generation,
        )

    def write_reply_drafts_markdown(
        self,
        content: str,
        *,
        generation: AssetGenerationProvenance | None = None,
    ) -> None:
        _atomic_write_text(self.reply_drafts_markdown_path, content)
        self.register_asset(
            "reply_drafts.md",
            artifact_type="reply_drafts_markdown",
            generation=generation,
        )

    def write_review_checkpoint_json(
        self,
        review_type: str,
        payload: dict[str, Any],
    ) -> None:
        path = (
            self.memo_review_json_path
            if review_type == "memo"
            else self.reply_review_json_path
        )
        _atomic_write_json(path, payload)
        self.register_asset(
            _relative_path(path, self.run_dir),
            artifact_type=f"{review_type}_review_checkpoint",
        )

    def write_run_report_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.run_report_json_path, payload)
        self.register_asset("run_report.json", artifact_type="run_report")

    def register_asset(
        self,
        artifact_path: str,
        *,
        artifact_type: str,
        generation: AssetGenerationProvenance | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        registry = self.load_asset_registry()
        entry = RegisteredAsset(
            artifact_path=artifact_path,
            artifact_type=artifact_type,
            created_at=_now_utc(),
            generation=generation,
            metadata=metadata or {},
        )
        registry = [item for item in registry if item.artifact_path != artifact_path]
        registry.append(entry)
        registry.sort(key=lambda item: (item.artifact_path, item.artifact_type))
        _atomic_write_json(
            self.asset_registry_path,
            [item.model_dump(mode="json") for item in registry],
        )

    def load_asset_registry(self) -> list[RegisteredAsset]:
        if not self.asset_registry_path.exists():
            return []
        payload = json.loads(self.asset_registry_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("asset_registry.json must contain a list")
        return [RegisteredAsset.model_validate(item) for item in payload]


def build_artifact_store(
    output_root: Path,
    run_slug: str,
    explicit_output_dir: Path | None = None,
) -> ArtifactStore:
    run_dir = explicit_output_dir or (output_root / run_slug)
    run_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactStore(run_dir)


def _now_utc() -> Any:
    return datetime.now(UTC)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        handle.write(content)
        if not content.endswith("\n"):
            handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _safe_name(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-") or "value"


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
