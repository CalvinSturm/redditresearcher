from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

from .models import CandidatePost, RequestLogEntry, RunManifest, SearchRequestSpec, SubmissionCommentsArtifact


class ArtifactStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.raw_search_dir = run_dir / "raw" / "search"
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
        self.run_report_json_path = run_dir / "run_report.json"
        self.raw_search_dir.mkdir(parents=True, exist_ok=True)
        self.raw_comments_dir.mkdir(parents=True, exist_ok=True)
        self.raw_llm_dir.mkdir(parents=True, exist_ok=True)
        self.comments_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, manifest: RunManifest) -> None:
        _atomic_write_json(self.manifest_path, manifest.model_dump(mode="json"))

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
        return str(path.relative_to(self.run_dir))

    def write_candidate_posts(self, posts: list[CandidatePost]) -> None:
        _atomic_write_json(
            self.candidate_posts_path,
            [post.model_dump(mode="json") for post in posts],
        )

    def write_post_ranking_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.post_ranking_json_path, payload)

    def write_selected_posts_json(self, payload: list[dict[str, Any]]) -> None:
        _atomic_write_json(self.selected_posts_json_path, payload)

    def write_theme_summary_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.theme_summary_json_path, payload)

    def write_cluster_evidence_validation_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.cluster_evidence_validation_json_path, payload)

    def write_raw_comment_payload(self, submission_id: str, payload: Any) -> str:
        path = self.raw_comments_dir / f"{_safe_name(submission_id)}.json"
        _atomic_write_json(path, payload)
        return str(path.relative_to(self.run_dir))

    def write_submission_comments(self, artifact: SubmissionCommentsArtifact) -> str:
        path = self.comments_dir / f"{_safe_name(artifact.submission_id)}.json"
        _atomic_write_json(path, artifact.model_dump(mode="json"))
        return str(path.relative_to(self.run_dir))

    def write_comment_enrichment_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.comment_enrichment_json_path, payload)

    def write_candidate_screening_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.candidate_screening_json_path, payload)

    def write_comment_selection_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.comment_selection_json_path, payload)

    def write_prompt_text(self, name: str, prompt: str) -> str:
        path = self.prompts_dir / f"{_safe_name(name)}.txt"
        _atomic_write_text(path, prompt)
        return str(path.relative_to(self.run_dir))

    def write_raw_llm_response(self, name: str, payload: dict[str, Any]) -> str:
        path = self.raw_llm_dir / f"{_safe_name(name)}.json"
        _atomic_write_json(path, payload)
        return str(path.relative_to(self.run_dir))

    def write_evidence_summary_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.evidence_summary_json_path, payload)

    def write_evidence_summary_markdown(self, content: str) -> None:
        _atomic_write_text(self.evidence_summary_markdown_path, content)

    def write_final_memo_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.final_memo_json_path, payload)

    def write_final_memo_markdown(self, content: str) -> None:
        _atomic_write_text(self.final_memo_markdown_path, content)

    def write_run_report_json(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.run_report_json_path, payload)


def build_artifact_store(
    output_root: Path,
    run_slug: str,
    explicit_output_dir: Path | None = None,
) -> ArtifactStore:
    run_dir = explicit_output_dir or (output_root / run_slug)
    run_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactStore(run_dir)


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
