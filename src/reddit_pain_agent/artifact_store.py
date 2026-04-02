from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

from .models import CandidatePost, RequestLogEntry, RunManifest, SearchRequestSpec


class ArtifactStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.raw_search_dir = run_dir / "raw" / "search"
        self.request_log_path = run_dir / "request_log.jsonl"
        self.manifest_path = run_dir / "manifest.json"
        self.candidate_posts_path = run_dir / "candidate_posts.json"
        self.raw_search_dir.mkdir(parents=True, exist_ok=True)

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


def _safe_name(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-") or "value"
