from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import hashlib
from pathlib import Path
import sys
from time import perf_counter

from .artifact_store import ArtifactStore
from .config import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_EXPAND_QUERIES,
    DEFAULT_LIMIT,
    DEFAULT_PAGES_PER_QUERY,
    DEFAULT_SORT,
    DEFAULT_TIME_FILTER,
    ConfigurationError,
    build_run_paths,
    ensure_repo_layout,
    load_llm_config,
    load_runtime_config,
)
from .clustering import cluster_run_posts
from .llm import LMStudioClient
from .memo_writer import write_final_memo
from .models import RunReportArtifact, RunStageReport
from .pain_analysis import summarize_candidate_posts
from .playwright_capture import capture_reddit_threads
from .reply_writer import draft_reply_suggestions
from .ranking import rank_run_candidates
from .retrieval import (
    enrich_run_with_comments,
    import_manual_search_bundle,
    resolve_search_plan,
    run_search_command,
)


STAGE_ORDER = ["search", "comments", "rank", "cluster", "summarize", "memo"]


def _read_template(path: Path, fallback: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


def init_run(run_name: str, root: Path | None = None) -> tuple[dict[str, Path], dict[str, str]]:
    paths = ensure_repo_layout(root)
    run_paths = build_run_paths(run_name, root)
    run_paths.output_dir.mkdir(parents=True, exist_ok=True)

    templates_dir = paths["templates_dir"]
    brief_template = _read_template(
        templates_dir / "research_brief.md",
        "# Research Brief\n\n## Topic\n\n## Target Subreddits\n\n## Notes\n",
    )

    memo_template = _read_template(
        templates_dir / "final_memo.md",
        "# Executive Summary\n",
    )

    if not run_paths.brief_path.exists():
        run_paths.brief_path.write_text(
            brief_template.replace("{{run_name}}", run_paths.slug),
            encoding="utf-8",
        )

    if not run_paths.final_memo_path.exists():
        run_paths.final_memo_path.write_text(
            memo_template.replace("{{run_name}}", run_paths.slug),
            encoding="utf-8",
        )

    return paths, {
        "slug": run_paths.slug,
        "brief": str(run_paths.brief_path),
        "output_dir": str(run_paths.output_dir),
        "candidate_posts": str(run_paths.candidate_posts_path),
        "selected_posts": str(run_paths.selected_posts_path),
        "theme_summary": str(run_paths.theme_summary_path),
        "final_memo": str(run_paths.final_memo_path),
    }


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)


def _build_run_output_paths(run_dir: Path) -> dict[str, str]:
    return {
        "manifest": str(run_dir / "manifest.json"),
        "candidate_posts": str(run_dir / "candidate_posts.json"),
        "comment_enrichment": str(run_dir / "comment_enrichment.json"),
        "candidate_screening": str(run_dir / "candidate_screening.json"),
        "post_ranking": str(run_dir / "post_ranking.json"),
        "selected_posts": str(run_dir / "selected_posts.json"),
        "theme_summary": str(run_dir / "theme_summary.json"),
        "cluster_evidence_validation": str(run_dir / "cluster_evidence_validation.json"),
        "comment_selection": str(run_dir / "comment_selection.json"),
        "evidence_summary": str(run_dir / "evidence_summary.json"),
        "final_memo": str(run_dir / "final_memo.md"),
        "final_memo_json": str(run_dir / "final_memo.json"),
        "reply_drafts": str(run_dir / "reply_drafts.md"),
        "reply_drafts_json": str(run_dir / "reply_drafts.json"),
        "run_report": str(run_dir / "run_report.json"),
    }


def _load_run_report(run_dir: Path) -> RunReportArtifact:
    path = run_dir / "run_report.json"
    if not path.exists():
        raise FileNotFoundError(f"run report not found: {path}")
    return RunReportArtifact.model_validate_json(path.read_text(encoding="utf-8"))


def _stage_params_from_args(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    manual_input = getattr(args, "manual_input", None)
    return {
        "search": {
            "search_mode": "manual" if manual_input else "api",
            "manual_input_path": str(manual_input) if manual_input else None,
            "subreddits": list(args.subreddits),
            "queries": list(args.queries),
            "sort": args.sort,
            "time_filter": args.time_filter,
            "additional_sorts": list(args.additional_sorts),
            "additional_time_filters": list(args.additional_time_filters),
            "limit": args.limit,
            "pages_per_query": args.pages_per_query,
            "expand_queries": args.expand_queries,
            "min_score": args.min_score,
            "min_comments": args.min_comments,
            "filter_nsfw": args.filter_nsfw,
            "allowed_subreddits": list(args.allowed_subreddits),
            "denied_subreddits": list(args.denied_subreddits),
        },
        "comments": {
            "max_posts": args.comment_max_posts,
            "comment_limit": args.comment_limit,
            "comment_depth": args.comment_depth,
            "comment_sort": args.comment_sort,
            "max_morechildren_requests": args.max_morechildren_requests,
            "morechildren_batch_size": args.morechildren_batch_size,
        },
        "rank": {
            "max_selected_posts": args.max_selected_posts,
            "min_non_trivial_comments": args.min_non_trivial_comments,
            "min_complaint_signal_comments": args.min_complaint_signal_comments,
        },
        "cluster": {
            "similarity_threshold": args.similarity_threshold,
            "min_shared_terms": args.min_shared_terms,
            "min_cluster_complaint_posts": args.min_cluster_complaint_posts,
        },
        "summarize": {
            "model": args.model,
            "max_posts": args.summary_max_posts,
        },
        "memo": {
            "model": args.model,
            "min_cluster_posts": args.min_cluster_posts,
            "max_posts": args.memo_max_posts,
        },
    }


def _stage_artifacts_exist(run_dir: Path, stage: str) -> bool:
    expected = _stage_artifact_paths(run_dir, stage)
    return bool(expected) and all(path.exists() for path in expected)


def _stage_artifact_paths(run_dir: Path, stage: str) -> list[Path]:
    outputs = _build_run_output_paths(run_dir)
    stage_paths: dict[str, list[Path]] = {
        "search": [Path(outputs["candidate_posts"])],
        "comments": [Path(outputs["comment_enrichment"])],
        "rank": [
            Path(outputs["candidate_screening"]),
            Path(outputs["post_ranking"]),
            Path(outputs["selected_posts"]),
        ],
        "cluster": [
            Path(outputs["theme_summary"]),
            Path(outputs["cluster_evidence_validation"]),
        ],
        "summarize": [Path(outputs["comment_selection"]), Path(outputs["evidence_summary"])],
        "memo": [Path(outputs["final_memo"]), Path(outputs["final_memo_json"])],
    }
    if stage == "comments":
        stage_paths["comments"].extend(sorted((run_dir / "comments").glob("*.json")))
    return stage_paths.get(stage, [])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_artifact_fingerprints(run_dir: Path, stage: str) -> dict[str, str]:
    artifact_paths = _stage_artifact_paths(run_dir, stage)
    if not artifact_paths or any(not path.exists() for path in artifact_paths):
        return {}
    return {
        path.relative_to(run_dir).as_posix(): _file_sha256(path)
        for path in artifact_paths
    }


def _stage_artifacts_match_fingerprints(
    run_dir: Path,
    stage: str,
    expected_fingerprints: dict[str, str],
) -> bool:
    if not expected_fingerprints:
        return False
    current_fingerprints = _stage_artifact_fingerprints(run_dir, stage)
    return current_fingerprints == expected_fingerprints


def _resolve_resume_state(
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[RunReportArtifact, list[RunStageReport], str | None]:
    previous_report = _load_run_report(run_dir)
    params_by_stage = _stage_params_from_args(args)
    completed_prefix: list[RunStageReport] = []
    previous_by_stage = {item.stage: item for item in previous_report.stage_reports}

    next_stage: str | None = None
    for stage in STAGE_ORDER:
        previous = previous_by_stage.get(stage)
        if previous is None or previous.status != "completed":
            next_stage = stage
            break
        previous_params = previous.details.get("params")
        if stage == "search":
            previous_params = _normalize_search_stage_params(previous_params)
        if previous_params != params_by_stage[stage]:
            next_stage = stage
            break
        if not _stage_artifacts_exist(run_dir, stage):
            next_stage = stage
            break
        if not _stage_artifacts_match_fingerprints(run_dir, stage, previous.artifact_fingerprints):
            next_stage = stage
            break
        completed_prefix.append(previous)
    return previous_report, completed_prefix, next_stage


def _normalize_search_stage_params(payload: object) -> dict[str, object]:
    normalized = dict(payload) if isinstance(payload, dict) else {}
    normalized.setdefault("search_mode", "api")
    normalized.setdefault("manual_input_path", None)
    return normalized


def _should_run_stage(resumed_from_stage: str | None, stage: str) -> bool:
    if resumed_from_stage is None:
        return True
    return STAGE_ORDER.index(stage) >= STAGE_ORDER.index(resumed_from_stage)


def _write_run_report(
    *,
    run_slug: str,
    run_dir: Path,
    status: str,
    started_at: datetime,
    subreddits: list[str],
    queries: list[str],
    sort: str,
    time_filter: str,
    limit: int,
    stage_reports: list[RunStageReport],
    provider: str | None = None,
    model: str | None = None,
    stop_reason: str | None = None,
    error: str | None = None,
) -> Path:
    artifact = RunReportArtifact(
        run_slug=run_slug,
        run_dir=str(run_dir),
        status=status,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        subreddits=subreddits,
        queries=queries,
        sort=sort,
        time_filter=time_filter,
        limit=limit,
        provider=provider,
        model=model,
        stop_reason=stop_reason,
        error=error,
        stage_reports=stage_reports,
        output_paths=_build_run_output_paths(run_dir),
    )
    store = ArtifactStore(run_dir)
    store.write_run_report_json(artifact.model_dump(mode="json"))
    return store.run_report_json_path


def _run_handoff_command(argv: list[str]) -> int:
    return main(argv)


def _build_capture_handoff_argv(
    args: argparse.Namespace,
    capture_output_path: Path,
) -> list[str]:
    command = ["manual-import"] if args.handoff == "manual-import" else ["run"]
    if args.handoff == "manual-import":
        command.extend(["--input", str(capture_output_path)])
    else:
        command.extend(["--manual-input", str(capture_output_path)])

    if args.output_dir is not None:
        command.extend(["--output-dir", str(args.output_dir)])

    for subreddit in args.subreddits:
        command.extend(["--subreddit", subreddit])
    for query in args.queries:
        command.extend(["--query", query])

    command.extend(["--sort", args.sort, "--time-filter", args.time_filter, "--limit", str(args.limit)])

    if args.min_score:
        command.extend(["--min-score", str(args.min_score)])
    if args.min_comments:
        command.extend(["--min-comments", str(args.min_comments)])
    if args.filter_nsfw:
        command.append("--filter-nsfw")
    for subreddit in args.allowed_subreddits:
        command.extend(["--allow-subreddit", subreddit])
    for subreddit in args.denied_subreddits:
        command.extend(["--deny-subreddit", subreddit])

    if args.handoff == "run":
        if args.model:
            command.extend(["--model", args.model])
        if args.min_cluster_posts != 5:
            command.extend(["--min-cluster-posts", str(args.min_cluster_posts)])
        if args.min_cluster_complaint_posts != 2:
            command.extend(["--min-cluster-complaint-posts", str(args.min_cluster_complaint_posts)])

    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reddit pain research scaffold CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("layout", help="Create and print the expected repo layout")

    init_parser = subparsers.add_parser("init-run", help="Create folders and templates for a run")
    init_parser.add_argument("run_name", help="Human-readable run name")

    search_parser = subparsers.add_parser("search", help="Retrieve Reddit candidate posts")
    search_parser.add_argument(
        "--subreddit",
        action="append",
        dest="subreddits",
        required=True,
        help="Target subreddit name without the r/ prefix",
    )
    search_parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        required=True,
        help="Search query to run within each target subreddit",
    )
    search_parser.add_argument("--sort", default=DEFAULT_SORT, help="Reddit search sort")
    search_parser.add_argument(
        "--time-filter",
        default=DEFAULT_TIME_FILTER,
        help="Reddit time filter",
    )
    search_parser.add_argument(
        "--additional-sort",
        action="append",
        default=[],
        dest="additional_sorts",
        help="Additional Reddit search sort to include in the deterministic search plan",
    )
    search_parser.add_argument(
        "--additional-time-filter",
        action="append",
        default=[],
        dest="additional_time_filters",
        help="Additional Reddit time filter to include in the deterministic search plan",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Per-request result limit (max 100)",
    )
    search_parser.add_argument(
        "--min-score",
        type=int,
        default=0,
        help="Minimum Reddit post score required for a candidate to survive retrieval filtering",
    )
    search_parser.add_argument(
        "--min-comments",
        type=int,
        default=0,
        help="Minimum Reddit comment count required for a candidate to survive retrieval filtering",
    )
    search_parser.add_argument(
        "--filter-nsfw",
        action="store_true",
        help="Exclude NSFW submissions during retrieval filtering",
    )
    search_parser.add_argument(
        "--allow-subreddit",
        action="append",
        default=[],
        dest="allowed_subreddits",
        help="Optional subreddit allowlist applied after retrieval normalization",
    )
    search_parser.add_argument(
        "--deny-subreddit",
        action="append",
        default=[],
        dest="denied_subreddits",
        help="Optional subreddit denylist applied after retrieval normalization",
    )
    search_parser.add_argument(
        "--pages-per-query",
        type=int,
        default=DEFAULT_PAGES_PER_QUERY,
        help="Maximum number of paginated search result pages to fetch per subreddit/query variant",
    )
    search_parser.set_defaults(expand_queries=DEFAULT_EXPAND_QUERIES)
    search_query_expansion_group = search_parser.add_mutually_exclusive_group()
    search_query_expansion_group.add_argument(
        "--expand-queries",
        dest="expand_queries",
        action="store_true",
        help="Enable deterministic query expansion for broader search coverage",
    )
    search_query_expansion_group.add_argument(
        "--no-expand-queries",
        dest="expand_queries",
        action="store_false",
        help="Disable deterministic query expansion and search only the provided queries",
    )
    search_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional explicit run directory",
    )

    capture_parser = subparsers.add_parser(
        "capture",
        help="Use Playwright to capture Reddit search results and thread pages into the manual import schema",
    )
    capture_parser.add_argument(
        "--subreddit",
        action="append",
        dest="subreddits",
        required=True,
        help="Target subreddit name without the r/ prefix",
    )
    capture_parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        required=True,
        help="Search query to run within each target subreddit",
    )
    capture_parser.add_argument("--sort", default=DEFAULT_SORT, help="Reddit search sort")
    capture_parser.add_argument(
        "--time-filter",
        default=DEFAULT_TIME_FILTER,
        help="Reddit time filter",
    )
    capture_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Metadata limit to forward if the capture is handed to manual-import or run",
    )
    capture_parser.add_argument(
        "--thread-url",
        action="append",
        default=[],
        dest="thread_urls",
        help="Optional direct Reddit thread URL to capture in addition to discovered search results",
    )
    capture_parser.add_argument(
        "--select-result",
        action="append",
        type=int,
        default=[],
        dest="select_results",
        help="1-based discovered search-result index to capture. Defaults to the top results.",
    )
    capture_parser.add_argument(
        "--max-posts",
        type=int,
        default=5,
        help="Maximum number of thread pages to capture",
    )
    capture_parser.add_argument(
        "--max-comments",
        type=int,
        default=12,
        help="Maximum number of visible comments to save per captured thread",
    )
    capture_parser.add_argument(
        "--page-timeout-seconds",
        type=float,
        default=20.0,
        help="Per-page Playwright navigation timeout",
    )
    capture_parser.add_argument(
        "--page-wait-ms",
        type=int,
        default=1000,
        help="Additional post-navigation wait time before extracting page data",
    )
    capture_parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Launch Chromium with a visible window instead of headless mode",
    )
    capture_parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip subreddit search pages and capture only the provided --thread-url values",
    )
    capture_parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional explicit path for the captured import JSON bundle",
    )
    capture_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional run directory to reuse when handing capture output to manual-import or run",
    )
    capture_parser.add_argument(
        "--handoff",
        choices=["none", "manual-import", "run"],
        default="none",
        help="What to do after capture: stop, import into a run, or execute the full run pipeline",
    )
    capture_parser.add_argument(
        "--min-score",
        type=int,
        default=0,
        help="Minimum Reddit post score required if the capture is handed to manual-import or run",
    )
    capture_parser.add_argument(
        "--min-comments",
        type=int,
        default=0,
        help="Minimum Reddit comment count required if the capture is handed to manual-import or run",
    )
    capture_parser.add_argument(
        "--filter-nsfw",
        action="store_true",
        help="Exclude NSFW captured submissions if the bundle is handed to manual-import or run",
    )
    capture_parser.add_argument(
        "--allow-subreddit",
        action="append",
        default=[],
        dest="allowed_subreddits",
        help="Optional subreddit allowlist to forward into manual-import or run",
    )
    capture_parser.add_argument(
        "--deny-subreddit",
        action="append",
        default=[],
        dest="denied_subreddits",
        help="Optional subreddit denylist to forward into manual-import or run",
    )
    capture_parser.add_argument(
        "--model",
        help="Optional model override to forward when handoff=run",
    )
    capture_parser.add_argument(
        "--min-cluster-posts",
        type=int,
        default=5,
        help="Minimum strongest-cluster size to forward when handoff=run",
    )
    capture_parser.add_argument(
        "--min-cluster-complaint-posts",
        type=int,
        default=2,
        help="Minimum strongest-cluster complaint-signal post count to forward when handoff=run",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Execute the full search-to-memo pipeline for a single run",
    )
    run_parser.add_argument(
        "--subreddit",
        action="append",
        dest="subreddits",
        required=True,
        help="Target subreddit name without the r/ prefix",
    )
    run_parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        required=True,
        help="Search query to run within each target subreddit",
    )
    run_parser.add_argument("--sort", default=DEFAULT_SORT, help="Reddit search sort")
    run_parser.add_argument(
        "--time-filter",
        default=DEFAULT_TIME_FILTER,
        help="Reddit time filter",
    )
    run_parser.add_argument(
        "--additional-sort",
        action="append",
        default=[],
        dest="additional_sorts",
        help="Additional Reddit search sort to include in the deterministic search plan",
    )
    run_parser.add_argument(
        "--additional-time-filter",
        action="append",
        default=[],
        dest="additional_time_filters",
        help="Additional Reddit time filter to include in the deterministic search plan",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Per-request result limit (max 100)",
    )
    run_parser.add_argument(
        "--min-score",
        type=int,
        default=0,
        help="Minimum Reddit post score required for a candidate to survive retrieval filtering",
    )
    run_parser.add_argument(
        "--min-comments",
        type=int,
        default=0,
        help="Minimum Reddit comment count required for a candidate to survive retrieval filtering",
    )
    run_parser.add_argument(
        "--filter-nsfw",
        action="store_true",
        help="Exclude NSFW submissions during retrieval filtering",
    )
    run_parser.add_argument(
        "--allow-subreddit",
        action="append",
        default=[],
        dest="allowed_subreddits",
        help="Optional subreddit allowlist applied after retrieval normalization",
    )
    run_parser.add_argument(
        "--deny-subreddit",
        action="append",
        default=[],
        dest="denied_subreddits",
        help="Optional subreddit denylist applied after retrieval normalization",
    )
    run_parser.add_argument(
        "--pages-per-query",
        type=int,
        default=DEFAULT_PAGES_PER_QUERY,
        help="Maximum number of paginated search result pages to fetch per subreddit/query variant",
    )
    run_parser.set_defaults(expand_queries=DEFAULT_EXPAND_QUERIES)
    run_query_expansion_group = run_parser.add_mutually_exclusive_group()
    run_query_expansion_group.add_argument(
        "--expand-queries",
        dest="expand_queries",
        action="store_true",
        help="Enable deterministic query expansion for broader search coverage",
    )
    run_query_expansion_group.add_argument(
        "--no-expand-queries",
        dest="expand_queries",
        action="store_false",
        help="Disable deterministic query expansion and search only the provided queries",
    )
    run_parser.add_argument(
        "--manual-input",
        type=Path,
        help="Path to manual or Playwright-collected JSON input to import instead of calling the Reddit API",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional explicit run directory",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing run_report.json in --output-dir and skip matching completed stages",
    )
    run_parser.add_argument(
        "--comment-max-posts",
        type=int,
        default=5,
        help="Maximum number of top candidate posts to enrich with comments",
    )
    run_parser.add_argument(
        "--comment-limit",
        type=int,
        default=20,
        help="Per-submission comment listing limit",
    )
    run_parser.add_argument(
        "--comment-depth",
        type=int,
        default=3,
        help="Maximum nested reply depth to request",
    )
    run_parser.add_argument(
        "--comment-sort",
        default="best",
        help="Comment sort to request from Reddit",
    )
    run_parser.add_argument(
        "--max-morechildren-requests",
        type=int,
        default=3,
        help="Maximum number of bounded /api/morechildren requests to make per run",
    )
    run_parser.add_argument(
        "--morechildren-batch-size",
        type=int,
        default=20,
        help="Maximum number of MoreComments child IDs to request per /api/morechildren call",
    )
    run_parser.add_argument(
        "--max-selected-posts",
        type=int,
        default=10,
        help="Maximum number of ranked posts to persist into selected_posts.json",
    )
    run_parser.add_argument(
        "--min-nontrivial-comments",
        type=int,
        default=0,
        dest="min_non_trivial_comments",
        help="Minimum saved non-trivial comment count required for a post to survive into ranking",
    )
    run_parser.add_argument(
        "--min-complaint-signal-comments",
        type=int,
        default=0,
        help="Minimum saved complaint-signal comment count required for a post to survive into ranking",
    )
    run_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.22,
        help="Minimum token overlap similarity to merge posts into a cluster",
    )
    run_parser.add_argument(
        "--min-shared-terms",
        type=int,
        default=2,
        help="Minimum count of shared terms to merge posts into a cluster",
    )
    run_parser.add_argument(
        "--min-cluster-complaint-posts",
        type=int,
        default=2,
        help="Minimum strongest-cluster post count with complaint-signal comment evidence required before synthesis proceeds",
    )
    run_parser.add_argument(
        "--model",
        help="Optional model override. Otherwise LLM_MODEL is used.",
    )
    run_parser.add_argument(
        "--summary-max-posts",
        type=int,
        default=10,
        help="Maximum number of posts to include in the evidence summary prompt",
    )
    run_parser.add_argument(
        "--memo-max-posts",
        type=int,
        default=8,
        help="Maximum number of strongest-cluster posts to include in the final memo prompt",
    )
    run_parser.add_argument(
        "--min-cluster-posts",
        type=int,
        default=5,
        help="Minimum strongest-cluster size required before synthesis proceeds",
    )

    manual_import_parser = subparsers.add_parser(
        "manual-import",
        help="Import manual or Playwright-collected Reddit data into a run directory",
    )
    manual_import_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a JSON file containing manually collected posts and optional comments",
    )
    manual_import_parser.add_argument(
        "--subreddit",
        action="append",
        dest="subreddits",
        required=True,
        help="Target subreddit name without the r/ prefix",
    )
    manual_import_parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        required=True,
        help="Seed query metadata to attach to imported posts",
    )
    manual_import_parser.add_argument("--sort", default=DEFAULT_SORT, help="Source search sort metadata")
    manual_import_parser.add_argument(
        "--time-filter",
        default=DEFAULT_TIME_FILTER,
        help="Source time filter metadata",
    )
    manual_import_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Source search limit metadata",
    )
    manual_import_parser.add_argument(
        "--min-score",
        type=int,
        default=0,
        help="Minimum Reddit post score required for an imported candidate to survive filtering",
    )
    manual_import_parser.add_argument(
        "--min-comments",
        type=int,
        default=0,
        help="Minimum Reddit comment count required for an imported candidate to survive filtering",
    )
    manual_import_parser.add_argument(
        "--filter-nsfw",
        action="store_true",
        help="Exclude NSFW imported submissions during filtering",
    )
    manual_import_parser.add_argument(
        "--allow-subreddit",
        action="append",
        default=[],
        dest="allowed_subreddits",
        help="Optional subreddit allowlist applied after manual import normalization",
    )
    manual_import_parser.add_argument(
        "--deny-subreddit",
        action="append",
        default=[],
        dest="denied_subreddits",
        help="Optional subreddit denylist applied after manual import normalization",
    )
    manual_import_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional explicit run directory",
    )

    comments_parser = subparsers.add_parser(
        "comments",
        help="Fetch comment evidence for top candidate posts in an existing run",
    )
    comments_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a run directory containing candidate_posts.json",
    )
    comments_parser.add_argument(
        "--max-posts",
        type=int,
        default=5,
        help="Maximum number of top candidate posts to enrich with comments",
    )
    comments_parser.add_argument(
        "--comment-limit",
        type=int,
        default=20,
        help="Per-submission comment listing limit",
    )
    comments_parser.add_argument(
        "--comment-depth",
        type=int,
        default=3,
        help="Maximum nested reply depth to request",
    )
    comments_parser.add_argument(
        "--comment-sort",
        default="best",
        help="Comment sort to request from Reddit",
    )
    comments_parser.add_argument(
        "--max-morechildren-requests",
        type=int,
        default=3,
        help="Maximum number of bounded /api/morechildren requests to make per run",
    )
    comments_parser.add_argument(
        "--morechildren-batch-size",
        type=int,
        default=20,
        help="Maximum number of MoreComments child IDs to request per /api/morechildren call",
    )

    rank_parser = subparsers.add_parser(
        "rank",
        help="Rank candidate posts deterministically and persist a selected shortlist",
    )
    rank_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a run directory containing candidate_posts.json",
    )
    rank_parser.add_argument(
        "--max-selected-posts",
        type=int,
        default=10,
        help="Maximum number of ranked posts to persist into selected_posts.json",
    )
    rank_parser.add_argument(
        "--min-nontrivial-comments",
        type=int,
        default=0,
        dest="min_non_trivial_comments",
        help="Minimum saved non-trivial comment count required for a post to survive into ranking",
    )
    rank_parser.add_argument(
        "--min-complaint-signal-comments",
        type=int,
        default=0,
        help="Minimum saved complaint-signal comment count required for a post to survive into ranking",
    )

    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Cluster the ranked shortlist into repeated pain themes",
    )
    cluster_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a run directory containing selected_posts.json or candidate_posts.json",
    )
    cluster_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.22,
        help="Minimum token overlap similarity to merge posts into a cluster",
    )
    cluster_parser.add_argument(
        "--min-shared-terms",
        type=int,
        default=2,
        help="Minimum count of shared terms to merge posts into a cluster",
    )
    cluster_parser.add_argument(
        "--min-cluster-complaint-posts",
        type=int,
        default=0,
        help="Minimum strongest-cluster post count with complaint-signal comment evidence to record in cluster validation",
    )

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize an existing run's candidate posts with the configured LLM provider",
    )
    summarize_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a run directory containing candidate_posts.json",
    )
    summarize_parser.add_argument(
        "--model",
        help="Optional model override. Otherwise LLM_MODEL is used.",
    )
    summarize_parser.add_argument(
        "--max-posts",
        type=int,
        default=10,
        help="Maximum number of candidate posts to include in the synthesis prompt",
    )

    memo_parser = subparsers.add_parser(
        "memo",
        help="Write a grounded final memo from the strongest saved theme cluster",
    )
    memo_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a run directory containing theme_summary.json and evidence_summary.json",
    )
    memo_parser.add_argument(
        "--model",
        help="Optional model override. Otherwise LLM_MODEL is used.",
    )
    memo_parser.add_argument(
        "--min-cluster-posts",
        type=int,
        default=5,
        help="Minimum strongest-cluster size required before memo generation proceeds",
    )
    memo_parser.add_argument(
        "--max-posts",
        type=int,
        default=8,
        help="Maximum number of strongest-cluster posts to include in the memo prompt",
    )

    reply_parser = subparsers.add_parser(
        "reply-drafts",
        help="Draft Reddit-friendly reply suggestions for manual review from the top saved run posts",
    )
    reply_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a run directory containing selected_posts.json or candidate_posts.json",
    )
    reply_parser.add_argument(
        "--voice",
        required=True,
        help="Short description of the user's writing voice to use for the reply drafts",
    )
    reply_parser.add_argument(
        "--model",
        help="Optional model override for reply draft generation",
    )
    reply_parser.add_argument(
        "--max-posts",
        type=int,
        default=3,
        help="Maximum number of posts to draft replies for",
    )
    reply_parser.add_argument(
        "--score-threshold",
        type=float,
        default=4.0,
        help="Average evaluation score required for a reply draft to pass the improvement loop",
    )
    reply_parser.add_argument(
        "--minimum-dimension-score",
        type=float,
        default=3.0,
        help="Minimum per-dimension evaluation score required for a reply draft to pass",
    )
    reply_parser.add_argument(
        "--max-improvement-rounds",
        type=int,
        default=3,
        help="Maximum number of revision rounds after the initial draft",
    )

    llm_parser = subparsers.add_parser("llm", help="Interact with the configured LLM provider")
    llm_subparsers = llm_parser.add_subparsers(dest="llm_command", required=True)

    llm_subparsers.add_parser("models", help="List available models from the configured provider")

    llm_prompt_parser = llm_subparsers.add_parser("prompt", help="Run a one-shot prompt")
    llm_prompt_parser.add_argument("--prompt", required=True, help="Prompt text to send")
    llm_prompt_parser.add_argument("--model", help="Optional model override")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "layout":
        paths = ensure_repo_layout()
        for name, path in paths.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "init-run":
        _, created = init_run(args.run_name)
        for name, value in created.items():
            print(f"{name}: {value}")
        return 0

    if args.command == "search":
        try:
            config = load_runtime_config()
            result = asyncio.run(
                run_search_command(
                    config=config,
                    subreddits=args.subreddits,
                    queries=args.queries,
                    sort=args.sort,
                    time_filter=args.time_filter,
                    additional_sorts=args.additional_sorts,
                    additional_time_filters=args.additional_time_filters,
                    limit=args.limit,
                    min_score=args.min_score,
                    min_comments=args.min_comments,
                    filter_nsfw=args.filter_nsfw,
                    allowed_subreddits=args.allowed_subreddits,
                    denied_subreddits=args.denied_subreddits,
                    pages_per_query=args.pages_per_query,
                    expand_queries=args.expand_queries,
                    output_dir=args.output_dir,
                )
            )
        except (ConfigurationError, ValueError) as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Search failed: {exc}", file=sys.stderr)
            return 1

        print(f"run_slug: {result.run_slug}")
        print(f"output_dir: {result.run_dir}")
        print(f"query_variants: {result.query_variant_count}")
        print(f"search_specs: {result.search_spec_count}")
        print(f"sorts: {result.sort_count}")
        print(f"time_filters: {result.time_filter_count}")
        print(f"pages_per_query: {result.pages_per_query}")
        print(f"requests: {result.request_count}")
        print(f"candidate_posts: {result.candidate_count}")
        if result.filtered_counts:
            filtered_summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(result.filtered_counts.items())
            )
            print(f"filtered: {filtered_summary}")
        else:
            print("filtered: none")
        return 0

    if args.command == "capture":
        try:
            if args.max_posts <= 0:
                raise ValueError("--max-posts must be greater than 0")
            if args.max_comments < 0:
                raise ValueError("--max-comments must be 0 or greater")
            if args.page_timeout_seconds <= 0:
                raise ValueError("--page-timeout-seconds must be greater than 0")
            if args.page_wait_ms < 0:
                raise ValueError("--page-wait-ms must be 0 or greater")
            if args.min_score < 0:
                raise ValueError("--min-score must be 0 or greater")
            if args.min_comments < 0:
                raise ValueError("--min-comments must be 0 or greater")
            if args.limit <= 0:
                raise ValueError("--limit must be greater than 0")
            if args.skip_search and not args.thread_urls:
                raise ValueError("--skip-search requires at least one --thread-url")

            output_json = args.output_json
            if output_json is None and args.output_dir is not None:
                output_json = args.output_dir / "manual_capture.json"

            result = asyncio.run(
                capture_reddit_threads(
                    subreddits=args.subreddits,
                    queries=args.queries,
                    sort=args.sort,
                    time_filter=args.time_filter,
                    thread_urls=args.thread_urls,
                    select_results=args.select_results,
                    max_posts=args.max_posts,
                    max_comments=args.max_comments,
                    headless=not args.show_browser,
                    page_timeout_seconds=args.page_timeout_seconds,
                    page_wait_ms=args.page_wait_ms,
                    output_path=output_json,
                    skip_search=args.skip_search,
                )
            )
        except (ValueError, RuntimeError) as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Capture failed: {exc}", file=sys.stderr)
            return 1

        print(f"capture_json: {result.output_path}")
        print(f"capture_log: {result.log_path}")
        print(f"capture_snapshots: {result.snapshot_dir}")
        print(f"search_urls: {result.search_url_count}")
        print(f"discovered_threads: {result.discovered_thread_count}")
        print(f"selected_threads: {result.selected_thread_count}")
        print(f"captured_posts: {result.captured_post_count}")
        print(f"captured_comments: {result.captured_comment_count}")
        print(f"html_snapshots: {result.html_snapshot_count}")
        print(f"screenshots: {result.screenshot_count}")
        print(f"page_errors: {result.page_error_count}")
        if result.selected_thread_urls:
            print(f"selected_thread_urls: {', '.join(result.selected_thread_urls)}")
        else:
            print("selected_thread_urls: none")
        if args.handoff == "none":
            return 0
        handoff_argv = _build_capture_handoff_argv(args, result.output_path)
        print(f"handoff: {args.handoff}")
        print(f"handoff_command: {' '.join(handoff_argv)}")
        return _run_handoff_command(handoff_argv)

    if args.command == "manual-import":
        try:
            if args.min_score < 0:
                raise ValueError("--min-score must be 0 or greater")
            if args.min_comments < 0:
                raise ValueError("--min-comments must be 0 or greater")
            if args.limit <= 0:
                raise ValueError("--limit must be greater than 0")
            output_root = args.output_dir.parent if args.output_dir is not None else DEFAULT_OUTPUT_ROOT
            result = import_manual_search_bundle(
                input_path=args.input,
                output_root=output_root,
                subreddits=args.subreddits,
                queries=args.queries,
                sort=args.sort,
                time_filter=args.time_filter,
                limit=args.limit,
                min_score=args.min_score,
                min_comments=args.min_comments,
                filter_nsfw=args.filter_nsfw,
                allowed_subreddits=args.allowed_subreddits,
                denied_subreddits=args.denied_subreddits,
                output_dir=args.output_dir,
            )
        except ValueError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Manual import failed: {exc}", file=sys.stderr)
            return 1

        print(f"run_slug: {result.search_result.run_slug}")
        print(f"output_dir: {result.search_result.run_dir}")
        print(f"manual_input: {args.input}")
        print(f"imported_submissions: {result.imported_submission_count}")
        print(f"candidate_posts: {result.search_result.candidate_count}")
        print(f"saved_comment_submissions: {result.commented_submission_count}")
        print(f"saved_comments: {result.comments_result.comment_count}")
        if result.search_result.filtered_counts:
            filtered_summary = ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(result.search_result.filtered_counts.items())
            )
            print(f"filtered: {filtered_summary}")
        else:
            print("filtered: none")
        print(f"candidate_posts_json: {result.search_result.run_dir / 'candidate_posts.json'}")
        print(f"comment_enrichment_json: {result.search_result.run_dir / 'comment_enrichment.json'}")
        print(f"raw_manual_artifacts: {', '.join(result.raw_manual_artifacts)}")
        return 0

    if args.command == "run":
        stage_reports: list[RunStageReport] = []
        run_started_at = datetime.now(UTC)
        search_result = None
        comments_result = None
        ranking_result = None
        clustering_result = None
        summary_artifact = None
        memo_artifact = None
        run_report_path: Path | None = None
        current_stage: str | None = None
        current_stage_started: float | None = None
        resumed_from_stage: str | None = None
        manual_import_result = None
        provider: str | None = None
        model: str | None = None
        stage_params = _stage_params_from_args(args)
        try:
            if args.comment_max_posts <= 0:
                raise ValueError("--comment-max-posts must be greater than 0")
            if args.comment_limit <= 0:
                raise ValueError("--comment-limit must be greater than 0")
            if args.comment_depth <= 0:
                raise ValueError("--comment-depth must be greater than 0")
            if args.max_morechildren_requests < 0:
                raise ValueError("--max-morechildren-requests must be 0 or greater")
            if args.morechildren_batch_size <= 0:
                raise ValueError("--morechildren-batch-size must be greater than 0")
            if args.min_score < 0:
                raise ValueError("--min-score must be 0 or greater")
            if args.min_comments < 0:
                raise ValueError("--min-comments must be 0 or greater")
            if args.pages_per_query <= 0:
                raise ValueError("--pages-per-query must be greater than 0")
            if args.max_selected_posts <= 0:
                raise ValueError("--max-selected-posts must be greater than 0")
            if args.min_non_trivial_comments < 0:
                raise ValueError("--min-nontrivial-comments must be 0 or greater")
            if args.min_complaint_signal_comments < 0:
                raise ValueError("--min-complaint-signal-comments must be 0 or greater")
            if args.min_cluster_complaint_posts < 0:
                raise ValueError("--min-cluster-complaint-posts must be 0 or greater")
            if args.summary_max_posts <= 0:
                raise ValueError("--summary-max-posts must be greater than 0")
            if args.memo_max_posts <= 0:
                raise ValueError("--memo-max-posts must be greater than 0")
            if args.min_cluster_posts <= 0:
                raise ValueError("--min-cluster-posts must be greater than 0")
            if args.resume and args.output_dir is None:
                raise ValueError("--resume requires --output-dir")

            runtime_config = None
            if args.manual_input is None:
                runtime_config = load_runtime_config()
            search_sorts, search_time_filters = resolve_search_plan(
                sort=args.sort,
                time_filter=args.time_filter,
                additional_sorts=args.additional_sorts,
                additional_time_filters=args.additional_time_filters,
            )
            if args.resume:
                previous_report, stage_reports, resumed_from_stage = _resolve_resume_state(
                    args.output_dir,
                    args,
                )
                run_started_at = previous_report.started_at
                if previous_report.subreddits != list(args.subreddits):
                    raise ValueError("resume subreddits do not match the existing run report")
                if previous_report.queries != list(args.queries):
                    raise ValueError("resume queries do not match the existing run report")
                if previous_report.sort != args.sort:
                    raise ValueError("resume sort does not match the existing run report")
                if previous_report.time_filter != args.time_filter:
                    raise ValueError("resume time filter does not match the existing run report")
                if previous_report.limit != args.limit:
                    raise ValueError("resume limit does not match the existing run report")
                search_stage_report = next(
                    (item for item in stage_reports if item.stage == "search"),
                    None,
                )
                if search_stage_report is not None:
                    search_result = type(
                        "SearchResult",
                        (),
                        {
                            "run_slug": previous_report.run_slug,
                            "run_dir": Path(previous_report.run_dir),
                            "request_count": int(search_stage_report.details.get("request_count", 0)),
                            "candidate_count": int(search_stage_report.details.get("candidate_count", 0)),
                            "query_variant_count": int(search_stage_report.details.get("query_variant_count", 0)),
                            "search_spec_count": int(search_stage_report.details.get("search_spec_count", 0)),
                            "sort_count": int(search_stage_report.details.get("sort_count", 0)),
                            "time_filter_count": int(search_stage_report.details.get("time_filter_count", 0)),
                            "pages_per_query": int(search_stage_report.details.get("pages_per_query", 0)),
                            "filtered_counts": dict(search_stage_report.details.get("filtered_counts", {})),
                        },
                    )()
                if resumed_from_stage is None and search_result is not None:
                    print(f"run_slug: {search_result.run_slug}")
                    print(f"run_dir: {search_result.run_dir}")
                    print("status: completed")
                    print("resume_status: no_work")
                    print(f"query_variants: {search_result.query_variant_count}")
                    print(f"search_specs: {search_result.search_spec_count}")
                    print(f"sorts: {search_result.sort_count}")
                    print(f"time_filters: {search_result.time_filter_count}")
                    print(f"pages_per_query: {search_result.pages_per_query}")
                    if search_result.filtered_counts:
                        filtered_summary = ", ".join(
                            f"{reason}={count}"
                            for reason, count in sorted(search_result.filtered_counts.items())
                        )
                        print(f"filtered: {filtered_summary}")
                    else:
                        print("filtered: none")
                    print(f"run_report_json: {search_result.run_dir / 'run_report.json'}")
                    return 0

            current_stage = "search"
            current_stage_started = perf_counter()
            if search_result is None or resumed_from_stage == "search":
                if args.manual_input is not None:
                    output_root = (
                        args.output_dir.parent if args.output_dir is not None else DEFAULT_OUTPUT_ROOT
                    )
                    manual_import_result = import_manual_search_bundle(
                        input_path=args.manual_input,
                        output_root=output_root,
                        subreddits=args.subreddits,
                        queries=args.queries,
                        sort=args.sort,
                        time_filter=args.time_filter,
                        limit=args.limit,
                        min_score=args.min_score,
                        min_comments=args.min_comments,
                        filter_nsfw=args.filter_nsfw,
                        allowed_subreddits=args.allowed_subreddits,
                        denied_subreddits=args.denied_subreddits,
                        output_dir=args.output_dir,
                    )
                    search_result = manual_import_result.search_result
                    comments_result = manual_import_result.comments_result
                else:
                    search_result = asyncio.run(
                        run_search_command(
                            config=runtime_config,
                            subreddits=args.subreddits,
                            queries=args.queries,
                            sort=args.sort,
                            time_filter=args.time_filter,
                            additional_sorts=args.additional_sorts,
                            additional_time_filters=args.additional_time_filters,
                            limit=args.limit,
                            min_score=args.min_score,
                            min_comments=args.min_comments,
                            filter_nsfw=args.filter_nsfw,
                            allowed_subreddits=args.allowed_subreddits,
                            denied_subreddits=args.denied_subreddits,
                            pages_per_query=args.pages_per_query,
                            expand_queries=args.expand_queries,
                            output_dir=args.output_dir,
                        )
                    )
                stage_reports = [item for item in stage_reports if item.stage != "search"]
                stage_reports.append(
                    RunStageReport(
                        stage="search",
                        status="completed",
                        duration_ms=_elapsed_ms(current_stage_started),
                        details={
                            "params": stage_params["search"],
                            "request_count": search_result.request_count,
                            "candidate_count": search_result.candidate_count,
                            "query_variant_count": search_result.query_variant_count,
                            "search_spec_count": search_result.search_spec_count,
                            "sort_count": search_result.sort_count,
                            "time_filter_count": search_result.time_filter_count,
                            "search_sorts": search_sorts,
                            "search_time_filters": search_time_filters,
                            "pages_per_query": search_result.pages_per_query,
                            "filtered_counts": search_result.filtered_counts,
                            "search_mode": "manual" if args.manual_input else "api",
                            "manual_input_path": str(args.manual_input) if args.manual_input else None,
                            "raw_manual_artifacts": (
                                manual_import_result.raw_manual_artifacts
                                if manual_import_result is not None
                                else []
                            ),
                            "imported_submission_count": (
                                manual_import_result.imported_submission_count
                                if manual_import_result is not None
                                else None
                            ),
                            "run_dir": str(search_result.run_dir),
                        },
                        artifact_fingerprints=_stage_artifact_fingerprints(
                            search_result.run_dir,
                            "search",
                        ),
                    )
            )
            current_stage = "comments"
            current_stage_started = perf_counter()
            if _should_run_stage(resumed_from_stage, "comments"):
                if args.manual_input is not None and comments_result is None:
                    output_root = (
                        args.output_dir.parent if args.output_dir is not None else DEFAULT_OUTPUT_ROOT
                    )
                    manual_import_result = import_manual_search_bundle(
                        input_path=args.manual_input,
                        output_root=output_root,
                        subreddits=args.subreddits,
                        queries=args.queries,
                        sort=args.sort,
                        time_filter=args.time_filter,
                        limit=args.limit,
                        min_score=args.min_score,
                        min_comments=args.min_comments,
                        filter_nsfw=args.filter_nsfw,
                        allowed_subreddits=args.allowed_subreddits,
                        denied_subreddits=args.denied_subreddits,
                        output_dir=search_result.run_dir,
                    )
                    comments_result = manual_import_result.comments_result
                elif args.manual_input is None:
                    comments_result = asyncio.run(
                        enrich_run_with_comments(
                            config=runtime_config,
                            run_dir=search_result.run_dir,
                            max_posts=args.comment_max_posts,
                            comment_limit=args.comment_limit,
                            comment_depth=args.comment_depth,
                            comment_sort=args.comment_sort,
                            max_morechildren_requests=args.max_morechildren_requests,
                            morechildren_batch_size=args.morechildren_batch_size,
                        )
                    )
                stage_reports = [item for item in stage_reports if item.stage != "comments"]
                stage_reports.append(
                    RunStageReport(
                        stage="comments",
                        status="completed",
                        duration_ms=_elapsed_ms(current_stage_started),
                        details={
                            "params": stage_params["comments"],
                            "requested_submission_count": comments_result.requested_submission_count,
                            "fetched_submission_count": comments_result.fetched_submission_count,
                            "comment_count": comments_result.comment_count,
                            "morechildren_request_count": comments_result.morechildren_request_count,
                            "source": "manual_import" if args.manual_input else "reddit_api",
                        },
                        artifact_fingerprints=_stage_artifact_fingerprints(
                            search_result.run_dir,
                            "comments",
                        ),
                    )
                )
            current_stage = "rank"
            current_stage_started = perf_counter()
            if _should_run_stage(resumed_from_stage, "rank"):
                ranking_result = rank_run_candidates(
                    run_dir=search_result.run_dir,
                    max_selected_posts=args.max_selected_posts,
                    min_non_trivial_comments=args.min_non_trivial_comments,
                    min_complaint_signal_comments=args.min_complaint_signal_comments,
                )
                stage_reports = [item for item in stage_reports if item.stage != "rank"]
                stage_reports.append(
                    RunStageReport(
                        stage="rank",
                        status="completed",
                        duration_ms=_elapsed_ms(current_stage_started),
                        details={
                            "params": stage_params["rank"],
                            "candidate_count": ranking_result.candidate_count,
                            "screened_candidate_count": ranking_result.screened_candidate_count,
                            "rejected_candidate_count": ranking_result.rejected_candidate_count,
                            "selected_count": ranking_result.selected_count,
                            "rejection_counts": ranking_result.rejection_counts,
                        },
                        artifact_fingerprints=_stage_artifact_fingerprints(
                            search_result.run_dir,
                            "rank",
                        ),
                    )
                )
            current_stage = "cluster"
            current_stage_started = perf_counter()
            if _should_run_stage(resumed_from_stage, "cluster"):
                clustering_result = cluster_run_posts(
                    run_dir=search_result.run_dir,
                    similarity_threshold=args.similarity_threshold,
                    min_shared_terms=args.min_shared_terms,
                    min_cluster_complaint_posts=args.min_cluster_complaint_posts,
                )
                strongest_cluster_posts = len(clustering_result.strongest_post_ids)
                stage_reports = [item for item in stage_reports if item.stage != "cluster"]
                stage_reports.append(
                    RunStageReport(
                        stage="cluster",
                        status="completed",
                        duration_ms=_elapsed_ms(current_stage_started),
                        details={
                            "params": stage_params["cluster"],
                            "cluster_count": clustering_result.cluster_count,
                            "strongest_cluster_id": clustering_result.strongest_cluster_id,
                            "strongest_cluster_post_count": strongest_cluster_posts,
                            "strongest_cluster_screened_post_count": clustering_result.strongest_cluster_screened_post_count,
                            "strongest_cluster_complaint_signal_post_count": clustering_result.strongest_cluster_complaint_signal_post_count,
                            "evidence_validation_passed": clustering_result.evidence_validation_passed,
                            "evidence_failure_reason": clustering_result.evidence_failure_reason,
                        },
                        artifact_fingerprints=_stage_artifact_fingerprints(
                            search_result.run_dir,
                            "cluster",
                        ),
                    )
                )
            else:
                cluster_stage_report = next(
                    item for item in stage_reports if item.stage == "cluster"
                )
                strongest_cluster_posts = int(
                    cluster_stage_report.details.get("strongest_cluster_post_count", 0)
                )
                clustering_result = type(
                    "ClusterResult",
                    (),
                    {
                        "cluster_count": int(cluster_stage_report.details.get("cluster_count", 0)),
                        "strongest_cluster_id": cluster_stage_report.details.get("strongest_cluster_id"),
                        "strongest_post_ids": [None] * strongest_cluster_posts,
                        "strongest_cluster_screened_post_count": int(
                            cluster_stage_report.details.get("strongest_cluster_screened_post_count", 0)
                        ),
                        "strongest_cluster_complaint_signal_post_count": int(
                            cluster_stage_report.details.get("strongest_cluster_complaint_signal_post_count", 0)
                        ),
                        "evidence_validation_passed": bool(
                            cluster_stage_report.details.get("evidence_validation_passed", False)
                        ),
                        "evidence_failure_reason": cluster_stage_report.details.get("evidence_failure_reason"),
                    },
                )()
            if strongest_cluster_posts < args.min_cluster_posts:
                stage_reports = [item for item in stage_reports if item.stage not in {"summarize", "memo"}]
                stage_reports.append(
                    RunStageReport(
                        stage="summarize",
                        status="skipped",
                        details={"params": stage_params["summarize"], "reason": "strongest_cluster_too_weak"},
                    )
                )
                stage_reports.append(
                    RunStageReport(
                        stage="memo",
                        status="stopped",
                        details={"params": stage_params["memo"], "reason": "strongest_cluster_too_weak"},
                    )
                )
                run_report_path = _write_run_report(
                    run_slug=search_result.run_slug,
                    run_dir=search_result.run_dir,
                    status="stopped",
                    started_at=run_started_at,
                    subreddits=args.subreddits,
                    queries=args.queries,
                    sort=args.sort,
                    time_filter=args.time_filter,
                    limit=args.limit,
                    stage_reports=stage_reports,
                    stop_reason="strongest_cluster_too_weak",
                )
                print(f"run_dir: {search_result.run_dir}")
                print("status: stopped")
                print("stop_reason: strongest_cluster_too_weak")
                if args.resume:
                    print(f"resume_from_stage: {resumed_from_stage}")
                print(
                    "query_variants: "
                    f"{next((item.details.get('query_variant_count', 0) for item in stage_reports if item.stage == 'search'), 0)}"
                )
                print(
                    "search_specs: "
                    f"{next((item.details.get('search_spec_count', 0) for item in stage_reports if item.stage == 'search'), 0)}"
                )
                print(
                    "sorts: "
                    f"{next((item.details.get('sort_count', 0) for item in stage_reports if item.stage == 'search'), 0)}"
                )
                print(
                    "time_filters: "
                    f"{next((item.details.get('time_filter_count', 0) for item in stage_reports if item.stage == 'search'), 0)}"
                )
                print(
                    "pages_per_query: "
                    f"{next((item.details.get('pages_per_query', 0) for item in stage_reports if item.stage == 'search'), 0)}"
                )
                print(f"requests: {search_result.request_count}")
                print(f"candidate_posts: {search_result.candidate_count}")
                if search_result.filtered_counts:
                    filtered_summary = ", ".join(
                        f"{reason}={count}" for reason, count in sorted(search_result.filtered_counts.items())
                    )
                    print(f"filtered: {filtered_summary}")
                else:
                    print("filtered: none")
                print(
                    "saved_comments: "
                    f"{next((item.details.get('comment_count', 0) for item in stage_reports if item.stage == 'comments'), 0)}"
                )
                print(
                    "rank_survivors: "
                    f"{next((item.details.get('screened_candidate_count', 0) for item in stage_reports if item.stage == 'rank'), 0)}"
                )
                print(
                    "rank_rejected: "
                    f"{next((item.details.get('rejected_candidate_count', 0) for item in stage_reports if item.stage == 'rank'), 0)}"
                )
                rank_rejections = next(
                    (item.details.get("rejection_counts", {}) for item in stage_reports if item.stage == "rank"),
                    {},
                )
                if rank_rejections:
                    rank_rejection_summary = ", ".join(
                        f"{reason}={count}" for reason, count in sorted(rank_rejections.items())
                    )
                    print(f"rank_filtered: {rank_rejection_summary}")
                else:
                    print("rank_filtered: none")
                print(
                    "selected_posts: "
                    f"{next((item.details.get('selected_count', 0) for item in stage_reports if item.stage == 'rank'), 0)}"
                )
                print(f"clusters: {clustering_result.cluster_count}")
                print(
                    f"strongest_cluster_id: {clustering_result.strongest_cluster_id or 'none'}"
                )
                print(f"strongest_cluster_posts: {strongest_cluster_posts}")
                print(f"required_cluster_posts: {args.min_cluster_posts}")
                print(
                    "cluster_complaint_posts: "
                    f"{getattr(clustering_result, 'strongest_cluster_complaint_signal_post_count', 0)}"
                )
                print(f"theme_summary_json: {search_result.run_dir / 'theme_summary.json'}")
                print(
                    f"cluster_evidence_validation_json: {search_result.run_dir / 'cluster_evidence_validation.json'}"
                )
                print(f"run_report_json: {run_report_path}")
                return 2
            if not getattr(clustering_result, "evidence_validation_passed", True):
                stage_reports = [item for item in stage_reports if item.stage not in {"summarize", "memo"}]
                stage_reports.append(
                    RunStageReport(
                        stage="summarize",
                        status="skipped",
                        details={
                            "params": stage_params["summarize"],
                            "reason": "strongest_cluster_evidence_too_weak",
                        },
                    )
                )
                stage_reports.append(
                    RunStageReport(
                        stage="memo",
                        status="stopped",
                        details={
                            "params": stage_params["memo"],
                            "reason": "strongest_cluster_evidence_too_weak",
                        },
                    )
                )
                run_report_path = _write_run_report(
                    run_slug=search_result.run_slug,
                    run_dir=search_result.run_dir,
                    status="stopped",
                    started_at=run_started_at,
                    subreddits=args.subreddits,
                    queries=args.queries,
                    sort=args.sort,
                    time_filter=args.time_filter,
                    limit=args.limit,
                    stage_reports=stage_reports,
                    stop_reason="strongest_cluster_evidence_too_weak",
                )
                print(f"run_dir: {search_result.run_dir}")
                print("status: stopped")
                print("stop_reason: strongest_cluster_evidence_too_weak")
                if args.resume:
                    print(f"resume_from_stage: {resumed_from_stage}")
                print(
                    "query_variants: "
                    f"{next((item.details.get('query_variant_count', 0) for item in stage_reports if item.stage == 'search'), 0)}"
                )
                print(
                    "search_specs: "
                    f"{next((item.details.get('search_spec_count', 0) for item in stage_reports if item.stage == 'search'), 0)}"
                )
                print(
                    "sorts: "
                    f"{next((item.details.get('sort_count', 0) for item in stage_reports if item.stage == 'search'), 0)}"
                )
                print(
                    "time_filters: "
                    f"{next((item.details.get('time_filter_count', 0) for item in stage_reports if item.stage == 'search'), 0)}"
                )
                print(
                    "pages_per_query: "
                    f"{next((item.details.get('pages_per_query', 0) for item in stage_reports if item.stage == 'search'), 0)}"
                )
                print(f"requests: {search_result.request_count}")
                print(f"candidate_posts: {search_result.candidate_count}")
                if search_result.filtered_counts:
                    filtered_summary = ", ".join(
                        f"{reason}={count}" for reason, count in sorted(search_result.filtered_counts.items())
                    )
                    print(f"filtered: {filtered_summary}")
                else:
                    print("filtered: none")
                print(
                    "saved_comments: "
                    f"{next((item.details.get('comment_count', 0) for item in stage_reports if item.stage == 'comments'), 0)}"
                )
                print(
                    "rank_survivors: "
                    f"{next((item.details.get('screened_candidate_count', 0) for item in stage_reports if item.stage == 'rank'), 0)}"
                )
                print(
                    "selected_posts: "
                    f"{next((item.details.get('selected_count', 0) for item in stage_reports if item.stage == 'rank'), 0)}"
                )
                print(f"clusters: {clustering_result.cluster_count}")
                print(
                    f"strongest_cluster_id: {clustering_result.strongest_cluster_id or 'none'}"
                )
                print(f"strongest_cluster_posts: {strongest_cluster_posts}")
                print(
                    "cluster_complaint_posts: "
                    f"{clustering_result.strongest_cluster_complaint_signal_post_count}"
                )
                print(
                    f"required_cluster_complaint_posts: {args.min_cluster_complaint_posts}"
                )
                print(
                    f"evidence_failure_reason: {clustering_result.evidence_failure_reason or 'unknown'}"
                )
                print(f"theme_summary_json: {search_result.run_dir / 'theme_summary.json'}")
                print(
                    f"cluster_evidence_validation_json: {search_result.run_dir / 'cluster_evidence_validation.json'}"
                )
                print(f"run_report_json: {run_report_path}")
                return 2

            llm_config = load_llm_config(require_model=not args.model)

            async def _run_summary_stage() -> object:
                async with LMStudioClient(llm_config) as client:
                    return await summarize_candidate_posts(
                        run_dir=search_result.run_dir,
                        client=client,
                        model=args.model,
                        max_posts=args.summary_max_posts,
                    )

            async def _run_memo_stage() -> object:
                async with LMStudioClient(llm_config) as client:
                    return await write_final_memo(
                        run_dir=search_result.run_dir,
                        client=client,
                        model=args.model,
                        min_cluster_posts=args.min_cluster_posts,
                        max_posts=args.memo_max_posts,
                    )

            current_stage = "summarize"
            current_stage_started = perf_counter()
            if _should_run_stage(resumed_from_stage, "summarize"):
                summary_artifact = asyncio.run(_run_summary_stage())
                summary_duration_ms = _elapsed_ms(current_stage_started)
                stage_reports = [item for item in stage_reports if item.stage != "summarize"]
                stage_reports.append(
                    RunStageReport(
                        stage="summarize",
                        status="completed",
                        duration_ms=summary_duration_ms,
                        details={
                            "params": stage_params["summarize"],
                            "candidate_count": summary_artifact.candidate_count,
                            "comment_count": summary_artifact.comment_count,
                            "selected_comment_count": summary_artifact.selected_comment_count,
                            "max_posts_used": summary_artifact.max_posts_used,
                        },
                        artifact_fingerprints=_stage_artifact_fingerprints(
                            search_result.run_dir,
                            "summarize",
                        ),
                    )
                )
            current_stage = "memo"
            current_stage_started = perf_counter()
            if _should_run_stage(resumed_from_stage, "memo"):
                memo_artifact = asyncio.run(_run_memo_stage())
                memo_duration_ms = _elapsed_ms(current_stage_started)
                provider = memo_artifact.provider
                model = memo_artifact.model
                stage_reports = [item for item in stage_reports if item.stage != "memo"]
                stage_reports.append(
                    RunStageReport(
                        stage="memo",
                        status="completed",
                        duration_ms=memo_duration_ms,
                        details={
                            "params": stage_params["memo"],
                            "strongest_cluster_id": memo_artifact.strongest_cluster_id,
                            "strongest_cluster_size": memo_artifact.strongest_cluster_size,
                            "included_post_count": memo_artifact.included_post_count,
                            "provider": memo_artifact.provider,
                            "model": memo_artifact.model,
                        },
                        artifact_fingerprints=_stage_artifact_fingerprints(
                            search_result.run_dir,
                            "memo",
                        ),
                    )
                )
            else:
                memo_stage_report = next(item for item in stage_reports if item.stage == "memo")
                provider = str(memo_stage_report.details.get("provider") or "") or None
                model = str(memo_stage_report.details.get("model") or "") or None
            run_report_path = _write_run_report(
                run_slug=search_result.run_slug,
                run_dir=search_result.run_dir,
                status="completed",
                started_at=run_started_at,
                subreddits=args.subreddits,
                queries=args.queries,
                sort=args.sort,
                time_filter=args.time_filter,
                limit=args.limit,
                stage_reports=stage_reports,
                provider=provider,
                model=model,
            )
        except (ConfigurationError, ValueError, FileNotFoundError) as exc:
            if search_result is not None:
                if current_stage is not None:
                    stage_reports = [item for item in stage_reports if item.stage != current_stage]
                    stage_reports.append(
                        RunStageReport(
                            stage=current_stage,
                            status="failed",
                            duration_ms=_elapsed_ms(current_stage_started)
                            if current_stage_started is not None
                            else None,
                            details={"params": stage_params.get(current_stage, {}), "error": str(exc)},
                        )
                    )
                _write_run_report(
                    run_slug=search_result.run_slug,
                    run_dir=search_result.run_dir,
                    status="failed",
                    started_at=run_started_at,
                    subreddits=args.subreddits,
                    queries=args.queries,
                    sort=args.sort,
                    time_filter=args.time_filter,
                    limit=args.limit,
                    stage_reports=stage_reports,
                    error=str(exc),
                )
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            if search_result is not None:
                if current_stage is not None:
                    stage_reports = [item for item in stage_reports if item.stage != current_stage]
                    stage_reports.append(
                        RunStageReport(
                            stage=current_stage,
                            status="failed",
                            duration_ms=_elapsed_ms(current_stage_started)
                            if current_stage_started is not None
                            else None,
                            details={"params": stage_params.get(current_stage, {}), "error": str(exc)},
                        )
                    )
                _write_run_report(
                    run_slug=search_result.run_slug,
                    run_dir=search_result.run_dir,
                    status="failed",
                    started_at=run_started_at,
                    subreddits=args.subreddits,
                    queries=args.queries,
                    sort=args.sort,
                    time_filter=args.time_filter,
                    limit=args.limit,
                    stage_reports=stage_reports,
                    error=str(exc),
                )
            print(f"Run failed: {exc}", file=sys.stderr)
            return 1

        print(f"run_slug: {search_result.run_slug}")
        print(f"run_dir: {search_result.run_dir}")
        print("status: completed")
        if args.resume:
            print(f"resume_from_stage: {resumed_from_stage}")
        print(f"provider: {provider or 'unknown'}")
        print(f"model: {model or 'unknown'}")
        print(
            "query_variants: "
            f"{next((item.details.get('query_variant_count', 0) for item in stage_reports if item.stage == 'search'), 0)}"
        )
        print(
            "search_specs: "
            f"{next((item.details.get('search_spec_count', 0) for item in stage_reports if item.stage == 'search'), 0)}"
        )
        print(
            "sorts: "
            f"{next((item.details.get('sort_count', 0) for item in stage_reports if item.stage == 'search'), 0)}"
        )
        print(
            "time_filters: "
            f"{next((item.details.get('time_filter_count', 0) for item in stage_reports if item.stage == 'search'), 0)}"
        )
        print(
            "pages_per_query: "
            f"{next((item.details.get('pages_per_query', 0) for item in stage_reports if item.stage == 'search'), 0)}"
        )
        print(
            "requests: "
            f"{next((item.details.get('request_count', 0) for item in stage_reports if item.stage == 'search'), 0)}"
        )
        print(
            "candidate_posts: "
            f"{next((item.details.get('candidate_count', 0) for item in stage_reports if item.stage == 'search'), 0)}"
        )
        if search_result.filtered_counts:
            filtered_summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(search_result.filtered_counts.items())
            )
            print(f"filtered: {filtered_summary}")
        else:
            print("filtered: none")
        print(
            "saved_comments: "
            f"{next((item.details.get('comment_count', 0) for item in stage_reports if item.stage == 'comments'), 0)}"
        )
        print(
            "rank_survivors: "
            f"{next((item.details.get('screened_candidate_count', 0) for item in stage_reports if item.stage == 'rank'), 0)}"
        )
        print(
            "rank_rejected: "
            f"{next((item.details.get('rejected_candidate_count', 0) for item in stage_reports if item.stage == 'rank'), 0)}"
        )
        rank_rejections = next(
            (item.details.get("rejection_counts", {}) for item in stage_reports if item.stage == "rank"),
            {},
        )
        if rank_rejections:
            rank_rejection_summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(rank_rejections.items())
            )
            print(f"rank_filtered: {rank_rejection_summary}")
        else:
            print("rank_filtered: none")
        print(
            "selected_posts: "
            f"{next((item.details.get('selected_count', 0) for item in stage_reports if item.stage == 'rank'), 0)}"
        )
        print(
            "clusters: "
            f"{next((item.details.get('cluster_count', 0) for item in stage_reports if item.stage == 'cluster'), 0)}"
        )
        print(
            "strongest_cluster_id: "
            f"{next((item.details.get('strongest_cluster_id') for item in stage_reports if item.stage == 'cluster'), None) or 'none'}"
        )
        print(
            "strongest_cluster_posts: "
            f"{next((item.details.get('strongest_cluster_post_count', 0) for item in stage_reports if item.stage == 'cluster'), 0)}"
        )
        print(
            "cluster_complaint_posts: "
            f"{next((item.details.get('strongest_cluster_complaint_signal_post_count', 0) for item in stage_reports if item.stage == 'cluster'), 0)}"
        )
        print(f"summary_json: {search_result.run_dir / 'evidence_summary.json'}")
        print(f"final_memo_markdown: {search_result.run_dir / 'final_memo.md'}")
        print(f"run_report_json: {run_report_path}")
        return 0

    if args.command == "comments":
        try:
            config = load_runtime_config()
            result = asyncio.run(
                enrich_run_with_comments(
                    config=config,
                    run_dir=args.run_dir,
                    max_posts=args.max_posts,
                    comment_limit=args.comment_limit,
                    comment_depth=args.comment_depth,
                    comment_sort=args.comment_sort,
                    max_morechildren_requests=args.max_morechildren_requests,
                    morechildren_batch_size=args.morechildren_batch_size,
                )
            )
        except (ConfigurationError, ValueError, FileNotFoundError) as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Comment enrichment failed: {exc}", file=sys.stderr)
            return 1

        print(f"run_dir: {result.run_dir}")
        print(f"requested_submissions: {result.requested_submission_count}")
        print(f"fetched_submissions: {result.fetched_submission_count}")
        print(f"saved_comments: {result.comment_count}")
        print(f"morechildren_requests: {result.morechildren_request_count}")
        print(f"comment_enrichment_json: {args.run_dir / 'comment_enrichment.json'}")
        return 0

    if args.command == "llm":
        try:
            config = load_llm_config(require_model=args.llm_command == "prompt" and not args.model)
            if config.provider != "lmstudio":
                raise ConfigurationError("Unsupported LLM provider")

            async def _run_llm_command() -> int:
                async with LMStudioClient(config) as client:
                    if args.llm_command == "models":
                        models = await client.list_models()
                        print(f"provider: {config.provider}")
                        print(f"base_url: {config.base_url}")
                        if not models:
                            print("models: none")
                            return 0
                        for model in models:
                            suffix = f" ({model.owned_by})" if model.owned_by else ""
                            print(f"- {model.id}{suffix}")
                        return 0

                    text = await client.generate_text(args.prompt, model=args.model)
                    print(f"provider: {config.provider}")
                    print(f"model: {args.model or config.model}")
                    print(text)
                    return 0

            return asyncio.run(_run_llm_command())
        except (ConfigurationError, ValueError) as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"LLM command failed: {exc}", file=sys.stderr)
            return 1

    if args.command == "rank":
        try:
            if args.min_non_trivial_comments < 0:
                raise ValueError("--min-nontrivial-comments must be 0 or greater")
            if args.min_complaint_signal_comments < 0:
                raise ValueError("--min-complaint-signal-comments must be 0 or greater")
            result = rank_run_candidates(
                run_dir=args.run_dir,
                max_selected_posts=args.max_selected_posts,
                min_non_trivial_comments=args.min_non_trivial_comments,
                min_complaint_signal_comments=args.min_complaint_signal_comments,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Ranking failed: {exc}", file=sys.stderr)
            return 1

        print(f"run_dir: {result.run_dir}")
        print(f"candidate_posts: {result.candidate_count}")
        print(f"rank_survivors: {result.screened_candidate_count}")
        print(f"rank_rejected: {result.rejected_candidate_count}")
        if result.rejection_counts:
            rank_rejection_summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(result.rejection_counts.items())
            )
            print(f"rank_filtered: {rank_rejection_summary}")
        else:
            print("rank_filtered: none")
        print(f"selected_posts: {result.selected_count}")
        print(f"candidate_screening_json: {args.run_dir / 'candidate_screening.json'}")
        print(f"post_ranking_json: {args.run_dir / 'post_ranking.json'}")
        print(f"selected_posts_json: {args.run_dir / 'selected_posts.json'}")
        return 0

    if args.command == "cluster":
        try:
            result = cluster_run_posts(
                run_dir=args.run_dir,
                similarity_threshold=args.similarity_threshold,
                min_shared_terms=args.min_shared_terms,
                min_cluster_complaint_posts=args.min_cluster_complaint_posts,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Clustering failed: {exc}", file=sys.stderr)
            return 1

        print(f"run_dir: {result.run_dir}")
        print(f"source_posts: {result.source_post_count}")
        print(f"clusters: {result.cluster_count}")
        print(f"strongest_cluster_id: {result.strongest_cluster_id or 'none'}")
        print(f"strongest_cluster_posts: {len(result.strongest_post_ids)}")
        print(f"cluster_complaint_posts: {result.strongest_cluster_complaint_signal_post_count}")
        print(f"cluster_evidence_valid: {'yes' if result.evidence_validation_passed else 'no'}")
        if not result.evidence_validation_passed:
            print(f"cluster_evidence_failure_reason: {result.evidence_failure_reason or 'unknown'}")
        print(f"theme_summary_json: {args.run_dir / 'theme_summary.json'}")
        print(
            f"cluster_evidence_validation_json: {args.run_dir / 'cluster_evidence_validation.json'}"
        )
        return 0

    if args.command == "summarize":
        try:
            if args.max_posts <= 0:
                raise ValueError("--max-posts must be greater than 0")
            config = load_llm_config(require_model=not args.model)

            async def _run_summary() -> int:
                async with LMStudioClient(config) as client:
                    artifact = await summarize_candidate_posts(
                        run_dir=args.run_dir,
                        client=client,
                        model=args.model,
                        max_posts=args.max_posts,
                    )
                print(f"run_dir: {artifact.run_dir}")
                print(f"provider: {artifact.provider}")
                print(f"model: {artifact.model}")
                print(f"candidate_posts: {artifact.candidate_count}")
                print(f"saved_comments: {artifact.comment_count}")
                print(f"selected_comments: {artifact.selected_comment_count}")
                print(f"max_posts_used: {artifact.max_posts_used}")
                print(f"comment_selection_json: {args.run_dir / 'comment_selection.json'}")
                print(f"summary_json: {args.run_dir / 'evidence_summary.json'}")
                print(f"summary_markdown: {args.run_dir / 'evidence_summary.md'}")
                return 0

            return asyncio.run(_run_summary())
        except (ConfigurationError, ValueError, FileNotFoundError) as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Summarize failed: {exc}", file=sys.stderr)
            return 1

    if args.command == "memo":
        try:
            if args.max_posts <= 0:
                raise ValueError("--max-posts must be greater than 0")
            if args.min_cluster_posts <= 0:
                raise ValueError("--min-cluster-posts must be greater than 0")
            config = load_llm_config(require_model=not args.model)

            async def _run_memo() -> int:
                async with LMStudioClient(config) as client:
                    artifact = await write_final_memo(
                        run_dir=args.run_dir,
                        client=client,
                        model=args.model,
                        min_cluster_posts=args.min_cluster_posts,
                        max_posts=args.max_posts,
                    )
                print(f"run_dir: {artifact.run_dir}")
                print(f"provider: {artifact.provider}")
                print(f"model: {artifact.model}")
                print(f"strongest_cluster_id: {artifact.strongest_cluster_id}")
                print(f"strongest_cluster_size: {artifact.strongest_cluster_size}")
                print(f"posts_used: {artifact.included_post_count}")
                print(f"final_memo_json: {args.run_dir / 'final_memo.json'}")
                print(f"final_memo_markdown: {args.run_dir / 'final_memo.md'}")
                return 0

            return asyncio.run(_run_memo())
        except (ConfigurationError, ValueError, FileNotFoundError) as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Memo generation failed: {exc}", file=sys.stderr)
            return 1

    if args.command == "reply-drafts":
        try:
            if args.max_posts <= 0:
                raise ValueError("--max-posts must be greater than 0")
            if not args.voice.strip():
                raise ValueError("--voice is required")
            config = load_llm_config(require_model=not args.model)

            async def _run_reply_drafts() -> int:
                async with LMStudioClient(config) as client:
                    artifact = await draft_reply_suggestions(
                        run_dir=args.run_dir,
                        client=client,
                        voice=args.voice,
                        model=args.model,
                        max_posts=args.max_posts,
                        score_threshold=args.score_threshold,
                        minimum_dimension_score=args.minimum_dimension_score,
                        max_improvement_rounds=args.max_improvement_rounds,
                    )
                print(f"run_dir: {artifact.run_dir}")
                print(f"provider: {artifact.provider}")
                print(f"model: {artifact.model}")
                print(f"selected_posts: {artifact.selected_post_count}")
                print(f"improvement_rounds: {artifact.improvement_rounds}")
                print(f"passed_threshold: {'yes' if artifact.passed_threshold else 'no'}")
                print(f"score_threshold: {artifact.score_threshold}")
                print(f"minimum_dimension_score: {artifact.minimum_dimension_score}")
                print("manual_review_only: yes")
                print(f"reply_drafts_json: {args.run_dir / 'reply_drafts.json'}")
                print(f"reply_drafts_markdown: {args.run_dir / 'reply_drafts.md'}")
                return 0

            return asyncio.run(_run_reply_drafts())
        except (ConfigurationError, ValueError, FileNotFoundError) as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Reply draft generation failed: {exc}", file=sys.stderr)
            return 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
