from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

from .config import (
    DEFAULT_LIMIT,
    DEFAULT_SORT,
    DEFAULT_TIME_FILTER,
    ConfigurationError,
    build_run_paths,
    ensure_repo_layout,
    load_runtime_config,
)
from .retrieval import run_search_command


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
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Per-request result limit (max 100)",
    )
    search_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional explicit run directory",
    )

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
                    limit=args.limit,
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

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
