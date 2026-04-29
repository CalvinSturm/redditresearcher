from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def merge_json_exports(input_files: list[Path], output_path: Path) -> dict:
    merged_posts: list[dict] = []
    seen_ids: set[str] = set()
    source_files: list[str] = []

    for fp in input_files:
        if not fp.exists():
            print(f"Warning: {fp} not found, skipping", file=sys.stderr)
            continue

        with open(fp, encoding="utf-8") as f:
            data = json.load(f)

        source_files.append(fp.name)
        posts = data.get("posts", [])
        if not isinstance(posts, list):
            print(f"Warning: {fp.name} has no 'posts' array, skipping", file=sys.stderr)
            continue

        for post in posts:
            post_id = post.get("id") or post.get("post_id")
            if post_id and post_id not in seen_ids:
                seen_ids.add(post_id)
                # Ensure source_queries field exists
                if "source_queries" not in post:
                    post["source_queries"] = []
                # Ensure source_subreddits field exists
                if "source_subreddits" not in post:
                    post["source_subreddits"] = []
                merged_posts.append(post)

    merged = {
        "source": "tampermonkey-merge",
        "merged_from": source_files,
        "posts": merged_posts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge Tampermonkey JSON exports into a single import bundle"
    )
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        required=True,
        help="Path to a Tampermonkey JSON export file (can specify multiple)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output merged JSON path (default: outputs/captures/merged-tampermonkey.json)",
    )
    args = parser.parse_args()

    input_files = [Path(p) for p in args.inputs]
    output_path = args.output or Path("outputs/captures/merged-tampermonkey.json")

    result = merge_json_exports(input_files, output_path)

    print(f"merged_from: {', '.join(result['merged_from'])}")
    print(f"total_posts: {len(result['posts'])}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
