from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reddit_pain_agent.artifact_store import build_artifact_store
from reddit_pain_agent.config import ConfigurationError, load_runtime_config
from reddit_pain_agent.main import main
from reddit_pain_agent.models import CandidatePost, RunManifest, SearchRequestSpec
from reddit_pain_agent.retrieval import build_search_specs, normalize_candidate


class ConfigTests(unittest.TestCase):
    def test_load_runtime_config_requires_client_id_and_user_agent(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ConfigurationError):
                load_runtime_config()

    def test_load_runtime_config_reads_explicit_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                "os.environ",
                {
                    "REDDIT_CLIENT_ID": "client-id",
                    "REDDIT_CLIENT_SECRET": "client-secret",
                    "REDDIT_USER_AGENT": "script:test:v1 (by /u/example)",
                    "REDDIT_OUTPUT_ROOT": tmpdir,
                    "REDDIT_REQUEST_TIMEOUT_SECONDS": "12",
                    "REDDIT_MAX_RETRIES": "4",
                    "REDDIT_MAX_CONCURRENT_REQUESTS": "2",
                },
                clear=True,
            ):
                config = load_runtime_config()

        self.assertEqual(config.reddit_client_id, "client-id")
        self.assertEqual(config.reddit_client_secret, "client-secret")
        self.assertEqual(config.output_root, Path(tmpdir))
        self.assertEqual(config.request_timeout_seconds, 12.0)
        self.assertEqual(config.max_retries, 4)
        self.assertEqual(config.max_concurrent_requests, 2)


class ArtifactStoreTests(unittest.TestCase):
    def test_artifact_store_writes_manifest_raw_payload_and_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_artifact_store(Path(tmpdir), "run-1")
            manifest = RunManifest(
                run_slug="run-1",
                status="running",
                started_at=datetime.now(UTC),
                output_dir=str(store.run_dir),
                subreddits=["Entrepreneur"],
                queries=["manual work"],
                sort="comments",
                time_filter="month",
                limit=25,
                request_timeout_seconds=30.0,
                max_retries=3,
                max_concurrent_requests=4,
            )
            store.write_manifest(manifest)
            raw_path = store.write_raw_search_payload(
                1,
                SearchRequestSpec(
                    subreddit="Entrepreneur",
                    query="manual work",
                    sort="comments",
                    time_filter="month",
                    limit=25,
                ),
                {"data": {"children": []}},
            )
            store.write_candidate_posts(
                [
                    CandidatePost(
                        id="abc123",
                        title="Painful workflow",
                        subreddit="Entrepreneur",
                        url="https://reddit.com/example",
                        source_queries=["manual work"],
                        source_subreddits=["Entrepreneur"],
                        retrieval_requests=["entrepreneur|manual work|comments|month|"],
                    )
                ]
            )

            self.assertTrue(store.manifest_path.exists())
            self.assertTrue((store.run_dir / raw_path).exists())
            self.assertTrue(store.candidate_posts_path.exists())


class RetrievalNormalizationTests(unittest.TestCase):
    def test_build_search_specs_expands_subreddits_and_queries(self) -> None:
        specs = build_search_specs(
            subreddits=["Entrepreneur", "vibecoding"],
            queries=["manual work", "client follow-up"],
            sort="comments",
            time_filter="month",
            limit=20,
        )
        self.assertEqual(len(specs), 4)

    def test_normalize_candidate_filters_deleted_and_normalizes_permalink(self) -> None:
        spec = SearchRequestSpec(
            subreddit="Entrepreneur",
            query="manual work",
            sort="comments",
            time_filter="month",
            limit=25,
        )
        candidate, reason = normalize_candidate(
            {
                "kind": "t3",
                "data": {
                    "id": "abc123",
                    "title": "Painful workflow",
                    "subreddit": "Entrepreneur",
                    "url": "https://reddit.com/example",
                    "permalink": "/r/Entrepreneur/comments/abc123/example/",
                    "score": 12,
                    "num_comments": 7,
                    "created_utc": 1710000000,
                    "selftext": "Too much copy/paste",
                },
            },
            spec,
        )

        self.assertEqual(reason, "")
        assert candidate is not None
        self.assertEqual(candidate.id, "abc123")
        self.assertEqual(
            candidate.permalink,
            "https://www.reddit.com/r/Entrepreneur/comments/abc123/example/",
        )

        deleted_candidate, deleted_reason = normalize_candidate(
            {
                "kind": "t3",
                "data": {
                    "id": "gone123",
                    "title": "[deleted]",
                    "subreddit": "Entrepreneur",
                    "url": "https://reddit.com/example",
                },
            },
            spec,
        )
        self.assertIsNone(deleted_candidate)
        self.assertEqual(deleted_reason, "deleted")


class CliTests(unittest.TestCase):
    def test_search_cli_prints_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                "os.environ",
                {
                    "REDDIT_CLIENT_ID": "client-id",
                    "REDDIT_USER_AGENT": "script:test:v1 (by /u/example)",
                    "REDDIT_OUTPUT_ROOT": tmpdir,
                },
                clear=True,
            ):
                fake_result = AsyncMock()
                fake_result.run_slug = "search-20260401-120000-entrepreneur-manual-work"
                fake_result.run_dir = Path(tmpdir) / fake_result.run_slug
                fake_result.request_count = 2
                fake_result.candidate_count = 5
                fake_result.filtered_counts = {"duplicate": 1, "deleted": 2}
                with patch(
                    "reddit_pain_agent.main.run_search_command",
                    AsyncMock(return_value=fake_result),
                ):
                    with patch("sys.stdout.write") as stdout_write:
                        exit_code = main(
                            [
                                "search",
                                "--subreddit",
                                "Entrepreneur",
                                "--query",
                                "manual work",
                            ]
                        )

        self.assertEqual(exit_code, 0)
        output = "".join(call.args[0] for call in stdout_write.call_args_list)
        self.assertIn("run_slug:", output)
        self.assertIn("candidate_posts: 5", output)
        self.assertIn("filtered: deleted=2, duplicate=1", output)


if __name__ == "__main__":
    unittest.main()
