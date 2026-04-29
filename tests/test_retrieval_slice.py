from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reddit_pain_agent.artifact_store import build_artifact_store
from reddit_pain_agent.clustering import (
    cluster_ranked_posts,
    cluster_run_posts,
    load_cluster_evidence_validation,
    load_strongest_cluster_posts,
    validate_cluster_evidence,
)
from reddit_pain_agent.config import ConfigurationError, LLMConfig, load_llm_config, load_runtime_config
from reddit_pain_agent.llm import LLMClientError, LMStudioClient, build_llm_client, extract_response_text
from reddit_pain_agent.main import main
from reddit_pain_agent.memo_writer import (
    build_final_memo_markdown,
    load_evidence_summary,
    load_strongest_cluster,
    write_final_memo,
)
from reddit_pain_agent.models import (
    AssetGenerationProvenance,
    CandidatePost,
    Comment,
    ManualImportPost,
    RankedCandidatePost,
    ReplyDraft,
    RunManifest,
    SearchRequestSpec,
    SubmissionCommentsArtifact,
    ThemeCluster,
)
from reddit_pain_agent.pain_analysis import (
    build_evidence_summary_markdown,
    load_candidate_posts,
    load_submission_comments,
    load_summary_posts,
    score_comment_for_evidence,
    select_comment_evidence,
)
from reddit_pain_agent.playwright_capture import (
    SearchResultPreview,
    build_reddit_search_urls,
    capture_reddit_threads,
    merge_captured_posts,
    repair_capture_timestamps,
    select_search_results,
)
from reddit_pain_agent.prompts import (
    build_candidate_evidence_prompt,
    build_final_memo_prompt,
    build_reply_drafts_prompt,
)
from reddit_pain_agent.reply_writer import (
    build_reply_drafts_markdown,
    draft_reply_suggestions,
    load_reply_source_posts,
    score_comment_opportunities,
)
from reddit_pain_agent.ranking import (
    analyze_comment_screening,
    has_complaint_signal,
    is_non_trivial_comment,
    load_selected_posts,
    rank_candidates,
    rank_run_candidates,
    score_candidate_post,
    score_thread_comment_opportunity,
)
from reddit_pain_agent.retrieval import (
    apply_candidate_quality_filters,
    build_search_specs,
    build_retrieval_quality_filters,
    expand_query_variants,
    enrich_run_with_comments,
    import_manual_search_bundle,
    normalize_candidate,
    normalize_comments_payload,
    normalize_morechildren_payload,
    run_search_command,
)


def _write_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprints_for_paths(run_dir: Path, *paths: Path) -> dict[str, str]:
    return {
        path.relative_to(run_dir).as_posix(): _sha256_file(path)
        for path in paths
    }


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

    def test_load_llm_config_supports_lmstudio(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "LLM_PROVIDER": "lmstudio",
                "LLM_BASE_URL": "http://127.0.0.1:1234/v1",
                "LLM_MODEL": "openai/gpt-oss-20b",
                "LLM_REQUEST_TIMEOUT_SECONDS": "45",
            },
            clear=True,
        ):
            config = load_llm_config(require_model=True)

        self.assertEqual(config.provider, "lmstudio")
        self.assertEqual(config.base_url, "http://127.0.0.1:1234/v1")
        self.assertEqual(config.model, "openai/gpt-oss-20b")
        self.assertEqual(config.request_timeout_seconds, 45.0)

    def test_load_llm_config_supports_openai(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-key",
                "LLM_MODEL": "gpt-5.2",
            },
            clear=True,
        ):
            config = load_llm_config(require_model=True)

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.base_url, "https://api.openai.com/v1")
        self.assertEqual(config.model, "gpt-5.2")
        self.assertEqual(config.api_key, "test-key")

    def test_load_llm_config_rejects_unsupported_provider(self) -> None:
        with patch.dict(
            "os.environ",
            {"LLM_PROVIDER": "anthropic"},
            clear=True,
        ):
            with self.assertRaises(ConfigurationError):
                load_llm_config()

    def test_load_llm_config_requires_openai_api_key(self) -> None:
        with patch.dict(
            "os.environ",
            {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-5.2"},
            clear=True,
        ):
            with self.assertRaises(ConfigurationError):
                load_llm_config(require_model=True)


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

    def test_artifact_store_writes_summary_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_artifact_store(Path(tmpdir), "run-2")
            generation = AssetGenerationProvenance(provider="lmstudio", model="test-model")
            prompt_path = store.write_prompt_text("candidate summary", "Prompt text", generation=generation)
            generation = generation.model_copy(update={"prompt_artifact_path": prompt_path})
            raw_response_path = store.write_raw_llm_response(
                "candidate summary",
                {"id": "resp_1"},
                generation=generation,
            )
            generation = generation.model_copy(update={"raw_response_artifact_path": raw_response_path})
            store.write_evidence_summary_markdown("# Evidence Summary\n", generation=generation)
            store.write_evidence_summary_json({"provider": "lmstudio"}, generation=generation)
            registry = json.loads(store.asset_registry_path.read_text(encoding="utf-8"))
            registry_by_path = {item["artifact_path"]: item for item in registry}

            self.assertTrue((store.run_dir / prompt_path).exists())
            self.assertTrue((store.run_dir / raw_response_path).exists())
            self.assertTrue(store.evidence_summary_markdown_path.exists())
            self.assertTrue(store.evidence_summary_json_path.exists())
            self.assertEqual(
                registry_by_path["evidence_summary.json"]["generation"],
                {
                    "provider": "lmstudio",
                    "model": "test-model",
                    "prompt_artifact_path": prompt_path,
                    "raw_response_artifact_path": raw_response_path,
                },
            )

    def test_artifact_store_writes_comment_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_artifact_store(Path(tmpdir), "run-3")
            raw_path = store.write_raw_comment_payload("abc123", [{"kind": "Listing"}])
            normalized_path = store.write_submission_comments(
                SubmissionCommentsArtifact(
                    submission_id="abc123",
                    subreddit="Entrepreneur",
                    permalink="/r/Entrepreneur/comments/abc123/example/",
                    title="Painful workflow",
                    fetched_comment_count=1,
                    comments=[Comment(id="c1", body="This is painful")],
                )
            )
            store.write_comment_enrichment_json({"comment_count": 1})

            self.assertTrue((store.run_dir / raw_path).exists())
            self.assertTrue((store.run_dir / normalized_path).exists())
            self.assertTrue(store.comment_enrichment_json_path.exists())

    def test_artifact_store_writes_manual_raw_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_artifact_store(Path(tmpdir), "run-manual")
            raw_path = store.write_raw_manual_payload("playwright-capture", {"posts": []})

            self.assertTrue((store.run_dir / raw_path).exists())

    def test_artifact_store_writes_ranking_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_artifact_store(Path(tmpdir), "run-4")
            store.write_candidate_screening_json({"candidate_count": 2, "kept_count": 1})
            store.write_post_ranking_json({"candidate_count": 2})
            store.write_selected_posts_json([{"candidate": {"id": "abc123", "title": "Pain"}}])

            self.assertTrue(store.candidate_screening_json_path.exists())
            self.assertTrue(store.post_ranking_json_path.exists())
            self.assertTrue(store.selected_posts_json_path.exists())

    def test_artifact_store_writes_theme_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_artifact_store(Path(tmpdir), "run-5")
            store.write_theme_summary_json({"cluster_count": 2})
            self.assertTrue(store.theme_summary_json_path.exists())

    def test_artifact_store_writes_final_memo_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_artifact_store(Path(tmpdir), "run-6")
            store.write_final_memo_json({"strongest_cluster_id": "cluster-1"})
            store.write_final_memo_markdown("# Final Memo\n")

            self.assertTrue(store.final_memo_json_path.exists())
            self.assertTrue(store.final_memo_markdown_path.exists())

    def test_artifact_store_writes_run_report_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_artifact_store(Path(tmpdir), "run-7")
            store.write_run_report_json({"status": "completed"})

            self.assertTrue(store.run_report_json_path.exists())

    def test_artifact_store_writes_reply_draft_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_artifact_store(Path(tmpdir), "run-8")
            store.write_reply_drafts_json({"selected_post_count": 2})
            store.write_reply_drafts_markdown("# Reply Drafts\n")

            self.assertTrue(store.reply_drafts_json_path.exists())
            self.assertTrue(store.reply_drafts_markdown_path.exists())

    def test_artifact_store_writes_comment_opportunities_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_artifact_store(Path(tmpdir), "run-9")
            store.write_comment_opportunities_json({"scored_post_count": 1, "opportunities": []})

            self.assertTrue(store.comment_opportunities_json_path.exists())


class RetrievalNormalizationTests(unittest.TestCase):
    def test_build_search_specs_expands_subreddits_and_queries(self) -> None:
        specs = build_search_specs(
            subreddits=["Entrepreneur", "vibecoding"],
            queries=["manual work", "client follow-up"],
            sort="comments",
            time_filter="month",
            limit=20,
            expand_queries=False,
        )
        self.assertEqual(len(specs), 4)

    def test_build_search_specs_expands_multiple_sorts_and_time_filters(self) -> None:
        specs = build_search_specs(
            subreddits=["Entrepreneur"],
            queries=["manual work"],
            sort="comments",
            time_filter="month",
            additional_sorts=["new"],
            additional_time_filters=["week"],
            limit=20,
            expand_queries=False,
        )
        self.assertEqual(len(specs), 4)
        self.assertEqual(
            {(spec.sort, spec.time_filter) for spec in specs},
            {
                ("comments", "month"),
                ("comments", "week"),
                ("new", "month"),
                ("new", "week"),
            },
        )

    def test_expand_query_variants_adds_phrase_and_keyword_forms(self) -> None:
        variants = expand_query_variants("pain points in coding workflows")
        self.assertEqual(
            variants,
            [
                "pain points in coding workflows",
                '"pain points in coding workflows"',
                "pain points coding workflows",
            ],
        )

    def test_build_retrieval_quality_filters_rejects_allow_deny_overlap(self) -> None:
        with self.assertRaises(ValueError):
            build_retrieval_quality_filters(
                allowed_subreddits=["Entrepreneur"],
                denied_subreddits=["entrepreneur"],
            )

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
        self.assertEqual(candidate.source_sorts, ["comments"])
        self.assertEqual(candidate.source_time_filters, ["month"])
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

    def test_apply_candidate_quality_filters_returns_expected_reason(self) -> None:
        candidate = CandidatePost(
            id="abc123",
            title="Painful workflow",
            subreddit="Entrepreneur",
            url="https://reddit.com/example",
            score=2,
            num_comments=1,
            over_18=True,
        )
        self.assertEqual(
            apply_candidate_quality_filters(
                candidate,
                build_retrieval_quality_filters(filter_nsfw=True),
            ),
            "nsfw",
        )
        self.assertEqual(
            apply_candidate_quality_filters(
                candidate,
                build_retrieval_quality_filters(min_score=3),
            ),
            "low_score",
        )
        self.assertEqual(
            apply_candidate_quality_filters(
                candidate,
                build_retrieval_quality_filters(min_comments=2),
            ),
            "low_comments",
        )
        self.assertEqual(
            apply_candidate_quality_filters(
                candidate,
                build_retrieval_quality_filters(allowed_subreddits=["vibecoding"]),
            ),
            "non_allowed_subreddit",
        )
        self.assertEqual(
            apply_candidate_quality_filters(
                candidate,
                build_retrieval_quality_filters(denied_subreddits=["Entrepreneur"]),
            ),
            "denied_subreddit",
        )

    def test_normalize_comments_payload_flattens_nested_comments(self) -> None:
        payload = [
            {"kind": "Listing", "data": {"children": []}},
            {
                "kind": "Listing",
                "data": {
                    "children": [
                        {
                            "kind": "t1",
                            "data": {
                                "id": "c1",
                                "body": "Top level complaint",
                                "score": 10,
                                "depth": 0,
                                "replies": {
                                    "data": {
                                        "children": [
                                            {
                                                "kind": "t1",
                                                "data": {
                                                    "id": "c2",
                                                    "body": "Nested agreement",
                                                    "score": 4,
                                                    "depth": 1,
                                                },
                                            }
                                        ]
                                    }
                                },
                            },
                        },
                        {
                            "kind": "t1",
                            "data": {"id": "c3", "body": "[deleted]"},
                        },
                    ]
                },
            },
        ]
        comments, morechildren = normalize_comments_payload(payload)
        self.assertEqual([comment.id for comment in comments], ["c1", "c2"])
        self.assertEqual(morechildren, [])

    def test_normalize_comments_payload_collects_morechildren_ids(self) -> None:
        payload = [
            {"kind": "Listing", "data": {"children": []}},
            {
                "kind": "Listing",
                "data": {
                    "children": [
                        {
                            "kind": "more",
                            "data": {"children": ["c3", "c4", "c3"]},
                        }
                    ]
                },
            },
        ]
        comments, morechildren = normalize_comments_payload(payload)
        self.assertEqual(comments, [])
        self.assertEqual(morechildren, ["c3", "c4"])

    def test_normalize_morechildren_payload_extracts_comments_and_nested_morechildren(self) -> None:
        payload = {
            "json": {
                "data": {
                    "things": [
                        {"kind": "more", "data": {"children": ["c9"]}},
                        {
                            "kind": "t1",
                            "data": {
                                "id": "c5",
                                "body": "Expanded comment",
                                "score": 3,
                                "depth": 1,
                            },
                        },
                    ]
                }
            }
        }
        comments, morechildren = normalize_morechildren_payload(payload)
        self.assertEqual([comment.id for comment in comments], ["c5"])
        self.assertEqual(morechildren, ["c9"])

    def test_run_search_command_paginates_and_merges_duplicates(self) -> None:
        class FakeRedditClient:
            def __init__(self) -> None:
                self.calls: list[SearchRequestSpec] = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def search_subreddit(self, spec: SearchRequestSpec):
                self.calls.append(spec)
                if spec.after is None:
                    payload = {
                        "data": {
                            "after": "t3_page2",
                            "children": [
                                {
                                    "kind": "t3",
                                    "data": {
                                        "id": "abc123",
                                        "title": "Painful workflow",
                                        "subreddit": spec.subreddit,
                                        "url": "https://reddit.com/example",
                                        "score": 10,
                                        "num_comments": 5,
                                        "selftext": "Too much manual work",
                                    },
                                }
                            ],
                        }
                    }
                else:
                    payload = {
                        "data": {
                            "after": None,
                            "children": [
                                {
                                    "kind": "t3",
                                    "data": {
                                        "id": "abc123",
                                        "title": "Painful workflow",
                                        "subreddit": spec.subreddit,
                                        "url": "https://reddit.com/example",
                                        "score": 11,
                                        "num_comments": 6,
                                        "selftext": "Still too much manual work",
                                    },
                                },
                                {
                                    "kind": "t3",
                                    "data": {
                                        "id": "def456",
                                        "title": "CRM follow-up breaks",
                                        "subreddit": spec.subreddit,
                                        "url": "https://reddit.com/example-2",
                                        "score": 7,
                                        "num_comments": 4,
                                        "selftext": "Leads fall through cracks",
                                    },
                                },
                            ],
                        }
                    }
                log_entry = type(
                    "LogEntry",
                    (),
                    {
                        "raw_artifact_path": None,
                        "model_dump": lambda self, mode="json": {
                            "requested_at": "2026-04-02T00:00:00Z",
                            "request_name": "search:test",
                            "method": "GET",
                            "url": "https://oauth.reddit.com/test",
                            "params": {"after": spec.after or ""},
                            "status_code": 200,
                            "duration_ms": 10.0,
                            "attempt": 1,
                            "rate_limit": None,
                            "raw_artifact_path": self.raw_artifact_path,
                            "error": None,
                        },
                    },
                )()
                return payload, log_entry

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
                config = load_runtime_config()
                fake_client = FakeRedditClient()
                run_dir = Path(tmpdir) / "run-1"
                result = asyncio.run(
                    run_search_command(
                        config=config,
                        subreddits=["Entrepreneur"],
                        queries=["manual follow-up pain"],
                        limit=25,
                        pages_per_query=2,
                        expand_queries=False,
                        output_dir=run_dir,
                        client=fake_client,
                    )
                )

            self.assertEqual(result.request_count, 2)
            self.assertEqual(result.candidate_count, 2)
            self.assertEqual(result.query_variant_count, 1)
            self.assertEqual(result.search_spec_count, 1)
            self.assertEqual(result.sort_count, 1)
            self.assertEqual(result.time_filter_count, 1)
            self.assertEqual(result.pages_per_query, 2)
            self.assertEqual(result.filtered_counts["duplicate"], 1)
            self.assertEqual([call.after for call in fake_client.calls], [None, "t3_page2"])
            manifest_payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["pages_per_query"], 2)
            self.assertEqual(manifest_payload["query_variants"], ["manual follow-up pain"])

    def test_run_search_command_merges_sort_and_time_filter_provenance(self) -> None:
        class FakeRedditClient:
            def __init__(self) -> None:
                self.calls: list[SearchRequestSpec] = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def search_subreddit(self, spec: SearchRequestSpec):
                self.calls.append(spec)
                payload = {
                    "data": {
                        "after": None,
                        "children": [
                            {
                                "kind": "t3",
                                "data": {
                                    "id": "abc123",
                                    "title": "Painful workflow",
                                    "subreddit": spec.subreddit,
                                    "url": "https://reddit.com/example",
                                    "score": 10,
                                    "num_comments": 5,
                                    "selftext": "Too much manual work",
                                },
                            }
                        ],
                    }
                }
                log_entry = type(
                    "LogEntry",
                    (),
                    {
                        "raw_artifact_path": None,
                        "model_dump": lambda self, mode="json": {
                            "requested_at": "2026-04-02T00:00:00Z",
                            "request_name": "search:test",
                            "method": "GET",
                            "url": "https://oauth.reddit.com/test",
                            "params": {"sort": spec.sort, "t": spec.time_filter},
                            "status_code": 200,
                            "duration_ms": 10.0,
                            "attempt": 1,
                            "rate_limit": None,
                            "raw_artifact_path": self.raw_artifact_path,
                            "error": None,
                        },
                    },
                )()
                return payload, log_entry

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
                config = load_runtime_config()
                run_dir = Path(tmpdir) / "run-2"
                asyncio.run(
                    run_search_command(
                        config=config,
                        subreddits=["Entrepreneur"],
                        queries=["manual follow-up pain"],
                        sort="comments",
                        time_filter="month",
                        additional_sorts=["new"],
                        additional_time_filters=["week"],
                        limit=25,
                        pages_per_query=1,
                        expand_queries=False,
                        output_dir=run_dir,
                        client=FakeRedditClient(),
                    )
                )

            candidates = json.loads((run_dir / "candidate_posts.json").read_text(encoding="utf-8"))
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["source_sorts"], ["comments", "new"])
            self.assertEqual(candidates[0]["source_time_filters"], ["month", "week"])

    def test_run_search_command_filters_low_quality_candidates(self) -> None:
        class FakeRedditClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def search_subreddit(self, spec: SearchRequestSpec):
                payload = {
                    "data": {
                        "after": None,
                        "children": [
                            {
                                "kind": "t3",
                                "data": {
                                    "id": "good123",
                                    "title": "Good candidate",
                                    "subreddit": "Entrepreneur",
                                    "url": "https://reddit.com/good",
                                    "score": 10,
                                    "num_comments": 6,
                                    "selftext": "Useful pain signal",
                                    "over_18": False,
                                },
                            },
                            {
                                "kind": "t3",
                                "data": {
                                    "id": "low-score",
                                    "title": "Low score candidate",
                                    "subreddit": "Entrepreneur",
                                    "url": "https://reddit.com/low-score",
                                    "score": 1,
                                    "num_comments": 8,
                                    "selftext": "Too weak",
                                },
                            },
                            {
                                "kind": "t3",
                                "data": {
                                    "id": "low-comments",
                                    "title": "Low comments candidate",
                                    "subreddit": "Entrepreneur",
                                    "url": "https://reddit.com/low-comments",
                                    "score": 10,
                                    "num_comments": 1,
                                    "selftext": "Too few comments",
                                },
                            },
                            {
                                "kind": "t3",
                                "data": {
                                    "id": "nsfw123",
                                    "title": "NSFW candidate",
                                    "subreddit": "Entrepreneur",
                                    "url": "https://reddit.com/nsfw",
                                    "score": 10,
                                    "num_comments": 10,
                                    "selftext": "Should be filtered",
                                    "over_18": True,
                                },
                            },
                            {
                                "kind": "t3",
                                "data": {
                                    "id": "denied123",
                                    "title": "Denied subreddit candidate",
                                    "subreddit": "vibecoding",
                                    "url": "https://reddit.com/denied",
                                    "score": 10,
                                    "num_comments": 10,
                                    "selftext": "Wrong subreddit",
                                },
                            },
                        ],
                    }
                }
                log_entry = type(
                    "LogEntry",
                    (),
                    {
                        "raw_artifact_path": None,
                        "model_dump": lambda self, mode="json": {
                            "requested_at": "2026-04-02T00:00:00Z",
                            "request_name": "search:test",
                            "method": "GET",
                            "url": "https://oauth.reddit.com/test",
                            "params": {},
                            "status_code": 200,
                            "duration_ms": 10.0,
                            "attempt": 1,
                            "rate_limit": None,
                            "raw_artifact_path": self.raw_artifact_path,
                            "error": None,
                        },
                    },
                )()
                return payload, log_entry

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
                config = load_runtime_config()
                run_dir = Path(tmpdir) / "run-quality"
                result = asyncio.run(
                    run_search_command(
                        config=config,
                        subreddits=["Entrepreneur"],
                        queries=["manual follow-up pain"],
                        min_score=3,
                        min_comments=3,
                        filter_nsfw=True,
                        denied_subreddits=["vibecoding"],
                        expand_queries=False,
                        output_dir=run_dir,
                        client=FakeRedditClient(),
                    )
                )

            self.assertEqual(result.candidate_count, 1)
            self.assertEqual(
                result.filtered_counts,
                {
                    "denied_subreddit": 1,
                    "low_comments": 1,
                    "low_score": 1,
                    "nsfw": 1,
                },
            )
            manifest_payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["min_score"], 3)
            self.assertEqual(manifest_payload["min_comments"], 3)
            self.assertTrue(manifest_payload["filter_nsfw"])
            self.assertEqual(manifest_payload["denied_subreddits"], ["vibecoding"])

    def test_import_manual_search_bundle_writes_candidate_and_comment_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "manual.json"
            _write_json_file(
                input_path,
                {
                    "posts": [
                        {
                            "id": "dup123",
                            "title": "Spreadsheet lead follow-up keeps breaking",
                            "subreddit": "Entrepreneur",
                            "url": "https://reddit.com/dup123",
                            "permalink": "/r/Entrepreneur/comments/dup123/example/",
                            "score": 22,
                            "num_comments": 9,
                            "selftext": "Still doing reminders by hand.",
                            "comments": [
                                {
                                    "id": "c1",
                                    "body": "Exact same issue here",
                                    "score": 5,
                                    "depth": 0,
                                }
                            ],
                        },
                        {
                            "id": "dup123",
                            "title": "Duplicate surfaced in another manual search",
                            "subreddit": "Entrepreneur",
                            "url": "https://reddit.com/dup123",
                            "score": 24,
                            "num_comments": 10,
                            "selftext": "Duplicate record",
                        },
                        {
                            "id": "low-comments",
                            "title": "Weak discussion",
                            "subreddit": "Entrepreneur",
                            "url": "https://reddit.com/low-comments",
                            "score": 30,
                            "num_comments": 1,
                            "selftext": "Too thin",
                        },
                    ]
                },
            )
            run_dir = Path(tmpdir) / "run-manual"

            result = import_manual_search_bundle(
                input_path=input_path,
                output_root=Path(tmpdir),
                subreddits=["Entrepreneur"],
                queries=["manual follow-up pain"],
                sort="comments",
                time_filter="month",
                min_comments=3,
                output_dir=run_dir,
            )

            self.assertEqual(result.imported_submission_count, 3)
            self.assertEqual(result.search_result.candidate_count, 1)
            self.assertEqual(result.search_result.filtered_counts, {"duplicate": 1, "low_comments": 1})
            self.assertEqual(result.commented_submission_count, 1)
            self.assertEqual(result.comments_result.comment_count, 1)
            self.assertTrue((run_dir / "candidate_posts.json").exists())
            self.assertTrue((run_dir / "comment_enrichment.json").exists())
            self.assertTrue((run_dir / "comments" / "dup123.json").exists())
            self.assertTrue((run_dir / result.raw_manual_artifacts[0]).exists())
            manifest_payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["retrieval_mode"], "manual")
            self.assertEqual(manifest_payload["manual_input_path"], str(input_path))
            self.assertEqual(manifest_payload["raw_manual_artifacts"], result.raw_manual_artifacts)

    def test_import_manual_search_bundle_accepts_userscript_export_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "userscript.json"
            _write_json_file(
                input_path,
                {
                    "meta": {
                        "mode": "posts_with_full_content_and_comments",
                        "scraped_at": "2026-04-02T15:00:00Z",
                    },
                    "posts": [
                        {
                            "post_id": "abc123",
                            "title": "Manual follow-up still breaks our spreadsheet CRM",
                            "subreddit": "r/Entrepreneur",
                            "permalink": "https://www.reddit.com/r/Entrepreneur/comments/abc123/example/",
                            "score": 41,
                            "comments": 18,
                            "created": "2026-04-02T14:00:00Z",
                            "body_full": "We still track follow-ups in a spreadsheet and things slip.",
                            "comments_full": [
                                {
                                    "comment_id": "c1",
                                    "body": "Same problem here. We still do this manually every day.",
                                    "author": "alice",
                                    "depth": 0,
                                    "created": "2026-04-02T14:10:00Z",
                                    "permalink": "https://www.reddit.com/r/Entrepreneur/comments/abc123/example/c1/",
                                }
                            ],
                        }
                    ],
                },
            )
            run_dir = Path(tmpdir) / "run-userscript"

            result = import_manual_search_bundle(
                input_path=input_path,
                output_root=Path(tmpdir),
                subreddits=["Entrepreneur"],
                queries=["manual follow-up pain"],
                sort="comments",
                time_filter="month",
                output_dir=run_dir,
            )

            self.assertEqual(result.search_result.candidate_count, 1)
            candidate_payload = json.loads((run_dir / "candidate_posts.json").read_text(encoding="utf-8"))
            self.assertEqual(candidate_payload[0]["id"], "abc123")
            self.assertEqual(candidate_payload[0]["subreddit"], "Entrepreneur")
            self.assertEqual(
                candidate_payload[0]["selftext"],
                "We still track follow-ups in a spreadsheet and things slip.",
            )
            comment_payload = json.loads((run_dir / "comments" / "abc123.json").read_text(encoding="utf-8"))
            self.assertEqual(comment_payload["comments"][0]["id"], "c1")
            self.assertEqual(
                comment_payload["comments"][0]["body"],
                "Same problem here. We still do this manually every day.",
            )


class PlaywrightCaptureTests(unittest.TestCase):
    def test_build_reddit_search_urls_composes_subreddits_and_queries(self) -> None:
        urls = build_reddit_search_urls(
            subreddits=["Entrepreneur", "sales"],
            queries=["manual follow-up pain"],
            sort="comments",
            time_filter="month",
        )

        self.assertEqual(
            urls,
            [
                "https://www.reddit.com/r/Entrepreneur/search/?q=manual+follow-up+pain&sort=comments&t=month&type=link",
                "https://www.reddit.com/r/sales/search/?q=manual+follow-up+pain&sort=comments&t=month&type=link",
            ],
        )

    def test_select_search_results_supports_top_n_and_explicit_indices(self) -> None:
        previews = [
            SearchResultPreview(
                title="A",
                url="https://www.reddit.com/r/Entrepreneur/comments/a/example",
                subreddit="Entrepreneur",
                source_search_url="https://reddit.com/search-1",
            ),
            SearchResultPreview(
                title="B",
                url="https://www.reddit.com/r/Entrepreneur/comments/b/example",
                subreddit="Entrepreneur",
                source_search_url="https://reddit.com/search-1",
            ),
            SearchResultPreview(
                title="B duplicate",
                url="https://www.reddit.com/r/Entrepreneur/comments/b/example",
                subreddit="Entrepreneur",
                source_search_url="https://reddit.com/search-2",
            ),
        ]

        top_results = select_search_results(previews, select_results=[], max_posts=2)
        self.assertEqual([item.title for item in top_results], ["A", "B"])

        explicit_results = select_search_results(previews, select_results=[2], max_posts=3)
        self.assertEqual([item.title for item in explicit_results], ["B"])

        with self.assertRaises(ValueError):
            select_search_results(previews, select_results=[4], max_posts=3)

    def test_merge_captured_posts_dedupes_comments_and_preserves_provenance(self) -> None:
        merged = merge_captured_posts(
            [
                ManualImportPost(
                    id="abc123",
                    title="Manual follow-up is painful",
                    subreddit="Entrepreneur",
                    url="https://reddit.com/a",
                    score=10,
                    num_comments=2,
                    created_utc=1710000000,
                    source_queries=["manual follow-up pain"],
                    source_subreddits=["Entrepreneur"],
                    source_sorts=["comments"],
                    source_time_filters=["month"],
                    retrieval_requests=["playwright:https://reddit.com/a"],
                    comments=[Comment(id="c1", body="same issue")],
                ),
                ManualImportPost(
                    id="abc123",
                    title="Manual follow-up is painful",
                    subreddit="Entrepreneur",
                    url="https://reddit.com/a",
                    score=12,
                    num_comments=3,
                    created_utc=1712000000,
                    source_queries=["spreadsheet crm pain"],
                    source_subreddits=["Entrepreneur"],
                    source_sorts=["new"],
                    source_time_filters=["week"],
                    retrieval_requests=["playwright:https://reddit.com/a?sort=new"],
                    comments=[Comment(id="c1", body="same issue"), Comment(id="c2", body="still manual")],
                ),
            ]
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].score, 12)
        self.assertEqual(merged[0].num_comments, 3)
        self.assertEqual(merged[0].created_utc, 1712000000)
        self.assertEqual([comment.id for comment in merged[0].comments], ["c1", "c2"])
        self.assertEqual(
            merged[0].source_queries,
            ["manual follow-up pain", "spreadsheet crm pain"],
        )

    def test_capture_reddit_threads_writes_log_and_snapshots(self) -> None:
        class FakePage:
            def __init__(self) -> None:
                self.current_url = ""

            async def goto(self, url: str, *, wait_until: str, timeout: int):
                self.current_url = url
                return None

            async def wait_for_timeout(self, timeout_ms: int):
                return None

            async def evaluate(self, expression: str):
                if "/search/" in self.current_url:
                    return [
                        {
                            "url": "https://www.reddit.com/r/Entrepreneur/comments/abc123/example/",
                            "title": "Manual follow-up is painful",
                            "subreddit": "Entrepreneur",
                        }
                    ]
                return {
                    "id": "abc123",
                    "title": "Manual follow-up is painful",
                    "subreddit": "Entrepreneur",
                    "url": "https://www.reddit.com/r/Entrepreneur/comments/abc123/example/",
                    "permalink": "/r/Entrepreneur/comments/abc123/example/",
                    "created_utc": 1775600000,
                    "selftext": "Still using spreadsheets.",
                    "num_comments": 1,
                    "comments": [
                        {
                            "id": "c1",
                            "body": "Same issue here",
                            "score": 5,
                            "depth": 0,
                            "created_utc": 1775600300,
                        }
                    ],
                }

            async def content(self) -> str:
                return f"<html><body>{self.current_url}</body></html>"

            async def screenshot(self, *, path: str, full_page: bool):
                Path(path).write_bytes(b"PNG")
                return None

            async def close(self):
                return None

        class FakeContext:
            def __init__(self):
                self.page = FakePage()

            async def __aenter__(self):
                return self.page

            async def __aexit__(self, exc_type, exc, tb):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "capture.json"
            with patch(
                "reddit_pain_agent.playwright_capture._open_playwright_page",
                return_value=FakeContext(),
            ):
                result = asyncio.run(
                    capture_reddit_threads(
                        subreddits=["Entrepreneur"],
                        queries=["manual follow-up pain"],
                        sort="comments",
                        time_filter="month",
                        max_posts=1,
                        max_comments=5,
                        output_path=output_path,
                    )
                )

            self.assertTrue(output_path.exists())
            self.assertTrue(result.log_path.exists())
            self.assertTrue(result.snapshot_dir.exists())
            self.assertTrue((result.snapshot_dir / "search-001.html").exists())
            self.assertTrue((result.snapshot_dir / "search-001.png").exists())
            self.assertTrue((result.snapshot_dir / "thread-001.html").exists())
            self.assertTrue((result.snapshot_dir / "thread-001.png").exists())
            self.assertEqual(result.html_snapshot_count, 2)
            self.assertEqual(result.screenshot_count, 2)
            self.assertEqual(result.page_error_count, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["posts"][0]["created_utc"], 1775600000)
            self.assertEqual(payload["posts"][0]["comments"][0]["created_utc"], 1775600300)
            log_text = result.log_path.read_text(encoding="utf-8")
            self.assertIn('"event": "search_page_extracted"', log_text)
            self.assertIn('"event": "thread_page_extracted"', log_text)

    def test_repair_capture_timestamps_backfills_from_saved_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            capture_json_path = Path(tmpdir) / "manual_capture.json"
            snapshot_dir = Path(tmpdir) / "manual_capture"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            _write_json_file(
                capture_json_path,
                {
                    "source": "playwright",
                    "posts": [
                        {
                            "id": "abc123",
                            "title": "Manual follow-up is painful",
                            "subreddit": "Entrepreneur",
                            "url": "https://www.reddit.com/r/Entrepreneur/comments/abc123/example/",
                            "created_utc": None,
                            "comments": [
                                {
                                    "id": "c1",
                                    "body": "Same issue here",
                                    "created_utc": None,
                                }
                            ],
                        }
                    ],
                },
            )
            (snapshot_dir / "thread-001.html").write_text(
                (
                    '<shreddit-post id="t3_abc123" created-timestamp="2026-04-07T12:00:00.000000+0000"></shreddit-post>'
                    '<shreddit-comment created="2026-04-07T12:05:00.000000+0000" thingid="t1_c1"></shreddit-comment>'
                ),
                encoding="utf-8",
            )

            result = repair_capture_timestamps(
                capture_json_path=capture_json_path,
                snapshot_dir=snapshot_dir,
            )

            repaired_payload = json.loads(capture_json_path.read_text(encoding="utf-8"))
            self.assertEqual(result.repaired_post_count, 1)
            self.assertEqual(result.repaired_comment_count, 1)
            self.assertAlmostEqual(
                repaired_payload["posts"][0]["created_utc"],
                datetime(2026, 4, 7, 12, 0, tzinfo=UTC).timestamp(),
            )
            self.assertAlmostEqual(
                repaired_payload["posts"][0]["comments"][0]["created_utc"],
                datetime(2026, 4, 7, 12, 5, tzinfo=UTC).timestamp(),
            )


class LLMTests(unittest.TestCase):
    def test_extract_response_text_prefers_output_text(self) -> None:
        text = extract_response_text({"output_text": "Hello from LM Studio"})
        self.assertEqual(text, "Hello from LM Studio")

    def test_extract_response_text_falls_back_to_output_blocks(self) -> None:
        payload = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "First line"},
                        {"type": "output_text", "text": "Second line"},
                    ]
                }
            ]
        }
        self.assertEqual(extract_response_text(payload), "First line\nSecond line")

    def test_lmstudio_client_surfaces_timeout_message(self) -> None:
        async def _run() -> None:
            async def handler(request: httpx.Request) -> httpx.Response:
                raise httpx.ReadTimeout("timed out", request=request)

            transport = httpx.MockTransport(handler)
            config = LLMConfig(
                provider="lmstudio",
                base_url="http://127.0.0.1:1234/v1",
                model="qwen/test",
                api_key=None,
                request_timeout_seconds=60.0,
            )
            async with LMStudioClient(config, transport=transport) as client:
                with self.assertRaises(LLMClientError) as ctx:
                    await client.generate_response("Hello")
            self.assertIn("timed out after 60s", str(ctx.exception))
            self.assertIn("qwen/test", str(ctx.exception))

        asyncio.run(_run())

    def test_lmstudio_client_surfaces_http_status_message(self) -> None:
        async def _run() -> None:
            async def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(503, json={"error": "model loading"}, request=request)

            transport = httpx.MockTransport(handler)
            config = LLMConfig(
                provider="lmstudio",
                base_url="http://127.0.0.1:1234/v1",
                model="qwen/test",
                api_key=None,
                request_timeout_seconds=60.0,
            )
            async with LMStudioClient(config, transport=transport) as client:
                with self.assertRaises(LLMClientError) as ctx:
                    await client.generate_response("Hello")
            self.assertIn("HTTP 503", str(ctx.exception))
            self.assertIn("model loading", str(ctx.exception))

        asyncio.run(_run())

    def test_build_llm_client_supports_openai_provider(self) -> None:
        config = LLMConfig(
            provider="openai",
            base_url="https://api.openai.com/v1",
            model="gpt-5.2",
            api_key="test-key",
            request_timeout_seconds=60.0,
        )
        client = build_llm_client(config)
        self.assertIsInstance(client, LMStudioClient)


class PromptAndSummaryTests(unittest.TestCase):
    def test_build_candidate_evidence_prompt_includes_post_fields(self) -> None:
        prompt = build_candidate_evidence_prompt(
            [
                CandidatePost(
                    id="abc123",
                    title="Painful workflow",
                    subreddit="Entrepreneur",
                    url="https://reddit.com/example",
                    score=12,
                    num_comments=7,
                    selftext="Too much copy/paste",
                    source_queries=["manual work"],
                    source_subreddits=["Entrepreneur"],
                    retrieval_requests=["entrepreneur|manual work|comments|month|"],
                )
            ],
            comments_by_submission={
                "abc123": [
                    Comment(id="c1", body="I have this exact issue", score=5, depth=0),
                ]
            },
        )
        self.assertIn("## Candidate Pain Themes", prompt)
        self.assertIn("Title: Painful workflow", prompt)
        self.assertIn("Queries: manual work", prompt)
        self.assertIn("Representative Comments:", prompt)
        self.assertIn("I have this exact issue", prompt)
        self.assertIn("# Research Context", prompt)

    def test_build_final_memo_prompt_includes_theme_summary_and_sections(self) -> None:
        prompt = build_final_memo_prompt(
            theme_cluster=ThemeCluster(
                cluster_id="cluster-1",
                label="crm / follow-up / spreadsheet",
                post_ids=["abc123"],
                size=5,
                average_post_score=8.4,
                total_comment_count=41,
                top_terms=["crm", "follow-up", "spreadsheet"],
                member_ranks=[1],
                cohesion_score=0.33,
            ),
            posts=[
                CandidatePost(
                    id="abc123",
                    title="Manual follow-up still breaks our CRM workflow",
                    subreddit="Entrepreneur",
                    url="https://reddit.com/example",
                    score=12,
                    num_comments=7,
                    selftext="Too much copy and paste every week.",
                )
            ],
            evidence_summary_text="## Repeated Complaints\n\nManual follow-up keeps showing up.",
        )
        self.assertIn("## Topic Overview", prompt)
        self.assertIn("## Product Opportunities", prompt)
        self.assertIn("cluster_id: cluster-1", prompt)
        self.assertIn("Manual follow-up keeps showing up.", prompt)
        self.assertIn("# Research Context", prompt)

    def test_build_reply_drafts_prompt_includes_voice_and_post_ids(self) -> None:
        prompt = build_reply_drafts_prompt(
            [
                RankedCandidatePost(
                    candidate=CandidatePost(
                        id="abc123",
                        title="Manual follow-up is eating my week",
                        subreddit="Entrepreneur",
                        url="https://reddit.com/example",
                        selftext="Still doing this in spreadsheets.",
                    ),
                    saved_comment_count=0,
                    breakdown={"total_score": 7.0},
                    rank=1,
                )
            ],
            voice="calm, practical, first-person founder voice",
            max_posts=1,
        )
        self.assertIn("User voice: calm, practical, first-person founder voice", prompt)
        self.assertIn("post_id: abc123", prompt)
        self.assertIn("1 to 3 short paragraphs", prompt)
        self.assertIn("Reddit-friendly", prompt)

    def test_load_candidate_posts_reads_candidate_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "candidate_posts.json").write_text(
                '[{"id":"abc123","title":"Painful workflow","subreddit":"Entrepreneur","url":"https://reddit.com/example"}]\n',
                encoding="utf-8",
            )
            posts = load_candidate_posts(run_dir)

        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].id, "abc123")

    def test_build_evidence_summary_markdown_wraps_summary(self) -> None:
        markdown = build_evidence_summary_markdown(
            "## Repeated Complaints\n\nPeople hate manual work.",
            provider="lmstudio",
            model="openai/gpt-oss-20b",
            candidate_count=3,
            comment_count=8,
            selected_comment_count=4,
        )
        self.assertIn("# Evidence Summary", markdown)
        self.assertIn("candidate_posts_used: 3", markdown)
        self.assertIn("saved_comments_used: 8", markdown)
        self.assertIn("selected_comments_used: 4", markdown)
        self.assertIn("People hate manual work.", markdown)

    def test_build_final_memo_markdown_wraps_memo(self) -> None:
        markdown = build_final_memo_markdown(
            "# Executive Summary\n\nThe theme is real.",
            provider="lmstudio",
            model="openai/gpt-oss-20b",
            strongest_cluster=ThemeCluster(
                cluster_id="cluster-1",
                label="crm / follow-up / spreadsheet",
                post_ids=["a", "b", "c", "d", "e"],
                size=5,
                average_post_score=8.7,
                total_comment_count=52,
                top_terms=["crm", "follow-up", "spreadsheet"],
                member_ranks=[1, 2, 3, 4, 5],
                cohesion_score=0.41,
            ),
            included_post_count=5,
            topic="manual follow-up pain",
            target_audience="founder-led sales teams",
            category="business",
            time_horizon="recent",
            source_thread_urls=["https://reddit.com/a", "https://reddit.com/b"],
        )
        self.assertIn("# Final Memo", markdown)
        self.assertIn("strongest_cluster_id: cluster-1", markdown)
        self.assertIn("posts_used: 5", markdown)
        self.assertIn("topic: manual follow-up pain", markdown)
        self.assertIn("## Source Threads", markdown)
        self.assertIn("https://reddit.com/a", markdown)
        self.assertIn("The theme is real.", markdown)

    def test_build_reply_drafts_markdown_renders_manual_review_metadata(self) -> None:
        markdown = build_reply_drafts_markdown(
            [
                ReplyDraft(
                    post_id="abc123",
                    title="Manual follow-up is eating my week",
                    subreddit="Entrepreneur",
                    url="https://reddit.com/example",
                    rank=1,
                    reply_text="I can relate to this because the manual handoff is where things usually break.",
                )
            ],
            provider="lmstudio",
            model="openai/gpt-oss-20b",
            voice="plainspoken founder",
        )
        self.assertIn("- manual_review_only: yes", markdown)
        self.assertIn("I can relate to this because the manual handoff is where things usually break.", markdown)

    def test_load_submission_comments_reads_comment_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            comments_dir = run_dir / "comments"
            comments_dir.mkdir(parents=True, exist_ok=True)
            (comments_dir / "abc123.json").write_text(
                (
                    '{"submission_id":"abc123","subreddit":"Entrepreneur","title":"Painful workflow",'
                    '"fetched_comment_count":1,"comments":[{"id":"c1","body":"Exact same issue"}]}'
                ),
                encoding="utf-8",
            )
            comments = load_submission_comments(run_dir)

        self.assertEqual(len(comments["abc123"]), 1)
        self.assertEqual(comments["abc123"][0].id, "c1")

    def test_load_evidence_summary_reads_summary_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "evidence_summary.json").write_text(
                (
                    '{"run_dir":"x","generated_at":"2026-04-02T00:00:00Z","provider":"lmstudio",'
                    '"model":"openai/gpt-oss-20b","candidate_count":5,"comment_count":4,'
                    '"selected_comment_count":3,"max_posts_used":5,'
                    '"prompt_artifact_path":"prompts/candidate-evidence-summary.txt",'
                    '"raw_response_artifact_path":"raw/llm/candidate-evidence-summary.json",'
                    '"summary_markdown_artifact_path":"evidence_summary.md",'
                    '"summary_text":"## Repeated Complaints\\n\\nManual follow-up keeps showing up."}'
                ),
                encoding="utf-8",
            )
            artifact = load_evidence_summary(run_dir)

        self.assertEqual(artifact.provider, "lmstudio")
        self.assertIn("Manual follow-up", artifact.summary_text)

    def test_load_strongest_cluster_reads_theme_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "theme_summary.json").write_text(
                (
                    '{"run_dir":"x","generated_at":"2026-04-02T00:00:00Z","source_post_count":5,'
                    '"cluster_count":1,"strongest_cluster_id":"cluster-1","strongest_post_ids":["a","b","c","d","e"],'
                    '"clusters":[{"cluster_id":"cluster-1","label":"crm / follow-up","post_ids":["a","b","c","d","e"],'
                    '"size":5,"average_post_score":8.6,"total_comment_count":47,'
                    '"top_terms":["crm","follow-up"],"member_ranks":[1,2,3,4,5],"cohesion_score":0.39}]}'
                ),
                encoding="utf-8",
            )
            cluster = load_strongest_cluster(run_dir)

        self.assertEqual(cluster.cluster_id, "cluster-1")
        self.assertEqual(cluster.size, 5)

    def test_load_summary_posts_prefers_selected_posts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "candidate_posts.json").write_text(
                '[{"id":"raw1","title":"Raw post","subreddit":"Entrepreneur","url":"https://reddit.com/raw"}]\n',
                encoding="utf-8",
            )
            (run_dir / "selected_posts.json").write_text(
                (
                    '[{"candidate":{"id":"sel1","title":"Selected post","subreddit":"Entrepreneur",'
                    '"url":"https://reddit.com/selected"},"saved_comment_count":0,'
                    '"breakdown":{"total_score":9.0},"rank":1}]'
                ),
                encoding="utf-8",
            )
            posts = load_summary_posts(run_dir)

        self.assertEqual(posts[0].id, "sel1")

    def test_load_reply_source_posts_prefers_selected_posts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "selected_posts.json").write_text(
                (
                    '[{"candidate":{"id":"abc123","title":"Manual follow-up is eating my week",'
                    '"subreddit":"Entrepreneur","url":"https://reddit.com/example"},'
                    '"saved_comment_count":0,"breakdown":{"total_score":7.0},"rank":1}]'
                ),
                encoding="utf-8",
            )
            posts = load_reply_source_posts(run_dir)

        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].candidate.id, "abc123")

    def test_load_reply_source_posts_prioritizes_high_value_comment_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "selected_posts.json").write_text(
                (
                    '[{"candidate":{"id":"weak","title":"What tools do people use for outreach",'
                    '"subreddit":"Entrepreneur","url":"https://reddit.com/weak","selftext":"Curious what people use.",'
                    '"source_queries":["manual follow-up pain"],"created_utc":1700000000},"saved_comment_count":0,"breakdown":{"total_score":9.0},"rank":1},'
                    '{"candidate":{"id":"strong","title":"We still miss leads because manual follow-up is killing our workflow",'
                    '"subreddit":"Entrepreneur","url":"https://reddit.com/strong","selftext":"I need to fix this now.",'
                    '"source_queries":["manual follow-up pain"],"created_utc":1775450000},"saved_comment_count":0,"breakdown":{"total_score":7.0},"rank":2}]'
                ),
                encoding="utf-8",
            )
            (run_dir / "comments").mkdir(parents=True, exist_ok=True)
            (run_dir / "comments" / "strong.json").write_text(
                (
                    '{"submission_id":"strong","subreddit":"Entrepreneur","title":"x","fetched_comment_count":2,'
                    '"comments":[{"id":"c1","body":"We still do this manually and it slows everything down.","score":5,"depth":0},'
                    '{"id":"c2","body":"Same issue here, we keep missing people because the workflow breaks.","score":4,"depth":0}]}'
                ),
                encoding="utf-8",
            )

            posts = load_reply_source_posts(run_dir)

        self.assertEqual(posts[0].candidate.id, "strong")

    def test_load_summary_posts_prefers_strongest_cluster_over_selected_posts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "candidate_posts.json").write_text(
                '[{"id":"raw1","title":"Raw post","subreddit":"Entrepreneur","url":"https://reddit.com/raw"}]\n',
                encoding="utf-8",
            )
            (run_dir / "selected_posts.json").write_text(
                (
                    '[{"candidate":{"id":"theme1","title":"Cluster winner","subreddit":"Entrepreneur",'
                    '"url":"https://reddit.com/theme1"},"saved_comment_count":0,'
                    '"breakdown":{"total_score":9.0},"rank":1},'
                    '{"candidate":{"id":"sel1","title":"Selected post","subreddit":"Entrepreneur",'
                    '"url":"https://reddit.com/selected"},"saved_comment_count":0,'
                    '"breakdown":{"total_score":8.0},"rank":2}]'
                ),
                encoding="utf-8",
            )
            (run_dir / "theme_summary.json").write_text(
                (
                    '{"run_dir":"x","generated_at":"2026-04-02T00:00:00Z","source_post_count":2,'
                    '"cluster_count":1,"strongest_cluster_id":"cluster-1","strongest_post_ids":["theme1"],'
                    '"clusters":[{"cluster_id":"cluster-1","label":"crm / spreadsheet","post_ids":["theme1"],"size":1}]}'
                ),
                encoding="utf-8",
            )
            posts = load_summary_posts(run_dir)

        self.assertEqual(posts[0].id, "theme1")

    def test_score_comment_for_evidence_prefers_detailed_comments(self) -> None:
        weak = score_comment_for_evidence(Comment(id="c1", body="same here", score=1, depth=0))
        strong = score_comment_for_evidence(
            Comment(
                id="c2",
                body="I still track every lead manually in a spreadsheet because CRM setup feels too heavy.",
                score=7,
                depth=1,
            )
        )
        self.assertGreater(strong.total_score, weak.total_score)

    def test_select_comment_evidence_limits_comments_per_post(self) -> None:
        post = CandidatePost(
            id="abc123",
            title="Painful workflow",
            subreddit="Entrepreneur",
            url="https://reddit.com/example",
        )
        selected = select_comment_evidence(
            [post],
            {
                "abc123": [
                    Comment(id="c1", body="same here", score=1, depth=0),
                    Comment(
                        id="c2",
                        body="I still track every lead manually in a spreadsheet because CRM setup feels too heavy.",
                        score=7,
                        depth=1,
                    ),
                    Comment(
                        id="c3",
                        body="We built a reminder sheet and still miss follow-up every Friday.",
                        score=5,
                        depth=0,
                    ),
                ]
            },
            max_posts=1,
            max_comments_per_post=2,
        )
        self.assertEqual([item.comment_id for item in selected["abc123"]], ["c2", "c3"])

    def test_draft_reply_suggestions_writes_reply_artifacts(self) -> None:
        class FakeLMStudioClient:
            async def generate_response(self, prompt, model=None):
                if "Evaluate the reply drafts below" in prompt:
                    payload = {
                        "provider": "lmstudio",
                        "model": model or "openai/gpt-oss-20b",
                        "output_text": json.dumps(
                            {
                                "evaluations": [
                                    {
                                        "post_id": "abc123",
                                        "relevance_score": 4,
                                        "conversation_value_score": 4,
                                        "voice_match_score": 4,
                                        "reddit_friendliness_score": 5,
                                        "feedback": "Make the take slightly sharper.",
                                    }
                                ]
                            }
                        ),
                        "raw_response": {"id": "resp_eval_1"},
                    }
                elif "Revise the reply drafts below" in prompt:
                    payload = {
                        "provider": "lmstudio",
                        "model": model or "openai/gpt-oss-20b",
                        "output_text": (
                            "## Post 1\n"
                            "post_id: abc123\n"
                            "reply: I can see why this is frustrating. The spreadsheet fallback usually means the process never became trustworthy in the first place.\n\n"
                            "What stands out to me is that people usually do not need more steps here, they need a workflow they will actually trust when things get busy.\n"
                        ),
                        "raw_response": {"id": "resp_reply_2"},
                    }
                else:
                    payload = {
                        "provider": "lmstudio",
                        "model": model or "openai/gpt-oss-20b",
                        "output_text": (
                            "## Post 1\n"
                            "post_id: abc123\n"
                            "reply: I can see why this is frustrating.\n"
                        ),
                        "raw_response": {"id": "resp_reply_1"},
                    }
                return type("Generation", (), payload)()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "selected_posts.json").write_text(
                (
                    '[{"candidate":{"id":"abc123","title":"Manual follow-up is eating my week",'
                    '"subreddit":"Entrepreneur","url":"https://reddit.com/example","selftext":"Still using spreadsheets."},'
                    '"saved_comment_count":0,"breakdown":{"total_score":7.0},"rank":1}]'
                ),
                encoding="utf-8",
            )
            artifact = asyncio.run(
                draft_reply_suggestions(
                    run_dir,
                    FakeLMStudioClient(),
                    voice="plainspoken founder",
                    max_posts=1,
                )
            )

            self.assertEqual(artifact.selected_post_count, 1)
            self.assertEqual(artifact.improvement_rounds, 0)
            self.assertTrue(artifact.passed_threshold)
            self.assertTrue((run_dir / "reply_drafts.json").exists())
            self.assertTrue((run_dir / "reply_drafts.md").exists())
            self.assertTrue((run_dir / "prompts" / "reply-drafts.txt").exists())
            self.assertTrue((run_dir / "raw" / "llm" / "reply-drafts.json").exists())
            self.assertTrue((run_dir / "prompts" / "reply-drafts-initial.txt").exists())
            self.assertTrue((run_dir / "raw" / "llm" / "reply-drafts-initial.json").exists())
            self.assertTrue((run_dir / "prompts" / "reply-drafts-evaluation-round-00.txt").exists())
            self.assertTrue((run_dir / "raw" / "llm" / "reply-drafts-evaluation-round-00.json").exists())

    def test_draft_reply_suggestions_normalizes_reply_to_three_plain_paragraphs_max(self) -> None:
        class FakeLMStudioClient:
            async def generate_response(self, prompt, model=None):
                if "Evaluate the reply drafts below" in prompt:
                    payload = {
                        "provider": "lmstudio",
                        "model": model or "openai/gpt-oss-20b",
                        "output_text": json.dumps(
                            {
                                "evaluations": [
                                    {
                                        "post_id": "abc123",
                                        "relevance_score": 4,
                                        "conversation_value_score": 4,
                                        "voice_match_score": 4,
                                        "reddit_friendliness_score": 4,
                                        "feedback": "Looks good.",
                                    }
                                ]
                            }
                        ),
                        "raw_response": {"id": "resp_eval_norm"},
                    }
                else:
                    payload = {
                        "provider": "lmstudio",
                        "model": model or "openai/gpt-oss-20b",
                        "output_text": (
                            "## Post 1\n"
                            "post_id: abc123\n"
                            "reply: # Heading\n"
                            "- first point\n\n"
                            "Second paragraph.\n\n"
                            "Third paragraph.\n\n"
                            "Fourth paragraph.\n"
                        ),
                        "raw_response": {"id": "resp_reply_norm"},
                    }
                return type("Generation", (), payload)()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "selected_posts.json").write_text(
                (
                    '[{"candidate":{"id":"abc123","title":"Manual follow-up is eating my week",'
                    '"subreddit":"Entrepreneur","url":"https://reddit.com/example","selftext":"Still using spreadsheets."},'
                    '"saved_comment_count":0,"breakdown":{"total_score":7.0},"rank":1}]'
                ),
                encoding="utf-8",
            )
            artifact = asyncio.run(
                draft_reply_suggestions(
                    run_dir,
                    FakeLMStudioClient(),
                    voice="plainspoken founder",
                    max_posts=1,
                )
            )

        self.assertNotIn("# Heading", artifact.drafts[0].reply_text)
        self.assertNotIn("- first point", artifact.drafts[0].reply_text)
        self.assertEqual(len([p for p in artifact.drafts[0].reply_text.split("\n\n") if p.strip()]), 3)

    def test_draft_reply_suggestions_revises_until_threshold_or_limit(self) -> None:
        class FakeLMStudioClient:
            def __init__(self) -> None:
                self.revision_count = 0

            async def generate_response(self, prompt, model=None):
                if "Evaluate the reply drafts below" in prompt and self.revision_count == 0:
                    payload = {
                        "provider": "lmstudio",
                        "model": model or "openai/gpt-oss-20b",
                        "output_text": json.dumps(
                            {
                                "evaluations": [
                                    {
                                        "post_id": "abc123",
                                        "relevance_score": 2,
                                        "conversation_value_score": 2,
                                        "voice_match_score": 3,
                                        "reddit_friendliness_score": 4,
                                        "feedback": "Be more specific to the post and add a clearer take.",
                                    }
                                ]
                            }
                        ),
                        "raw_response": {"id": "resp_eval_fail"},
                    }
                elif "Revise the reply drafts below" in prompt:
                    self.revision_count += 1
                    payload = {
                        "provider": "lmstudio",
                        "model": model or "openai/gpt-oss-20b",
                        "output_text": (
                            "## Post 1\n"
                            "post_id: abc123\n"
                            "reply: I think the trust issue is the real problem here, not just the spreadsheet itself.\n\n"
                            "Once people start keeping backup habits around a workflow, the tool is already losing because nobody trusts it when the day gets messy.\n"
                        ),
                        "raw_response": {"id": "resp_revision_pass"},
                    }
                elif "Evaluate the reply drafts below" in prompt and self.revision_count == 1:
                    payload = {
                        "provider": "lmstudio",
                        "model": model or "openai/gpt-oss-20b",
                        "output_text": json.dumps(
                            {
                                "evaluations": [
                                    {
                                        "post_id": "abc123",
                                        "relevance_score": 4,
                                        "conversation_value_score": 4,
                                        "voice_match_score": 4,
                                        "reddit_friendliness_score": 4,
                                        "feedback": "Good.",
                                    }
                                ]
                            }
                        ),
                        "raw_response": {"id": "resp_eval_pass"},
                    }
                else:
                    payload = {
                        "provider": "lmstudio",
                        "model": model or "openai/gpt-oss-20b",
                        "output_text": (
                            "## Post 1\n"
                            "post_id: abc123\n"
                            "reply: I can see why this is frustrating.\n"
                        ),
                        "raw_response": {"id": "resp_initial"},
                    }
                return type("Generation", (), payload)()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "selected_posts.json").write_text(
                (
                    '[{"candidate":{"id":"abc123","title":"Manual follow-up is eating my week",'
                    '"subreddit":"Entrepreneur","url":"https://reddit.com/example","selftext":"Still using spreadsheets."},'
                    '"saved_comment_count":0,"breakdown":{"total_score":7.0},"rank":1}]'
                ),
                encoding="utf-8",
            )
            artifact = asyncio.run(
                draft_reply_suggestions(
                    run_dir,
                    FakeLMStudioClient(),
                    voice="plainspoken founder",
                    max_posts=1,
                    max_improvement_rounds=2,
                )
            )
            self.assertTrue((run_dir / "prompts" / "reply-drafts-revision-round-01.txt").exists())

        self.assertEqual(artifact.improvement_rounds, 1)
        self.assertTrue(artifact.passed_threshold)
        self.assertEqual(artifact.drafts[0].passed_threshold, True)

    def test_score_comment_opportunities_writes_ranked_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "selected_posts.json").write_text(
                (
                    '[{"candidate":{"id":"weak","title":"What tools do people use for outreach",'
                    '"subreddit":"Entrepreneur","url":"https://reddit.com/weak","selftext":"Curious what people use.",'
                    '"source_queries":["manual follow-up pain"],"created_utc":1700000000},"saved_comment_count":0,"breakdown":{"total_score":9.0},"rank":1},'
                    '{"candidate":{"id":"strong","title":"We still miss leads because manual follow-up is killing our workflow",'
                    '"subreddit":"Entrepreneur","url":"https://reddit.com/strong","selftext":"I need to fix this now.",'
                    '"source_queries":["manual follow-up pain"],"created_utc":1775450000},"saved_comment_count":0,"breakdown":{"total_score":7.0},"rank":2}]'
                ),
                encoding="utf-8",
            )
            (run_dir / "comments").mkdir(parents=True, exist_ok=True)
            (run_dir / "comments" / "strong.json").write_text(
                (
                    '{"submission_id":"strong","subreddit":"Entrepreneur","title":"x","fetched_comment_count":2,'
                    '"comments":[{"id":"c1","body":"We still do this manually and it slows everything down.","score":5,"depth":0},'
                    '{"id":"c2","body":"Same issue here, we keep missing people because the workflow breaks.","score":4,"depth":0}]}'
                ),
                encoding="utf-8",
            )

            artifact = score_comment_opportunities(run_dir, max_posts=2)

            self.assertEqual(artifact.scored_post_count, 2)
            self.assertEqual(artifact.opportunities[0].post_id, "strong")
            self.assertTrue((run_dir / "comment_opportunities.json").exists())
            self.assertIn(artifact.opportunities[0].bucket, {"high_value", "watchlist"})


class RankingTests(unittest.TestCase):
    def test_comment_screening_detects_non_trivial_and_complaint_signals(self) -> None:
        comments = [
            Comment(id="c1", body="same here", score=1, depth=0),
            Comment(
                id="c2",
                body="I still track every lead manually in a spreadsheet because CRM setup feels too heavy.",
                score=7,
                depth=1,
            ),
            Comment(
                id="c3",
                body="We built a reminder sheet and still miss follow-up every Friday.",
                score=5,
                depth=0,
            ),
        ]
        breakdown = analyze_comment_screening(comments)

        self.assertFalse(is_non_trivial_comment(comments[0]))
        self.assertTrue(is_non_trivial_comment(comments[1]))
        self.assertTrue(has_complaint_signal(comments[1]))
        self.assertEqual(breakdown.saved_comment_count, 3)
        self.assertEqual(breakdown.non_trivial_comment_count, 2)
        self.assertEqual(breakdown.complaint_signal_comment_count, 2)

    def test_score_candidate_post_prefers_engaged_relevant_posts(self) -> None:
        now = datetime(2026, 4, 2, tzinfo=UTC)
        weak = score_candidate_post(
            CandidatePost(
                id="weak",
                title="Need help",
                subreddit="Entrepreneur",
                url="https://reddit.com/weak",
                score=1,
                num_comments=1,
                selftext="",
                source_queries=["manual follow-up pain"],
                created_utc=1700000000,
            ),
            saved_comments=[],
            now=now,
        )
        strong = score_candidate_post(
            CandidatePost(
                id="strong",
                title="Manual follow-up is eating my week",
                subreddit="Entrepreneur",
                url="https://reddit.com/strong",
                score=40,
                num_comments=25,
                selftext="We still track every lead manually in a spreadsheet.",
                source_queries=["manual follow-up pain"],
                created_utc=1712100000,
            ),
            saved_comments=[Comment(id="c1", body="Exact same issue", score=6, depth=0)],
            now=now,
        )
        self.assertGreater(strong.total_score, weak.total_score)

    def test_score_thread_comment_opportunity_prefers_recent_painful_confirmed_threads(self) -> None:
        now = datetime(2026, 4, 7, tzinfo=UTC)
        weak = score_thread_comment_opportunity(
            CandidatePost(
                id="weak",
                title="What tools are people using for outreach",
                subreddit="Entrepreneur",
                url="https://reddit.com/weak",
                selftext="Curious what everyone likes.",
                source_queries=["manual follow-up pain"],
                created_utc=1700000000,
            ),
            saved_comments=[],
            now=now,
        )
        strong = score_thread_comment_opportunity(
            CandidatePost(
                id="strong",
                title="We still miss leads because manual follow-up is killing our workflow",
                subreddit="Entrepreneur",
                url="https://reddit.com/strong",
                selftext="I need to fix this now because the process does not scale.",
                source_queries=["manual follow-up pain"],
                created_utc=1775450000,
                num_comments=8,
            ),
            saved_comments=[
                Comment(
                    id="c1",
                    body="We still do this manually and it slows everything down every week.",
                    score=5,
                    depth=0,
                ),
                Comment(
                    id="c2",
                    body="Same issue here, we keep missing people because the workflow breaks when things get busy.",
                    score=4,
                    depth=0,
                ),
            ],
            now=now,
        )

        self.assertGreater(strong.total_score, weak.total_score)
        self.assertEqual(strong.recommendation, "high_value")
        self.assertIn(weak.recommendation, {"ignore", "watchlist"})

    def test_rank_candidates_sorts_and_assigns_ranks(self) -> None:
        ranked = rank_candidates(
            [
                CandidatePost(
                    id="b",
                    title="Manual follow-up is painful",
                    subreddit="Entrepreneur",
                    url="https://reddit.com/b",
                    score=20,
                    num_comments=10,
                    selftext="Still using spreadsheets",
                    source_queries=["manual follow-up"],
                ),
                CandidatePost(
                    id="a",
                    title="Need help",
                    subreddit="Entrepreneur",
                    url="https://reddit.com/a",
                    score=1,
                    num_comments=1,
                    selftext="",
                    source_queries=["manual follow-up"],
                ),
            ],
            comments_by_submission={"b": [Comment(id="c1", body="same problem", score=4, depth=0)]},
            now=datetime(2026, 4, 2, tzinfo=UTC),
        )
        self.assertEqual([item.candidate.id for item in ranked], ["b", "a"])
        self.assertEqual([item.rank for item in ranked], [1, 2])

    def test_rank_run_candidates_writes_selected_posts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "candidate_posts.json").write_text(
                (
                    '[{"id":"a","title":"Need help","subreddit":"Entrepreneur","url":"https://reddit.com/a","score":1,"num_comments":1,"source_queries":["manual follow-up"]},'
                    '{"id":"b","title":"Manual follow-up is painful","subreddit":"Entrepreneur","url":"https://reddit.com/b","score":20,"num_comments":10,"selftext":"Still using spreadsheets","source_queries":["manual follow-up"]}]'
                ),
                encoding="utf-8",
            )
            result = rank_run_candidates(
                run_dir,
                max_selected_posts=1,
                now=datetime(2026, 4, 2, tzinfo=UTC),
            )

            self.assertEqual(result.selected_count, 1)
            self.assertTrue((run_dir / "candidate_screening.json").exists())
            self.assertTrue((run_dir / "post_ranking.json").exists())
            self.assertTrue((run_dir / "selected_posts.json").exists())
            self.assertEqual(load_selected_posts(run_dir)[0].id, "b")

    def test_rank_run_candidates_filters_by_discussion_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "candidate_posts.json").write_text(
                (
                    '[{"id":"a","title":"Need help","subreddit":"Entrepreneur","url":"https://reddit.com/a","score":30,"num_comments":14,"source_queries":["manual follow-up"]},'
                    '{"id":"b","title":"Manual follow-up is painful","subreddit":"Entrepreneur","url":"https://reddit.com/b","score":20,"num_comments":10,"selftext":"Still using spreadsheets","source_queries":["manual follow-up"]}]'
                ),
                encoding="utf-8",
            )
            comments_dir = run_dir / "comments"
            comments_dir.mkdir(parents=True, exist_ok=True)
            (comments_dir / "a.json").write_text(
                (
                    '{"submission_id":"a","subreddit":"Entrepreneur","title":"Need help","fetched_comment_count":2,'
                    '"comments":[{"id":"a1","body":"same here","score":1,"depth":0},'
                    '{"id":"a2","body":"this","score":1,"depth":0}]}'
                ),
                encoding="utf-8",
            )
            (comments_dir / "b.json").write_text(
                (
                    '{"submission_id":"b","subreddit":"Entrepreneur","title":"Manual follow-up is painful","fetched_comment_count":2,'
                    '"comments":[{"id":"b1","body":"I still track every lead manually in a spreadsheet because CRM setup feels too heavy.","score":7,"depth":1},'
                    '{"id":"b2","body":"We built a reminder sheet and still miss follow-up every Friday.","score":5,"depth":0}]}'
                ),
                encoding="utf-8",
            )

            result = rank_run_candidates(
                run_dir,
                max_selected_posts=5,
                min_non_trivial_comments=1,
                min_complaint_signal_comments=1,
                now=datetime(2026, 4, 2, tzinfo=UTC),
            )

            self.assertEqual(result.candidate_count, 2)
            self.assertEqual(result.screened_candidate_count, 1)
            self.assertEqual(result.rejected_candidate_count, 1)
            self.assertEqual(result.rejection_counts, {"low_non_trivial_comments": 1})
            self.assertEqual(result.selected_count, 1)
            self.assertEqual(load_selected_posts(run_dir)[0].id, "b")
            screening_payload = json.loads((run_dir / "candidate_screening.json").read_text(encoding="utf-8"))
            self.assertEqual(screening_payload["kept_count"], 1)
            self.assertEqual(screening_payload["rejected_count"], 1)
            self.assertEqual(screening_payload["rejection_counts"], {"low_non_trivial_comments": 1})
            self.assertEqual(
                screening_payload["decisions"][0]["rejection_reason"],
                "low_non_trivial_comments",
            )


class ClusteringTests(unittest.TestCase):
    def test_validate_cluster_evidence_counts_complaint_signal_posts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "theme_summary.json").write_text(
                (
                    '{"run_dir":"x","generated_at":"2026-04-02T00:00:00Z","source_post_count":3,'
                    '"cluster_count":1,"strongest_cluster_id":"cluster-1","strongest_post_ids":["a","b","c"],'
                    '"clusters":[{"cluster_id":"cluster-1","label":"crm / follow-up","post_ids":["a","b","c"],"size":3}]}'
                ),
                encoding="utf-8",
            )
            (run_dir / "candidate_screening.json").write_text(
                (
                    '{"run_dir":"x","generated_at":"2026-04-02T00:00:00Z","candidate_count":3,'
                    '"kept_count":3,"rejected_count":0,"min_non_trivial_comments":1,'
                    '"min_complaint_signal_comments":1,"rejection_counts":{},'
                    '"decisions":['
                    '{"candidate":{"id":"a","title":"A","subreddit":"Entrepreneur","url":"https://reddit.com/a"},"kept":true,"rejection_reason":null,'
                    '"breakdown":{"saved_comment_count":2,"non_trivial_comment_count":2,"complaint_signal_comment_count":1}},'
                    '{"candidate":{"id":"b","title":"B","subreddit":"Entrepreneur","url":"https://reddit.com/b"},"kept":true,"rejection_reason":null,'
                    '"breakdown":{"saved_comment_count":2,"non_trivial_comment_count":2,"complaint_signal_comment_count":1}},'
                    '{"candidate":{"id":"c","title":"C","subreddit":"Entrepreneur","url":"https://reddit.com/c"},"kept":true,"rejection_reason":null,'
                    '"breakdown":{"saved_comment_count":1,"non_trivial_comment_count":1,"complaint_signal_comment_count":0}}]}'
                ),
                encoding="utf-8",
            )

            artifact = validate_cluster_evidence(
                run_dir=run_dir,
                min_cluster_complaint_posts=2,
                generated_at=datetime(2026, 4, 2, tzinfo=UTC),
            )

        self.assertTrue(artifact.passes)
        self.assertEqual(artifact.screened_cluster_post_count, 3)
        self.assertEqual(artifact.complaint_signal_post_count, 2)

    def test_cluster_ranked_posts_groups_similar_posts(self) -> None:
        ranked_posts = [
            RankedCandidatePost(
                candidate=CandidatePost(
                    id="a",
                    title="Manual follow-up in spreadsheets is painful",
                    subreddit="Entrepreneur",
                    url="https://reddit.com/a",
                    selftext="CRM setup feels too heavy",
                    source_queries=["manual follow-up pain"],
                ),
                saved_comment_count=2,
                breakdown={"total_score": 8.0},
                rank=1,
            ),
            RankedCandidatePost(
                candidate=CandidatePost(
                    id="b",
                    title="Spreadsheet lead tracking keeps breaking follow-up",
                    subreddit="Entrepreneur",
                    url="https://reddit.com/b",
                    selftext="Still doing manual reminders every week",
                    source_queries=["manual follow-up pain"],
                ),
                saved_comment_count=1,
                breakdown={"total_score": 7.5},
                rank=2,
            ),
            RankedCandidatePost(
                candidate=CandidatePost(
                    id="c",
                    title="Laptop desk setup hurts my wrists",
                    subreddit="vibecoding",
                    url="https://reddit.com/c",
                    selftext="Need better ergonomics",
                    source_queries=["ergonomic pain"],
                ),
                saved_comment_count=0,
                breakdown={"total_score": 6.0},
                rank=3,
            ),
        ]
        clusters = cluster_ranked_posts(ranked_posts, similarity_threshold=0.12, min_shared_terms=2)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0].size, 2)
        self.assertEqual(set(clusters[0].post_ids), {"a", "b"})

    def test_cluster_run_posts_writes_theme_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "candidate_screening.json").write_text(
                (
                    '{"run_dir":"x","generated_at":"2026-04-02T00:00:00Z","candidate_count":2,'
                    '"kept_count":2,"rejected_count":0,"min_non_trivial_comments":0,'
                    '"min_complaint_signal_comments":0,"rejection_counts":{},'
                    '"decisions":['
                    '{"candidate":{"id":"a","title":"Manual follow-up in spreadsheets is painful","subreddit":"Entrepreneur","url":"https://reddit.com/a"},"kept":true,"rejection_reason":null,'
                    '"breakdown":{"saved_comment_count":2,"non_trivial_comment_count":2,"complaint_signal_comment_count":1}},'
                    '{"candidate":{"id":"b","title":"Spreadsheet lead tracking keeps breaking follow-up","subreddit":"Entrepreneur","url":"https://reddit.com/b"},"kept":true,"rejection_reason":null,'
                    '"breakdown":{"saved_comment_count":1,"non_trivial_comment_count":1,"complaint_signal_comment_count":1}}]}'
                ),
                encoding="utf-8",
            )
            (run_dir / "selected_posts.json").write_text(
                (
                    '[{"candidate":{"id":"a","title":"Manual follow-up in spreadsheets is painful","subreddit":"Entrepreneur","url":"https://reddit.com/a","selftext":"CRM setup feels too heavy","source_queries":["manual follow-up pain"]},"saved_comment_count":2,"breakdown":{"total_score":8.0},"rank":1},'
                    '{"candidate":{"id":"b","title":"Spreadsheet lead tracking keeps breaking follow-up","subreddit":"Entrepreneur","url":"https://reddit.com/b","selftext":"Still doing manual reminders every week","source_queries":["manual follow-up pain"]},"saved_comment_count":1,"breakdown":{"total_score":7.5},"rank":2}]'
                ),
                encoding="utf-8",
            )
            result = cluster_run_posts(
                run_dir,
                similarity_threshold=0.12,
                min_shared_terms=2,
                min_cluster_complaint_posts=2,
            )
            self.assertEqual(result.cluster_count, 1)
            self.assertTrue((run_dir / "theme_summary.json").exists())
            self.assertTrue((run_dir / "cluster_evidence_validation.json").exists())
            validation = load_cluster_evidence_validation(run_dir)
            self.assertIsNotNone(validation)
            self.assertTrue(validation.passes)  # type: ignore[union-attr]
            strongest = load_strongest_cluster_posts(run_dir)
            self.assertEqual([post.id for post in strongest], ["a", "b"])


class MemoWriterTests(unittest.TestCase):
    def test_write_final_memo_writes_artifacts(self) -> None:
        class FakeLMStudioClient:
            async def generate_response(self, prompt, model=None):
                return type(
                    "Generation",
                    (),
                    {
                        "provider": "lmstudio",
                        "model": model or "openai/gpt-oss-20b",
                        "prompt": prompt,
                        "output_text": (
                            "# Executive Summary\n\nThe strongest cluster is real.\n\n"
                            "## Research Takeaways\n\nSignal is repeated.\n\n"
                            "## Top 5 Product Ideas\n\n1. Idea one\n2. Idea two\n3. Idea three\n4. Idea four\n5. Idea five\n\n"
                            "## Best Single Bet\n\nBuild the follow-up system.\n\n"
                            "## 10 Content Hooks\n\n"
                            "1. Hook 1\n2. Hook 2\n3. Hook 3\n4. Hook 4\n5. Hook 5\n6. Hook 6\n7. Hook 7\n8. Hook 8\n9. Hook 9\n10. Hook 10\n\n"
                            "## Risks / Caveats\n\nEvidence is still Reddit-native."
                        ),
                        "raw_response": {"id": "resp_final_memo"},
                    },
                )()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "selected_posts.json").write_text(
                (
                    '[{"candidate":{"id":"a","title":"Manual follow-up in spreadsheets is painful","subreddit":"Entrepreneur","url":"https://reddit.com/a","selftext":"CRM setup feels too heavy","score":30,"num_comments":11},"saved_comment_count":2,"breakdown":{"total_score":8.0},"rank":1},'
                    '{"candidate":{"id":"b","title":"Spreadsheet lead tracking keeps breaking follow-up","subreddit":"Entrepreneur","url":"https://reddit.com/b","selftext":"Still doing manual reminders every week","score":28,"num_comments":10},"saved_comment_count":1,"breakdown":{"total_score":7.8},"rank":2},'
                    '{"candidate":{"id":"c","title":"CRM follow-up falls apart without manual reminders","subreddit":"Entrepreneur","url":"https://reddit.com/c","selftext":"We keep missing replies","score":24,"num_comments":8},"saved_comment_count":1,"breakdown":{"total_score":7.4},"rank":3},'
                    '{"candidate":{"id":"d","title":"Manual lead follow-up is killing our process","subreddit":"Entrepreneur","url":"https://reddit.com/d","selftext":"Everything lives in a spreadsheet","score":26,"num_comments":9},"saved_comment_count":1,"breakdown":{"total_score":7.3},"rank":4},'
                    '{"candidate":{"id":"e","title":"Client follow-up still depends on spreadsheets and CRM hacks","subreddit":"Entrepreneur","url":"https://reddit.com/e","selftext":"Too much copy and paste","score":23,"num_comments":7},"saved_comment_count":1,"breakdown":{"total_score":7.1},"rank":5}]'
                ),
                encoding="utf-8",
            )
            (run_dir / "theme_summary.json").write_text(
                (
                    '{"run_dir":"x","generated_at":"2026-04-02T00:00:00Z","source_post_count":5,'
                    '"cluster_count":1,"strongest_cluster_id":"cluster-1","strongest_post_ids":["a","b","c","d","e"],'
                    '"clusters":[{"cluster_id":"cluster-1","label":"crm / follow-up / spreadsheet","post_ids":["a","b","c","d","e"],'
                    '"size":5,"average_post_score":7.52,"total_comment_count":45,'
                    '"top_terms":["crm","follow-up","spreadsheet"],"member_ranks":[1,2,3,4,5],"cohesion_score":0.34}]}'
                ),
                encoding="utf-8",
            )
            (run_dir / "evidence_summary.json").write_text(
                (
                    '{"run_dir":"x","generated_at":"2026-04-02T00:00:00Z","provider":"lmstudio",'
                    '"model":"openai/gpt-oss-20b","candidate_count":5,"comment_count":12,"selected_comment_count":6,'
                    '"max_posts_used":5,"prompt_artifact_path":"prompts/candidate-evidence-summary.txt",'
                    '"raw_response_artifact_path":"raw/llm/candidate-evidence-summary.json",'
                    '"summary_markdown_artifact_path":"evidence_summary.md",'
                    '"summary_text":"## Repeated Complaints\\n\\nManual follow-up and spreadsheet-heavy CRM workflows keep failing."}'
                ),
                encoding="utf-8",
            )
            (run_dir / "manifest.json").write_text(
                (
                    '{"run_slug":"run-1","status":"completed","started_at":"2026-04-02T00:00:00Z",'
                    '"completed_at":"2026-04-02T00:10:00Z","output_dir":"x","topic":"manual follow-up pain",'
                    '"target_audience":"founder-led sales teams","category":"business","time_horizon":"recent",'
                    '"subreddits":["Entrepreneur"],"queries":["manual follow-up pain"],"sort":"relevance",'
                    '"time_filter":"month","limit":25,"request_timeout_seconds":30.0,"max_retries":3,"max_concurrent_requests":4}'
                ),
                encoding="utf-8",
            )

            artifact = asyncio.run(
                write_final_memo(
                    run_dir=run_dir,
                    client=FakeLMStudioClient(),
                    min_cluster_posts=5,
                    max_posts=5,
                )
            )

            self.assertEqual(artifact.strongest_cluster_id, "cluster-1")
            self.assertEqual(artifact.topic, "manual follow-up pain")
            self.assertEqual(artifact.target_audience, "founder-led sales teams")
            self.assertEqual(artifact.source_thread_urls[0], "https://reddit.com/a")
            self.assertTrue((run_dir / "final_memo.json").exists())
            self.assertTrue((run_dir / "final_memo.md").exists())
            self.assertTrue((run_dir / "prompts" / "final-memo.txt").exists())
            self.assertTrue((run_dir / "raw" / "llm" / "final-memo.json").exists())
            final_markdown = (run_dir / "final_memo.md").read_text(encoding="utf-8")
            self.assertIn("## Source Threads", final_markdown)
            self.assertIn("https://reddit.com/e", final_markdown)

    def test_write_final_memo_fails_for_weak_cluster(self) -> None:
        class FakeLMStudioClient:
            async def generate_response(self, prompt, model=None):
                raise AssertionError("should not be called")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "selected_posts.json").write_text(
                (
                    '[{"candidate":{"id":"a","title":"Manual follow-up is annoying","subreddit":"Entrepreneur","url":"https://reddit.com/a"},"saved_comment_count":0,"breakdown":{"total_score":6.0},"rank":1},'
                    '{"candidate":{"id":"b","title":"CRM follow-up is still manual","subreddit":"Entrepreneur","url":"https://reddit.com/b"},"saved_comment_count":0,"breakdown":{"total_score":5.9},"rank":2}]'
                ),
                encoding="utf-8",
            )
            (run_dir / "theme_summary.json").write_text(
                (
                    '{"run_dir":"x","generated_at":"2026-04-02T00:00:00Z","source_post_count":2,'
                    '"cluster_count":1,"strongest_cluster_id":"cluster-1","strongest_post_ids":["a","b"],'
                    '"clusters":[{"cluster_id":"cluster-1","label":"crm / follow-up","post_ids":["a","b"],'
                    '"size":2,"average_post_score":5.95,"total_comment_count":8,'
                    '"top_terms":["crm","follow-up"],"member_ranks":[1,2],"cohesion_score":0.21}]}'
                ),
                encoding="utf-8",
            )
            (run_dir / "evidence_summary.json").write_text(
                (
                    '{"run_dir":"x","generated_at":"2026-04-02T00:00:00Z","provider":"lmstudio",'
                    '"model":"openai/gpt-oss-20b","candidate_count":2,"comment_count":2,"selected_comment_count":1,'
                    '"max_posts_used":2,"prompt_artifact_path":"prompts/candidate-evidence-summary.txt",'
                    '"raw_response_artifact_path":"raw/llm/candidate-evidence-summary.json",'
                    '"summary_markdown_artifact_path":"evidence_summary.md",'
                    '"summary_text":"## Repeated Complaints\\n\\nSome pain is present."}'
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "requires at least 5"):
                asyncio.run(
                    write_final_memo(
                        run_dir=run_dir,
                        client=FakeLMStudioClient(),
                        min_cluster_posts=5,
                    )
                )


class CliTests(unittest.TestCase):
    def test_capture_cli_prints_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_result = AsyncMock()
            fake_result.output_path = Path(tmpdir) / "capture.json"
            fake_result.log_path = Path(tmpdir) / "capture.log"
            fake_result.snapshot_dir = Path(tmpdir) / "capture"
            fake_result.search_url_count = 2
            fake_result.discovered_thread_count = 7
            fake_result.selected_thread_count = 3
            fake_result.captured_post_count = 3
            fake_result.captured_comment_count = 11
            fake_result.html_snapshot_count = 5
            fake_result.screenshot_count = 5
            fake_result.page_error_count = 1
            fake_result.selected_thread_urls = [
                "https://www.reddit.com/r/Entrepreneur/comments/a/example",
                "https://www.reddit.com/r/Entrepreneur/comments/b/example",
            ]
            with patch(
                "reddit_pain_agent.main.capture_reddit_threads",
                AsyncMock(return_value=fake_result),
            ):
                with patch("sys.stdout.write") as stdout_write:
                    exit_code = main(
                        [
                            "capture",
                            "--subreddit",
                            "Entrepreneur",
                            "--query",
                            "manual follow-up pain",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        output = "".join(call.args[0] for call in stdout_write.call_args_list)
        self.assertIn("capture_json:", output)
        self.assertIn("capture_log:", output)
        self.assertIn("capture_snapshots:", output)
        self.assertIn("search_urls: 2", output)
        self.assertIn("discovered_threads: 7", output)
        self.assertIn("captured_comments: 11", output)
        self.assertIn("html_snapshots: 5", output)
        self.assertIn("screenshots: 5", output)
        self.assertIn("page_errors: 1", output)
        self.assertIn("selected_thread_urls:", output)

    def test_capture_cli_handoff_run_builds_manual_input_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            capture_json = Path(tmpdir) / "capture.json"
            fake_result = AsyncMock()
            fake_result.output_path = capture_json
            fake_result.log_path = Path(tmpdir) / "capture.log"
            fake_result.snapshot_dir = Path(tmpdir) / "capture"
            fake_result.search_url_count = 1
            fake_result.discovered_thread_count = 3
            fake_result.selected_thread_count = 2
            fake_result.captured_post_count = 2
            fake_result.captured_comment_count = 4
            fake_result.html_snapshot_count = 3
            fake_result.screenshot_count = 3
            fake_result.page_error_count = 0
            fake_result.selected_thread_urls = []

            with patch(
                "reddit_pain_agent.main.capture_reddit_threads",
                AsyncMock(return_value=fake_result),
            ), patch(
                "reddit_pain_agent.main._run_handoff_command",
                return_value=0,
            ) as handoff_mock, patch("sys.stdout.write") as stdout_write:
                exit_code = main(
                    [
                        "capture",
                        "--subreddit",
                        "Entrepreneur",
                        "--query",
                        "manual follow-up pain",
                        "--handoff",
                        "run",
                        "--output-dir",
                        str(Path(tmpdir) / "run-1"),
                        "--model",
                        "openai/gpt-oss-20b",
                    ]
                )

        self.assertEqual(exit_code, 0)
        handoff_argv = handoff_mock.call_args.args[0]
        self.assertEqual(handoff_argv[:4], ["run", "--manual-input", str(capture_json), "--output-dir"])
        self.assertIn("--model", handoff_argv)
        output = "".join(call.args[0] for call in stdout_write.call_args_list)
        self.assertIn("handoff: run", output)
        self.assertIn("handoff_command:", output)

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
                fake_result.query_variant_count = 3
                fake_result.search_spec_count = 6
                fake_result.sort_count = 2
                fake_result.time_filter_count = 1
                fake_result.pages_per_query = 2
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
        self.assertIn("query_variants: 3", output)
        self.assertIn("search_specs: 6", output)
        self.assertIn("sorts: 2", output)
        self.assertIn("time_filters: 1", output)
        self.assertIn("pages_per_query: 2", output)
        self.assertIn("candidate_posts: 5", output)
        self.assertIn("filtered: deleted=2, duplicate=1", output)

    def test_manual_import_cli_writes_summary_without_reddit_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "manual.json"
            _write_json_file(
                input_path,
                {
                    "posts": [
                        {
                            "id": "abc123",
                            "title": "Manual CRM follow-up is still painful",
                            "subreddit": "Entrepreneur",
                            "url": "https://reddit.com/abc123",
                            "score": 17,
                            "num_comments": 6,
                            "selftext": "Still using spreadsheets.",
                            "comments": [{"id": "c1", "body": "Same problem", "score": 4, "depth": 0}],
                        }
                    ]
                },
            )
            with patch.dict("os.environ", {}, clear=True):
                with patch("sys.stdout.write") as stdout_write:
                    exit_code = main(
                        [
                            "manual-import",
                            "--input",
                            str(input_path),
                            "--subreddit",
                            "Entrepreneur",
                            "--query",
                            "manual follow-up pain",
                            "--output-dir",
                            str(Path(tmpdir) / "run-manual"),
                        ]
                    )

        self.assertEqual(exit_code, 0)
        output = "".join(call.args[0] for call in stdout_write.call_args_list)
        self.assertIn("manual_input:", output)
        self.assertIn("imported_submissions: 1", output)
        self.assertIn("candidate_posts: 1", output)
        self.assertIn("saved_comments: 1", output)
        self.assertIn("comment_enrichment_json:", output)

    def test_run_cli_supports_manual_input_without_reddit_env(self) -> None:
        class FakeLMStudioClient:
            def __init__(self, config):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "manual.json"
            _write_json_file(
                input_path,
                {
                    "posts": [
                        {
                            "id": "a",
                            "title": "Manual follow-up is painful",
                            "subreddit": "Entrepreneur",
                            "url": "https://reddit.com/a",
                            "score": 30,
                            "num_comments": 12,
                            "selftext": "Everything still lives in spreadsheets.",
                            "comments": [{"id": "a1", "body": "Exact same issue", "score": 6, "depth": 0}],
                        },
                        {
                            "id": "b",
                            "title": "Lead tracking keeps breaking follow-up",
                            "subreddit": "Entrepreneur",
                            "url": "https://reddit.com/b",
                            "score": 28,
                            "num_comments": 10,
                            "selftext": "Still doing reminders by hand.",
                            "comments": [{"id": "b1", "body": "We do this too", "score": 5, "depth": 0}],
                        },
                        {
                            "id": "c",
                            "title": "CRM setup is too heavy so we stay manual",
                            "subreddit": "Entrepreneur",
                            "url": "https://reddit.com/c",
                            "score": 24,
                            "num_comments": 8,
                            "selftext": "Spreadsheet workflow never ends.",
                            "comments": [{"id": "c1", "body": "Still manual here too", "score": 5, "depth": 0}],
                        },
                        {
                            "id": "d",
                            "title": "Manual reminders kill our process",
                            "subreddit": "Entrepreneur",
                            "url": "https://reddit.com/d",
                            "score": 23,
                            "num_comments": 7,
                            "selftext": "Follow-up depends on copy and paste.",
                            "comments": [{"id": "d1", "body": "Same pain", "score": 4, "depth": 0}],
                        },
                        {
                            "id": "e",
                            "title": "Spreadsheet CRM is still our follow-up system",
                            "subreddit": "Entrepreneur",
                            "url": "https://reddit.com/e",
                            "score": 22,
                            "num_comments": 7,
                            "selftext": "Every handoff is manual.",
                            "comments": [{"id": "e1", "body": "I hate this workflow", "score": 4, "depth": 0}],
                        },
                    ]
                },
            )
            run_dir = Path(tmpdir) / "run-manual"

            fake_ranking_result = AsyncMock()
            fake_ranking_result.run_dir = run_dir
            fake_ranking_result.selected_count = 5
            fake_ranking_result.candidate_count = 5
            fake_ranking_result.screened_candidate_count = 5
            fake_ranking_result.rejected_candidate_count = 0
            fake_ranking_result.rejection_counts = {}

            fake_cluster_result = AsyncMock()
            fake_cluster_result.run_dir = run_dir
            fake_cluster_result.cluster_count = 1
            fake_cluster_result.strongest_cluster_id = "cluster-1"
            fake_cluster_result.strongest_post_ids = ["a", "b", "c", "d", "e"]
            fake_cluster_result.strongest_cluster_complaint_signal_post_count = 5
            fake_cluster_result.strongest_cluster_screened_post_count = 5
            fake_cluster_result.evidence_validation_passed = True
            fake_cluster_result.evidence_failure_reason = None

            fake_summary_artifact = AsyncMock()
            fake_summary_artifact.run_dir = str(run_dir)
            fake_summary_artifact.candidate_count = 5
            fake_summary_artifact.comment_count = 5
            fake_summary_artifact.selected_comment_count = 5
            fake_summary_artifact.max_posts_used = 5

            fake_memo_artifact = AsyncMock()
            fake_memo_artifact.run_dir = str(run_dir)
            fake_memo_artifact.provider = "lmstudio"
            fake_memo_artifact.model = "openai/gpt-oss-20b"
            fake_memo_artifact.strongest_cluster_id = "cluster-1"
            fake_memo_artifact.strongest_cluster_size = 5
            fake_memo_artifact.included_post_count = 5

            def fake_rank(*args, **kwargs):
                _write_json_file(
                    run_dir / "candidate_screening.json",
                    {"candidate_count": 5, "kept_count": 5, "rejected_count": 0},
                )
                _write_json_file(run_dir / "post_ranking.json", {"candidate_count": 5})
                _write_json_file(run_dir / "selected_posts.json", [])
                return fake_ranking_result

            def fake_cluster(*args, **kwargs):
                _write_json_file(
                    run_dir / "theme_summary.json",
                    {
                        "run_dir": str(run_dir),
                        "generated_at": "2026-04-02T00:00:00Z",
                        "source_post_count": 5,
                        "cluster_count": 1,
                        "strongest_cluster_id": "cluster-1",
                        "strongest_post_ids": ["a", "b", "c", "d", "e"],
                        "clusters": [],
                    },
                )
                _write_json_file(
                    run_dir / "cluster_evidence_validation.json",
                    {
                        "strongest_cluster_id": "cluster-1",
                        "strongest_cluster_post_count": 5,
                        "screening_available": True,
                        "screened_cluster_post_count": 5,
                        "complaint_signal_post_count": 5,
                        "min_cluster_complaint_posts": 2,
                        "passes": True,
                        "failure_reason": None,
                    },
                )
                return fake_cluster_result

            async def fake_summarize(*args, **kwargs):
                _write_json_file(run_dir / "comment_selection.json", {"selected_comment_count": 5})
                _write_json_file(run_dir / "evidence_summary.json", {"summary_text": "summary"})
                return fake_summary_artifact

            async def fake_memo(*args, **kwargs):
                _write_json_file(run_dir / "final_memo.json", {"memo_text": "memo"})
                (run_dir / "final_memo.md").write_text("# Final Memo\n", encoding="utf-8")
                return fake_memo_artifact

            with patch.dict(
                "os.environ",
                {
                    "LLM_PROVIDER": "lmstudio",
                    "LLM_BASE_URL": "http://127.0.0.1:1234/v1",
                    "LLM_MODEL": "openai/gpt-oss-20b",
                },
                clear=True,
            ):
                with patch(
                    "reddit_pain_agent.main.load_runtime_config",
                    side_effect=AssertionError("manual runs should not load Reddit config"),
                ), patch(
                    "reddit_pain_agent.main.enrich_run_with_comments",
                    AsyncMock(side_effect=AssertionError("manual runs should not fetch Reddit comments")),
                ) as comments_mock, patch(
                    "reddit_pain_agent.main.rank_run_candidates",
                    side_effect=fake_rank,
                ), patch(
                    "reddit_pain_agent.main.cluster_run_posts",
                    side_effect=fake_cluster,
                ), patch(
                    "reddit_pain_agent.main.summarize_candidate_posts",
                    AsyncMock(side_effect=fake_summarize),
                ), patch(
                    "reddit_pain_agent.main.write_final_memo",
                    AsyncMock(side_effect=fake_memo),
                ), patch(
                    "reddit_pain_agent.main.LMStudioClient",
                    FakeLMStudioClient,
                ), patch("sys.stdout.write") as stdout_write:
                    exit_code = main(
                        [
                            "run",
                            "--manual-input",
                            str(input_path),
                            "--output-dir",
                            str(run_dir),
                            "--subreddit",
                            "Entrepreneur",
                            "--query",
                            "manual follow-up pain",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertEqual(comments_mock.await_count, 0)
            report_payload = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report_payload["status"], "completed")
            self.assertEqual(report_payload["stage_reports"][0]["details"]["search_mode"], "manual")
            self.assertEqual(report_payload["stage_reports"][1]["details"]["source"], "manual_import")
            output = "".join(call.args[0] for call in stdout_write.call_args_list)
            self.assertIn("status: completed", output)
            self.assertIn("requests: 0", output)
            self.assertIn("saved_comments: 5", output)
            self.assertIn("run_report_json:", output)

    def test_run_cli_orchestrates_pipeline_and_prints_summary(self) -> None:
        class FakeLMStudioClient:
            def __init__(self, config):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-1"
            fake_search_result = AsyncMock()
            fake_search_result.run_slug = "search-20260402-120000-entrepreneur-manual-follow-up"
            fake_search_result.run_dir = run_dir
            fake_search_result.request_count = 4
            fake_search_result.candidate_count = 12
            fake_search_result.query_variant_count = 2
            fake_search_result.search_spec_count = 2
            fake_search_result.sort_count = 1
            fake_search_result.time_filter_count = 1
            fake_search_result.pages_per_query = 2
            fake_search_result.filtered_counts = {}

            fake_comments_result = AsyncMock()
            fake_comments_result.run_dir = run_dir
            fake_comments_result.comment_count = 27
            fake_comments_result.requested_submission_count = 5
            fake_comments_result.fetched_submission_count = 5
            fake_comments_result.morechildren_request_count = 1

            fake_ranking_result = AsyncMock()
            fake_ranking_result.run_dir = run_dir
            fake_ranking_result.selected_count = 8
            fake_ranking_result.candidate_count = 12
            fake_ranking_result.screened_candidate_count = 8
            fake_ranking_result.rejected_candidate_count = 4
            fake_ranking_result.rejection_counts = {"low_non_trivial_comments": 4}

            fake_cluster_result = AsyncMock()
            fake_cluster_result.run_dir = run_dir
            fake_cluster_result.cluster_count = 3
            fake_cluster_result.strongest_cluster_id = "cluster-1"
            fake_cluster_result.strongest_post_ids = ["a", "b", "c", "d", "e"]
            fake_cluster_result.strongest_cluster_complaint_signal_post_count = 5
            fake_cluster_result.strongest_cluster_screened_post_count = 5
            fake_cluster_result.evidence_validation_passed = True
            fake_cluster_result.evidence_failure_reason = None

            fake_summary_artifact = AsyncMock()
            fake_summary_artifact.run_dir = str(run_dir)
            fake_summary_artifact.candidate_count = 5
            fake_summary_artifact.comment_count = 9
            fake_summary_artifact.selected_comment_count = 4
            fake_summary_artifact.max_posts_used = 5

            fake_memo_artifact = AsyncMock()
            fake_memo_artifact.run_dir = str(run_dir)
            fake_memo_artifact.provider = "lmstudio"
            fake_memo_artifact.model = "openai/gpt-oss-20b"
            fake_memo_artifact.strongest_cluster_id = "cluster-1"
            fake_memo_artifact.strongest_cluster_size = 5
            fake_memo_artifact.included_post_count = 5

            async def fake_search(*args, **kwargs):
                _write_json_file(run_dir / "candidate_posts.json", [])
                return fake_search_result

            async def fake_comments(*args, **kwargs):
                _write_json_file(run_dir / "comment_enrichment.json", {"comment_count": 27})
                _write_json_file(run_dir / "comments" / "a.json", {"submission_id": "a", "comments": []})
                return fake_comments_result

            def fake_rank(*args, **kwargs):
                _write_json_file(
                    run_dir / "candidate_screening.json",
                    {"candidate_count": 12, "kept_count": 8, "rejected_count": 4},
                )
                _write_json_file(run_dir / "post_ranking.json", {"candidate_count": 12})
                _write_json_file(run_dir / "selected_posts.json", [])
                return fake_ranking_result

            def fake_cluster(*args, **kwargs):
                _write_json_file(
                    run_dir / "theme_summary.json",
                    {
                        "run_dir": str(run_dir),
                        "generated_at": "2026-04-02T00:00:00Z",
                        "source_post_count": 8,
                        "cluster_count": 3,
                        "strongest_cluster_id": "cluster-1",
                        "strongest_post_ids": ["a", "b", "c", "d", "e"],
                        "clusters": [],
                    },
                )
                _write_json_file(
                    run_dir / "cluster_evidence_validation.json",
                    {
                        "strongest_cluster_id": "cluster-1",
                        "strongest_cluster_post_count": 5,
                        "screening_available": True,
                        "screened_cluster_post_count": 5,
                        "complaint_signal_post_count": 5,
                        "min_cluster_complaint_posts": 2,
                        "passes": True,
                        "failure_reason": None,
                    },
                )
                return fake_cluster_result

            async def fake_summarize(*args, **kwargs):
                _write_json_file(run_dir / "comment_selection.json", {"selected_comment_count": 4})
                _write_json_file(run_dir / "evidence_summary.json", {"summary_text": "summary"})
                return fake_summary_artifact

            async def fake_memo(*args, **kwargs):
                _write_json_file(run_dir / "final_memo.json", {"memo_text": "memo"})
                (run_dir / "final_memo.md").write_text("# Final Memo\n", encoding="utf-8")
                return fake_memo_artifact

            with patch.dict(
                "os.environ",
                {
                    "REDDIT_CLIENT_ID": "client-id",
                    "REDDIT_USER_AGENT": "script:test:v1 (by /u/example)",
                    "REDDIT_OUTPUT_ROOT": tmpdir,
                    "LLM_PROVIDER": "lmstudio",
                    "LLM_BASE_URL": "http://127.0.0.1:1234/v1",
                    "LLM_MODEL": "openai/gpt-oss-20b",
                },
                clear=True,
            ):
                with patch(
                    "reddit_pain_agent.main.run_search_command",
                    AsyncMock(side_effect=fake_search),
                ), patch(
                    "reddit_pain_agent.main.enrich_run_with_comments",
                    AsyncMock(side_effect=fake_comments),
                ), patch(
                    "reddit_pain_agent.main.rank_run_candidates",
                    side_effect=fake_rank,
                ), patch(
                    "reddit_pain_agent.main.cluster_run_posts",
                    side_effect=fake_cluster,
                ), patch(
                    "reddit_pain_agent.main.summarize_candidate_posts",
                    AsyncMock(side_effect=fake_summarize),
                ) as summarize_mock, patch(
                    "reddit_pain_agent.main.write_final_memo",
                    AsyncMock(side_effect=fake_memo),
                ) as memo_mock, patch(
                    "reddit_pain_agent.main.LMStudioClient",
                    FakeLMStudioClient,
                ), patch("sys.stdout.write") as stdout_write:
                    exit_code = main(
                        [
                            "run",
                            "--subreddit",
                            "Entrepreneur",
                            "--query",
                            "manual follow-up pain",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertEqual(summarize_mock.await_count, 1)
            self.assertEqual(memo_mock.await_count, 1)
            report_payload = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report_payload["status"], "completed")
            self.assertEqual(report_payload["provider"], "lmstudio")
            self.assertEqual(report_payload["model"], "openai/gpt-oss-20b")
            self.assertEqual([item["stage"] for item in report_payload["stage_reports"]], ["search", "comments", "rank", "cluster", "summarize", "memo"])
            self.assertIn("candidate_posts.json", report_payload["stage_reports"][0]["artifact_fingerprints"])
            output = "".join(call.args[0] for call in stdout_write.call_args_list)
            self.assertIn("status: completed", output)
            self.assertIn("provider: lmstudio", output)
            self.assertIn("candidate_posts: 12", output)
            self.assertIn("rank_survivors: 8", output)
            self.assertIn("rank_filtered: low_non_trivial_comments=4", output)
            self.assertIn("strongest_cluster_posts: 5", output)
            self.assertIn("cluster_complaint_posts: 5", output)
            self.assertIn("final_memo_markdown:", output)
            self.assertIn("run_report_json:", output)

    def test_run_cli_stops_cleanly_when_cluster_is_too_weak(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-1"
            fake_search_result = AsyncMock()
            fake_search_result.run_slug = "search-20260402-120000-entrepreneur-manual-follow-up"
            fake_search_result.run_dir = run_dir
            fake_search_result.request_count = 4
            fake_search_result.candidate_count = 12
            fake_search_result.query_variant_count = 2
            fake_search_result.search_spec_count = 2
            fake_search_result.sort_count = 1
            fake_search_result.time_filter_count = 1
            fake_search_result.pages_per_query = 2
            fake_search_result.filtered_counts = {}

            fake_comments_result = AsyncMock()
            fake_comments_result.run_dir = run_dir
            fake_comments_result.comment_count = 11
            fake_comments_result.requested_submission_count = 5
            fake_comments_result.fetched_submission_count = 5
            fake_comments_result.morechildren_request_count = 0

            fake_ranking_result = AsyncMock()
            fake_ranking_result.run_dir = run_dir
            fake_ranking_result.selected_count = 6
            fake_ranking_result.candidate_count = 12
            fake_ranking_result.screened_candidate_count = 6
            fake_ranking_result.rejected_candidate_count = 6
            fake_ranking_result.rejection_counts = {"low_non_trivial_comments": 6}

            fake_cluster_result = AsyncMock()
            fake_cluster_result.run_dir = run_dir
            fake_cluster_result.cluster_count = 3
            fake_cluster_result.strongest_cluster_id = "cluster-2"
            fake_cluster_result.strongest_post_ids = ["a", "b", "c"]
            fake_cluster_result.strongest_cluster_complaint_signal_post_count = 3
            fake_cluster_result.strongest_cluster_screened_post_count = 3
            fake_cluster_result.evidence_validation_passed = True
            fake_cluster_result.evidence_failure_reason = None

            async def fake_search(*args, **kwargs):
                _write_json_file(run_dir / "candidate_posts.json", [])
                return fake_search_result

            async def fake_comments(*args, **kwargs):
                _write_json_file(run_dir / "comment_enrichment.json", {"comment_count": 11})
                _write_json_file(run_dir / "comments" / "a.json", {"submission_id": "a", "comments": []})
                return fake_comments_result

            def fake_rank(*args, **kwargs):
                _write_json_file(
                    run_dir / "candidate_screening.json",
                    {"candidate_count": 12, "kept_count": 6, "rejected_count": 6},
                )
                _write_json_file(run_dir / "post_ranking.json", {"candidate_count": 12})
                _write_json_file(run_dir / "selected_posts.json", [])
                return fake_ranking_result

            def fake_cluster(*args, **kwargs):
                _write_json_file(
                    run_dir / "theme_summary.json",
                    {
                        "run_dir": str(run_dir),
                        "generated_at": "2026-04-02T00:00:00Z",
                        "source_post_count": 6,
                        "cluster_count": 3,
                        "strongest_cluster_id": "cluster-2",
                        "strongest_post_ids": ["a", "b", "c"],
                        "clusters": [],
                    },
                )
                _write_json_file(
                    run_dir / "cluster_evidence_validation.json",
                    {
                        "strongest_cluster_id": "cluster-2",
                        "strongest_cluster_post_count": 3,
                        "screening_available": True,
                        "screened_cluster_post_count": 3,
                        "complaint_signal_post_count": 3,
                        "min_cluster_complaint_posts": 2,
                        "passes": True,
                        "failure_reason": None,
                    },
                )
                return fake_cluster_result

            with patch.dict(
                "os.environ",
                {
                    "REDDIT_CLIENT_ID": "client-id",
                    "REDDIT_USER_AGENT": "script:test:v1 (by /u/example)",
                    "REDDIT_OUTPUT_ROOT": tmpdir,
                },
                clear=True,
            ):
                with patch(
                    "reddit_pain_agent.main.run_search_command",
                    AsyncMock(side_effect=fake_search),
                ), patch(
                    "reddit_pain_agent.main.enrich_run_with_comments",
                    AsyncMock(side_effect=fake_comments),
                ), patch(
                    "reddit_pain_agent.main.rank_run_candidates",
                    side_effect=fake_rank,
                ), patch(
                    "reddit_pain_agent.main.cluster_run_posts",
                    side_effect=fake_cluster,
                ), patch(
                    "reddit_pain_agent.main.summarize_candidate_posts",
                    AsyncMock(),
                ) as summarize_mock, patch(
                    "reddit_pain_agent.main.write_final_memo",
                    AsyncMock(),
                ) as memo_mock, patch("sys.stdout.write") as stdout_write:
                    exit_code = main(
                        [
                            "run",
                            "--subreddit",
                            "Entrepreneur",
                            "--query",
                            "manual follow-up pain",
                            "--min-cluster-posts",
                            "5",
                        ]
                    )

            self.assertEqual(exit_code, 2)
            self.assertEqual(summarize_mock.await_count, 0)
            self.assertEqual(memo_mock.await_count, 0)
            report_payload = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report_payload["status"], "stopped")
            self.assertEqual(report_payload["stop_reason"], "strongest_cluster_too_weak")
            self.assertEqual(report_payload["stage_reports"][-1]["stage"], "memo")
            self.assertEqual(report_payload["stage_reports"][-1]["status"], "stopped")
            output = "".join(call.args[0] for call in stdout_write.call_args_list)
            self.assertIn("status: stopped", output)
            self.assertIn("stop_reason: strongest_cluster_too_weak", output)
            self.assertIn("rank_filtered: low_non_trivial_comments=6", output)
            self.assertIn("required_cluster_posts: 5", output)
            self.assertIn("run_report_json:", output)

    def test_run_cli_stops_when_cluster_evidence_is_too_weak(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-1"
            fake_search_result = AsyncMock()
            fake_search_result.run_slug = "search-20260402-120000-entrepreneur-manual-follow-up"
            fake_search_result.run_dir = run_dir
            fake_search_result.request_count = 4
            fake_search_result.candidate_count = 12
            fake_search_result.query_variant_count = 2
            fake_search_result.search_spec_count = 2
            fake_search_result.sort_count = 1
            fake_search_result.time_filter_count = 1
            fake_search_result.pages_per_query = 2
            fake_search_result.filtered_counts = {}

            fake_comments_result = AsyncMock()
            fake_comments_result.run_dir = run_dir
            fake_comments_result.comment_count = 21
            fake_comments_result.requested_submission_count = 5
            fake_comments_result.fetched_submission_count = 5
            fake_comments_result.morechildren_request_count = 1

            fake_ranking_result = AsyncMock()
            fake_ranking_result.run_dir = run_dir
            fake_ranking_result.selected_count = 8
            fake_ranking_result.candidate_count = 12
            fake_ranking_result.screened_candidate_count = 8
            fake_ranking_result.rejected_candidate_count = 4
            fake_ranking_result.rejection_counts = {"low_non_trivial_comments": 4}

            fake_cluster_result = AsyncMock()
            fake_cluster_result.run_dir = run_dir
            fake_cluster_result.cluster_count = 2
            fake_cluster_result.strongest_cluster_id = "cluster-1"
            fake_cluster_result.strongest_post_ids = ["a", "b", "c", "d", "e"]
            fake_cluster_result.strongest_cluster_complaint_signal_post_count = 1
            fake_cluster_result.strongest_cluster_screened_post_count = 5
            fake_cluster_result.evidence_validation_passed = False
            fake_cluster_result.evidence_failure_reason = "insufficient_cluster_complaint_signal_posts"

            async def fake_search(*args, **kwargs):
                _write_json_file(run_dir / "candidate_posts.json", [])
                return fake_search_result

            async def fake_comments(*args, **kwargs):
                _write_json_file(run_dir / "comment_enrichment.json", {"comment_count": 21})
                _write_json_file(run_dir / "comments" / "a.json", {"submission_id": "a", "comments": []})
                return fake_comments_result

            def fake_rank(*args, **kwargs):
                _write_json_file(
                    run_dir / "candidate_screening.json",
                    {"candidate_count": 12, "kept_count": 8, "rejected_count": 4},
                )
                _write_json_file(run_dir / "post_ranking.json", {"candidate_count": 12})
                _write_json_file(run_dir / "selected_posts.json", [])
                return fake_ranking_result

            def fake_cluster(*args, **kwargs):
                _write_json_file(
                    run_dir / "theme_summary.json",
                    {
                        "run_dir": str(run_dir),
                        "generated_at": "2026-04-02T00:00:00Z",
                        "source_post_count": 8,
                        "cluster_count": 2,
                        "strongest_cluster_id": "cluster-1",
                        "strongest_post_ids": ["a", "b", "c", "d", "e"],
                        "clusters": [],
                    },
                )
                _write_json_file(
                    run_dir / "cluster_evidence_validation.json",
                    {
                        "strongest_cluster_id": "cluster-1",
                        "strongest_cluster_post_count": 5,
                        "screening_available": True,
                        "screened_cluster_post_count": 5,
                        "complaint_signal_post_count": 1,
                        "min_cluster_complaint_posts": 2,
                        "passes": False,
                        "failure_reason": "insufficient_cluster_complaint_signal_posts",
                    },
                )
                return fake_cluster_result

            with patch.dict(
                "os.environ",
                {
                    "REDDIT_CLIENT_ID": "client-id",
                    "REDDIT_USER_AGENT": "script:test:v1 (by /u/example)",
                    "REDDIT_OUTPUT_ROOT": tmpdir,
                },
                clear=True,
            ):
                with patch(
                    "reddit_pain_agent.main.run_search_command",
                    AsyncMock(side_effect=fake_search),
                ), patch(
                    "reddit_pain_agent.main.enrich_run_with_comments",
                    AsyncMock(side_effect=fake_comments),
                ), patch(
                    "reddit_pain_agent.main.rank_run_candidates",
                    side_effect=fake_rank,
                ), patch(
                    "reddit_pain_agent.main.cluster_run_posts",
                    side_effect=fake_cluster,
                ), patch(
                    "reddit_pain_agent.main.summarize_candidate_posts",
                    AsyncMock(),
                ) as summarize_mock, patch(
                    "reddit_pain_agent.main.write_final_memo",
                    AsyncMock(),
                ) as memo_mock, patch("sys.stdout.write") as stdout_write:
                    exit_code = main(
                        [
                            "run",
                            "--subreddit",
                            "Entrepreneur",
                            "--query",
                            "manual follow-up pain",
                        ]
                    )

            self.assertEqual(exit_code, 2)
            self.assertEqual(summarize_mock.await_count, 0)
            self.assertEqual(memo_mock.await_count, 0)
            report_payload = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report_payload["status"], "stopped")
            self.assertEqual(report_payload["stop_reason"], "strongest_cluster_evidence_too_weak")
            output = "".join(call.args[0] for call in stdout_write.call_args_list)
            self.assertIn("status: stopped", output)
            self.assertIn("stop_reason: strongest_cluster_evidence_too_weak", output)
            self.assertIn("cluster_complaint_posts: 1", output)
            self.assertIn("required_cluster_complaint_posts: 2", output)
            self.assertIn(
                "evidence_failure_reason: insufficient_cluster_complaint_signal_posts",
                output,
            )

    def test_run_cli_resume_skips_completed_stages(self) -> None:
        class FakeLMStudioClient:
            def __init__(self, config):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-1"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json_file(run_dir / "candidate_posts.json", [])
            _write_json_file(run_dir / "comment_enrichment.json", {})
            _write_json_file(run_dir / "comments" / "abc123.json", {"submission_id": "abc123", "comments": []})
            _write_json_file(run_dir / "candidate_screening.json", {})
            _write_json_file(run_dir / "post_ranking.json", {})
            _write_json_file(run_dir / "selected_posts.json", {})
            _write_json_file(run_dir / "theme_summary.json", {})
            _write_json_file(run_dir / "cluster_evidence_validation.json", {})
            _write_json_file(run_dir / "cluster_evidence_validation.json", {})
            search_fingerprints = _fingerprints_for_paths(run_dir, run_dir / "candidate_posts.json")
            comments_fingerprints = _fingerprints_for_paths(
                run_dir,
                run_dir / "comment_enrichment.json",
                run_dir / "comments" / "abc123.json",
            )
            rank_fingerprints = _fingerprints_for_paths(
                run_dir,
                run_dir / "candidate_screening.json",
                run_dir / "post_ranking.json",
                run_dir / "selected_posts.json",
            )
            cluster_fingerprints = _fingerprints_for_paths(
                run_dir,
                run_dir / "theme_summary.json",
                run_dir / "cluster_evidence_validation.json",
            )
            (run_dir / "run_report.json").write_text(
                json.dumps(
                    {
                        "run_slug": "search-20260402-120000-entrepreneur-manual-follow-up",
                        "run_dir": str(run_dir),
                        "status": "stopped",
                        "started_at": "2026-04-02T00:00:00Z",
                        "completed_at": "2026-04-02T00:05:00Z",
                        "subreddits": ["Entrepreneur"],
                        "queries": ["manual follow-up pain"],
                        "sort": "relevance",
                        "time_filter": "all",
                        "limit": 25,
                        "provider": None,
                        "model": None,
                        "stop_reason": "strongest_cluster_too_weak",
                        "error": None,
                        "stage_reports": [
                            {
                                "stage": "search",
                                "status": "completed",
                                "duration_ms": 10.0,
                                "details": {
                                    "params": {
                                        "subreddits": ["Entrepreneur"],
                                        "queries": ["manual follow-up pain"],
                                        "sort": "relevance",
                                        "time_filter": "all",
                                        "additional_sorts": [],
                                        "additional_time_filters": [],
                                        "limit": 25,
                                        "pages_per_query": 2,
                                        "expand_queries": True,
                                        "min_score": 0,
                                        "min_comments": 0,
                                        "filter_nsfw": False,
                                        "allowed_subreddits": [],
                                        "denied_subreddits": [],
                                    },
                                    "request_count": 4,
                                    "candidate_count": 12,
                                    "query_variant_count": 2,
                                    "search_spec_count": 2,
                                    "sort_count": 1,
                                    "time_filter_count": 1,
                                    "pages_per_query": 2,
                                    "filtered_counts": {},
                                    "run_dir": str(run_dir),
                                },
                                "artifact_fingerprints": search_fingerprints,
                            },
                            {
                                "stage": "comments",
                                "status": "completed",
                                "duration_ms": 8.0,
                                "details": {
                                    "params": {
                                        "max_posts": 5,
                                        "comment_limit": 20,
                                        "comment_depth": 3,
                                        "comment_sort": "best",
                                        "max_morechildren_requests": 3,
                                        "morechildren_batch_size": 20,
                                    },
                                    "requested_submission_count": 5,
                                    "fetched_submission_count": 5,
                                    "comment_count": 14,
                                    "morechildren_request_count": 1,
                                },
                                "artifact_fingerprints": comments_fingerprints,
                            },
                            {
                                "stage": "rank",
                                "status": "completed",
                                "duration_ms": 4.0,
                                "details": {
                                    "params": {
                                        "max_selected_posts": 10,
                                        "min_non_trivial_comments": 0,
                                        "min_complaint_signal_comments": 0,
                                    },
                                    "candidate_count": 12,
                                    "screened_candidate_count": 12,
                                    "rejected_candidate_count": 0,
                                    "selected_count": 8,
                                    "rejection_counts": {},
                                },
                                "artifact_fingerprints": rank_fingerprints,
                            },
                            {
                                "stage": "cluster",
                                "status": "completed",
                                "duration_ms": 3.0,
                                "details": {
                                    "params": {
                                        "similarity_threshold": 0.22,
                                        "min_shared_terms": 2,
                                        "min_cluster_complaint_posts": 2,
                                    },
                                    "cluster_count": 3,
                                    "strongest_cluster_id": "cluster-2",
                                    "strongest_cluster_post_count": 5,
                                    "strongest_cluster_screened_post_count": 5,
                                    "strongest_cluster_complaint_signal_post_count": 5,
                                    "evidence_validation_passed": True,
                                    "evidence_failure_reason": None,
                                },
                                "artifact_fingerprints": cluster_fingerprints,
                            },
                            {
                                "stage": "summarize",
                                "status": "skipped",
                                "details": {
                                    "params": {"model": None, "max_posts": 10},
                                    "reason": "strongest_cluster_too_weak",
                                },
                            },
                            {
                                "stage": "memo",
                                "status": "stopped",
                                "details": {
                                    "params": {"model": None, "min_cluster_posts": 5, "max_posts": 8},
                                    "reason": "strongest_cluster_too_weak",
                                },
                            },
                        ],
                        "output_paths": {},
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            fake_summary_artifact = AsyncMock()
            fake_summary_artifact.candidate_count = 5
            fake_summary_artifact.comment_count = 9
            fake_summary_artifact.selected_comment_count = 4
            fake_summary_artifact.max_posts_used = 5

            fake_memo_artifact = AsyncMock()
            fake_memo_artifact.provider = "lmstudio"
            fake_memo_artifact.model = "openai/gpt-oss-20b"
            fake_memo_artifact.strongest_cluster_id = "cluster-2"
            fake_memo_artifact.strongest_cluster_size = 5
            fake_memo_artifact.included_post_count = 5

            async def fake_summarize(*args, **kwargs):
                _write_json_file(run_dir / "comment_selection.json", {"selected_comment_count": 4})
                _write_json_file(run_dir / "evidence_summary.json", {"summary_text": "summary"})
                return fake_summary_artifact

            async def fake_memo(*args, **kwargs):
                _write_json_file(run_dir / "final_memo.json", {"memo_text": "memo"})
                (run_dir / "final_memo.md").write_text("# Final Memo\n", encoding="utf-8")
                return fake_memo_artifact

            with patch.dict(
                "os.environ",
                {
                    "REDDIT_CLIENT_ID": "client-id",
                    "REDDIT_USER_AGENT": "script:test:v1 (by /u/example)",
                    "REDDIT_OUTPUT_ROOT": tmpdir,
                    "LLM_PROVIDER": "lmstudio",
                    "LLM_BASE_URL": "http://127.0.0.1:1234/v1",
                    "LLM_MODEL": "openai/gpt-oss-20b",
                },
                clear=True,
            ):
                with patch(
                    "reddit_pain_agent.main.run_search_command",
                    AsyncMock(),
                ) as search_mock, patch(
                    "reddit_pain_agent.main.enrich_run_with_comments",
                    AsyncMock(),
                ) as comments_mock, patch(
                    "reddit_pain_agent.main.rank_run_candidates",
                ) as rank_mock, patch(
                    "reddit_pain_agent.main.cluster_run_posts",
                ) as cluster_mock, patch(
                    "reddit_pain_agent.main.summarize_candidate_posts",
                    AsyncMock(side_effect=fake_summarize),
                ) as summarize_mock, patch(
                    "reddit_pain_agent.main.write_final_memo",
                    AsyncMock(side_effect=fake_memo),
                ) as memo_mock, patch(
                    "reddit_pain_agent.main.LMStudioClient",
                    FakeLMStudioClient,
                ), patch("sys.stdout.write") as stdout_write:
                    exit_code = main(
                        [
                            "run",
                            "--resume",
                            "--output-dir",
                            str(run_dir),
                            "--subreddit",
                            "Entrepreneur",
                            "--query",
                            "manual follow-up pain",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertEqual(search_mock.await_count, 0)
            self.assertEqual(comments_mock.await_count, 0)
            self.assertEqual(rank_mock.call_count, 0)
            self.assertEqual(cluster_mock.call_count, 0)
            self.assertEqual(summarize_mock.await_count, 1)
            self.assertEqual(memo_mock.await_count, 1)
            report_payload = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report_payload["status"], "completed")
            self.assertEqual(report_payload["stage_reports"][-2]["stage"], "summarize")
            self.assertEqual(report_payload["stage_reports"][-1]["stage"], "memo")
            output = "".join(call.args[0] for call in stdout_write.call_args_list)
            self.assertIn("resume_from_stage: summarize", output)
            self.assertIn("status: completed", output)

    def test_run_cli_resume_reruns_cluster_when_cluster_params_change(self) -> None:
        class FakeLMStudioClient:
            def __init__(self, config):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-1"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json_file(run_dir / "candidate_posts.json", [])
            _write_json_file(run_dir / "comment_enrichment.json", {})
            _write_json_file(run_dir / "comments" / "abc123.json", {"submission_id": "abc123", "comments": []})
            _write_json_file(run_dir / "candidate_screening.json", {})
            _write_json_file(run_dir / "post_ranking.json", {})
            _write_json_file(run_dir / "selected_posts.json", {})
            _write_json_file(run_dir / "theme_summary.json", {})
            _write_json_file(run_dir / "cluster_evidence_validation.json", {"passes": True})
            search_fingerprints = _fingerprints_for_paths(run_dir, run_dir / "candidate_posts.json")
            comments_fingerprints = _fingerprints_for_paths(
                run_dir,
                run_dir / "comment_enrichment.json",
                run_dir / "comments" / "abc123.json",
            )
            rank_fingerprints = _fingerprints_for_paths(
                run_dir,
                run_dir / "candidate_screening.json",
                run_dir / "post_ranking.json",
                run_dir / "selected_posts.json",
            )
            cluster_fingerprints = _fingerprints_for_paths(
                run_dir,
                run_dir / "theme_summary.json",
                run_dir / "cluster_evidence_validation.json",
            )
            (run_dir / "run_report.json").write_text(
                json.dumps(
                    {
                        "run_slug": "search-20260402-120000-entrepreneur-manual-follow-up",
                        "run_dir": str(run_dir),
                        "status": "failed",
                        "started_at": "2026-04-02T00:00:00Z",
                        "completed_at": "2026-04-02T00:05:00Z",
                        "subreddits": ["Entrepreneur"],
                        "queries": ["manual follow-up pain"],
                        "sort": "relevance",
                        "time_filter": "all",
                        "limit": 25,
                        "provider": None,
                        "model": None,
                        "stop_reason": None,
                        "error": "cluster drift",
                        "stage_reports": [
                            {
                                "stage": "search",
                                "status": "completed",
                                "details": {
                                    "params": {
                                        "subreddits": ["Entrepreneur"],
                                        "queries": ["manual follow-up pain"],
                                        "sort": "relevance",
                                        "time_filter": "all",
                                        "additional_sorts": [],
                                        "additional_time_filters": [],
                                        "limit": 25,
                                        "pages_per_query": 2,
                                        "expand_queries": True,
                                        "min_score": 0,
                                        "min_comments": 0,
                                        "filter_nsfw": False,
                                        "allowed_subreddits": [],
                                        "denied_subreddits": [],
                                    },
                                    "request_count": 4,
                                    "candidate_count": 12,
                                    "query_variant_count": 2,
                                    "search_spec_count": 2,
                                    "sort_count": 1,
                                    "time_filter_count": 1,
                                    "pages_per_query": 2,
                                    "filtered_counts": {},
                                    "run_dir": str(run_dir),
                                },
                                "artifact_fingerprints": search_fingerprints,
                            },
                            {
                                "stage": "comments",
                                "status": "completed",
                                "details": {
                                    "params": {
                                        "max_posts": 5,
                                        "comment_limit": 20,
                                        "comment_depth": 3,
                                        "comment_sort": "best",
                                        "max_morechildren_requests": 3,
                                        "morechildren_batch_size": 20,
                                    },
                                    "requested_submission_count": 5,
                                    "fetched_submission_count": 5,
                                    "comment_count": 14,
                                    "morechildren_request_count": 1,
                                },
                                "artifact_fingerprints": comments_fingerprints,
                            },
                            {
                                "stage": "rank",
                                "status": "completed",
                                "details": {
                                    "params": {
                                        "max_selected_posts": 10,
                                        "min_non_trivial_comments": 0,
                                        "min_complaint_signal_comments": 0,
                                    },
                                    "candidate_count": 12,
                                    "screened_candidate_count": 12,
                                    "rejected_candidate_count": 0,
                                    "selected_count": 8,
                                    "rejection_counts": {},
                                },
                                "artifact_fingerprints": rank_fingerprints,
                            },
                            {
                                "stage": "cluster",
                                "status": "completed",
                                "details": {
                                    "params": {
                                        "similarity_threshold": 0.22,
                                        "min_shared_terms": 2,
                                        "min_cluster_complaint_posts": 2,
                                    },
                                    "cluster_count": 3,
                                    "strongest_cluster_id": "cluster-2",
                                    "strongest_cluster_post_count": 5,
                                    "strongest_cluster_screened_post_count": 5,
                                    "strongest_cluster_complaint_signal_post_count": 5,
                                    "evidence_validation_passed": True,
                                    "evidence_failure_reason": None,
                                },
                                "artifact_fingerprints": cluster_fingerprints,
                            },
                        ],
                        "output_paths": {},
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            fake_cluster_result = AsyncMock()
            fake_cluster_result.cluster_count = 2
            fake_cluster_result.strongest_cluster_id = "cluster-1"
            fake_cluster_result.strongest_post_ids = ["a", "b", "c", "d", "e"]
            fake_cluster_result.strongest_cluster_complaint_signal_post_count = 5
            fake_cluster_result.strongest_cluster_screened_post_count = 5
            fake_cluster_result.evidence_validation_passed = True
            fake_cluster_result.evidence_failure_reason = None

            fake_summary_artifact = AsyncMock()
            fake_summary_artifact.candidate_count = 5
            fake_summary_artifact.comment_count = 9
            fake_summary_artifact.selected_comment_count = 4
            fake_summary_artifact.max_posts_used = 5

            fake_memo_artifact = AsyncMock()
            fake_memo_artifact.provider = "lmstudio"
            fake_memo_artifact.model = "openai/gpt-oss-20b"
            fake_memo_artifact.strongest_cluster_id = "cluster-1"
            fake_memo_artifact.strongest_cluster_size = 5
            fake_memo_artifact.included_post_count = 5

            def fake_cluster(*args, **kwargs):
                _write_json_file(
                    run_dir / "theme_summary.json",
                    {
                        "run_dir": str(run_dir),
                        "generated_at": "2026-04-02T00:00:00Z",
                        "source_post_count": 8,
                        "cluster_count": 2,
                        "strongest_cluster_id": "cluster-1",
                        "strongest_post_ids": ["a", "b", "c", "d", "e"],
                        "clusters": [],
                    },
                )
                _write_json_file(
                    run_dir / "cluster_evidence_validation.json",
                    {
                        "strongest_cluster_id": "cluster-1",
                        "strongest_cluster_post_count": 5,
                        "screening_available": True,
                        "screened_cluster_post_count": 5,
                        "complaint_signal_post_count": 5,
                        "min_cluster_complaint_posts": 2,
                        "passes": True,
                        "failure_reason": None,
                    },
                )
                return fake_cluster_result

            async def fake_summarize(*args, **kwargs):
                _write_json_file(run_dir / "comment_selection.json", {"selected_comment_count": 4})
                _write_json_file(run_dir / "evidence_summary.json", {"summary_text": "summary"})
                return fake_summary_artifact

            async def fake_memo(*args, **kwargs):
                _write_json_file(run_dir / "final_memo.json", {"memo_text": "memo"})
                (run_dir / "final_memo.md").write_text("# Final Memo\n", encoding="utf-8")
                return fake_memo_artifact

            with patch.dict(
                "os.environ",
                {
                    "REDDIT_CLIENT_ID": "client-id",
                    "REDDIT_USER_AGENT": "script:test:v1 (by /u/example)",
                    "REDDIT_OUTPUT_ROOT": tmpdir,
                    "LLM_PROVIDER": "lmstudio",
                    "LLM_BASE_URL": "http://127.0.0.1:1234/v1",
                    "LLM_MODEL": "openai/gpt-oss-20b",
                },
                clear=True,
            ):
                with patch(
                    "reddit_pain_agent.main.run_search_command",
                    AsyncMock(),
                ) as search_mock, patch(
                    "reddit_pain_agent.main.enrich_run_with_comments",
                    AsyncMock(),
                ) as comments_mock, patch(
                    "reddit_pain_agent.main.rank_run_candidates",
                ) as rank_mock, patch(
                    "reddit_pain_agent.main.cluster_run_posts",
                    side_effect=fake_cluster,
                ) as cluster_mock, patch(
                    "reddit_pain_agent.main.summarize_candidate_posts",
                    AsyncMock(side_effect=fake_summarize),
                ) as summarize_mock, patch(
                    "reddit_pain_agent.main.write_final_memo",
                    AsyncMock(side_effect=fake_memo),
                ) as memo_mock, patch(
                    "reddit_pain_agent.main.LMStudioClient",
                    FakeLMStudioClient,
                ), patch("sys.stdout.write") as stdout_write:
                    exit_code = main(
                        [
                            "run",
                            "--resume",
                            "--output-dir",
                            str(run_dir),
                            "--subreddit",
                            "Entrepreneur",
                            "--query",
                            "manual follow-up pain",
                            "--similarity-threshold",
                            "0.3",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertEqual(search_mock.await_count, 0)
            self.assertEqual(comments_mock.await_count, 0)
            self.assertEqual(rank_mock.call_count, 0)
            self.assertEqual(cluster_mock.call_count, 1)
            self.assertEqual(summarize_mock.await_count, 1)
            self.assertEqual(memo_mock.await_count, 1)
            output = "".join(call.args[0] for call in stdout_write.call_args_list)
            self.assertIn("resume_from_stage: cluster", output)

    def test_run_cli_resume_reruns_rank_when_selected_posts_fingerprint_changes(self) -> None:
        class FakeLMStudioClient:
            def __init__(self, config):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-1"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json_file(run_dir / "candidate_posts.json", [])
            _write_json_file(run_dir / "comment_enrichment.json", {})
            _write_json_file(run_dir / "comments" / "abc123.json", {"submission_id": "abc123", "comments": []})
            _write_json_file(run_dir / "candidate_screening.json", {"kept_count": 8})
            _write_json_file(run_dir / "post_ranking.json", {"candidate_count": 12})
            _write_json_file(run_dir / "selected_posts.json", {"selected_count": 8})
            _write_json_file(run_dir / "theme_summary.json", {"cluster_count": 3})
            _write_json_file(run_dir / "cluster_evidence_validation.json", {"passes": True})
            search_fingerprints = _fingerprints_for_paths(run_dir, run_dir / "candidate_posts.json")
            comments_fingerprints = _fingerprints_for_paths(
                run_dir,
                run_dir / "comment_enrichment.json",
                run_dir / "comments" / "abc123.json",
            )
            rank_fingerprints = _fingerprints_for_paths(
                run_dir,
                run_dir / "candidate_screening.json",
                run_dir / "post_ranking.json",
                run_dir / "selected_posts.json",
            )
            cluster_fingerprints = _fingerprints_for_paths(
                run_dir,
                run_dir / "theme_summary.json",
                run_dir / "cluster_evidence_validation.json",
            )
            _write_json_file(run_dir / "selected_posts.json", {"selected_count": 99})
            (run_dir / "run_report.json").write_text(
                json.dumps(
                    {
                        "run_slug": "search-20260402-120000-entrepreneur-manual-follow-up",
                        "run_dir": str(run_dir),
                        "status": "failed",
                        "started_at": "2026-04-02T00:00:00Z",
                        "completed_at": "2026-04-02T00:05:00Z",
                        "subreddits": ["Entrepreneur"],
                        "queries": ["manual follow-up pain"],
                        "sort": "relevance",
                        "time_filter": "all",
                        "limit": 25,
                        "provider": None,
                        "model": None,
                        "stop_reason": None,
                        "error": "stale ranking artifact",
                        "stage_reports": [
                            {
                                "stage": "search",
                                "status": "completed",
                                "details": {
                                    "params": {
                                        "subreddits": ["Entrepreneur"],
                                        "queries": ["manual follow-up pain"],
                                        "sort": "relevance",
                                        "time_filter": "all",
                                        "additional_sorts": [],
                                        "additional_time_filters": [],
                                        "limit": 25,
                                        "pages_per_query": 2,
                                        "expand_queries": True,
                                        "min_score": 0,
                                        "min_comments": 0,
                                        "filter_nsfw": False,
                                        "allowed_subreddits": [],
                                        "denied_subreddits": [],
                                    },
                                    "request_count": 4,
                                    "candidate_count": 12,
                                    "query_variant_count": 2,
                                    "search_spec_count": 2,
                                    "sort_count": 1,
                                    "time_filter_count": 1,
                                    "pages_per_query": 2,
                                    "filtered_counts": {},
                                    "run_dir": str(run_dir),
                                },
                                "artifact_fingerprints": search_fingerprints,
                            },
                            {
                                "stage": "comments",
                                "status": "completed",
                                "details": {
                                    "params": {
                                        "max_posts": 5,
                                        "comment_limit": 20,
                                        "comment_depth": 3,
                                        "comment_sort": "best",
                                        "max_morechildren_requests": 3,
                                        "morechildren_batch_size": 20,
                                    },
                                    "requested_submission_count": 5,
                                    "fetched_submission_count": 5,
                                    "comment_count": 14,
                                    "morechildren_request_count": 1,
                                },
                                "artifact_fingerprints": comments_fingerprints,
                            },
                            {
                                "stage": "rank",
                                "status": "completed",
                                "details": {
                                    "params": {
                                        "max_selected_posts": 10,
                                        "min_non_trivial_comments": 0,
                                        "min_complaint_signal_comments": 0,
                                    },
                                    "candidate_count": 12,
                                    "screened_candidate_count": 8,
                                    "rejected_candidate_count": 0,
                                    "selected_count": 8,
                                    "rejection_counts": {},
                                },
                                "artifact_fingerprints": rank_fingerprints,
                            },
                            {
                                "stage": "cluster",
                                "status": "completed",
                                "details": {
                                    "params": {
                                        "similarity_threshold": 0.22,
                                        "min_shared_terms": 2,
                                        "min_cluster_complaint_posts": 2,
                                    },
                                    "cluster_count": 3,
                                    "strongest_cluster_id": "cluster-2",
                                    "strongest_cluster_post_count": 5,
                                    "strongest_cluster_screened_post_count": 5,
                                    "strongest_cluster_complaint_signal_post_count": 5,
                                    "evidence_validation_passed": True,
                                    "evidence_failure_reason": None,
                                },
                                "artifact_fingerprints": cluster_fingerprints,
                            },
                        ],
                        "output_paths": {},
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            fake_ranking_result = AsyncMock()
            fake_ranking_result.run_dir = run_dir
            fake_ranking_result.selected_count = 8
            fake_ranking_result.candidate_count = 12
            fake_ranking_result.screened_candidate_count = 8
            fake_ranking_result.rejected_candidate_count = 0
            fake_ranking_result.rejection_counts = {}

            fake_cluster_result = AsyncMock()
            fake_cluster_result.run_dir = run_dir
            fake_cluster_result.cluster_count = 2
            fake_cluster_result.strongest_cluster_id = "cluster-1"
            fake_cluster_result.strongest_post_ids = ["a", "b", "c", "d", "e"]
            fake_cluster_result.strongest_cluster_complaint_signal_post_count = 5
            fake_cluster_result.strongest_cluster_screened_post_count = 5
            fake_cluster_result.evidence_validation_passed = True
            fake_cluster_result.evidence_failure_reason = None

            fake_summary_artifact = AsyncMock()
            fake_summary_artifact.candidate_count = 5
            fake_summary_artifact.comment_count = 9
            fake_summary_artifact.selected_comment_count = 4
            fake_summary_artifact.max_posts_used = 5

            fake_memo_artifact = AsyncMock()
            fake_memo_artifact.provider = "lmstudio"
            fake_memo_artifact.model = "openai/gpt-oss-20b"
            fake_memo_artifact.strongest_cluster_id = "cluster-1"
            fake_memo_artifact.strongest_cluster_size = 5
            fake_memo_artifact.included_post_count = 5

            def fake_rank(*args, **kwargs):
                _write_json_file(run_dir / "candidate_screening.json", {"kept_count": 8})
                _write_json_file(run_dir / "post_ranking.json", {"candidate_count": 12})
                _write_json_file(run_dir / "selected_posts.json", {"selected_count": 8})
                return fake_ranking_result

            def fake_cluster(*args, **kwargs):
                _write_json_file(
                    run_dir / "theme_summary.json",
                    {
                        "run_dir": str(run_dir),
                        "generated_at": "2026-04-02T00:00:00Z",
                        "source_post_count": 8,
                        "cluster_count": 2,
                        "strongest_cluster_id": "cluster-1",
                        "strongest_post_ids": ["a", "b", "c", "d", "e"],
                        "clusters": [],
                    },
                )
                _write_json_file(
                    run_dir / "cluster_evidence_validation.json",
                    {
                        "strongest_cluster_id": "cluster-1",
                        "strongest_cluster_post_count": 5,
                        "screening_available": True,
                        "screened_cluster_post_count": 5,
                        "complaint_signal_post_count": 5,
                        "min_cluster_complaint_posts": 2,
                        "passes": True,
                        "failure_reason": None,
                    },
                )
                return fake_cluster_result

            async def fake_summarize(*args, **kwargs):
                _write_json_file(run_dir / "comment_selection.json", {"selected_comment_count": 4})
                _write_json_file(run_dir / "evidence_summary.json", {"summary_text": "summary"})
                return fake_summary_artifact

            async def fake_memo(*args, **kwargs):
                _write_json_file(run_dir / "final_memo.json", {"memo_text": "memo"})
                (run_dir / "final_memo.md").write_text("# Final Memo\n", encoding="utf-8")
                return fake_memo_artifact

            with patch.dict(
                "os.environ",
                {
                    "REDDIT_CLIENT_ID": "client-id",
                    "REDDIT_USER_AGENT": "script:test:v1 (by /u/example)",
                    "REDDIT_OUTPUT_ROOT": tmpdir,
                    "LLM_PROVIDER": "lmstudio",
                    "LLM_BASE_URL": "http://127.0.0.1:1234/v1",
                    "LLM_MODEL": "openai/gpt-oss-20b",
                },
                clear=True,
            ):
                with patch(
                    "reddit_pain_agent.main.run_search_command",
                    AsyncMock(),
                ) as search_mock, patch(
                    "reddit_pain_agent.main.enrich_run_with_comments",
                    AsyncMock(),
                ) as comments_mock, patch(
                    "reddit_pain_agent.main.rank_run_candidates",
                    side_effect=fake_rank,
                ) as rank_mock, patch(
                    "reddit_pain_agent.main.cluster_run_posts",
                    side_effect=fake_cluster,
                ) as cluster_mock, patch(
                    "reddit_pain_agent.main.summarize_candidate_posts",
                    AsyncMock(side_effect=fake_summarize),
                ) as summarize_mock, patch(
                    "reddit_pain_agent.main.write_final_memo",
                    AsyncMock(side_effect=fake_memo),
                ) as memo_mock, patch(
                    "reddit_pain_agent.main.LMStudioClient",
                    FakeLMStudioClient,
                ), patch("sys.stdout.write") as stdout_write:
                    exit_code = main(
                        [
                            "run",
                            "--resume",
                            "--output-dir",
                            str(run_dir),
                            "--subreddit",
                            "Entrepreneur",
                            "--query",
                            "manual follow-up pain",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertEqual(search_mock.await_count, 0)
            self.assertEqual(comments_mock.await_count, 0)
            self.assertEqual(rank_mock.call_count, 1)
            self.assertEqual(cluster_mock.call_count, 1)
            self.assertEqual(summarize_mock.await_count, 1)
            self.assertEqual(memo_mock.await_count, 1)
            output = "".join(call.args[0] for call in stdout_write.call_args_list)
            self.assertIn("resume_from_stage: rank", output)

    def test_comments_cli_prints_enrichment_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-1"
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
                fake_result.run_dir = run_dir
                fake_result.requested_submission_count = 5
                fake_result.fetched_submission_count = 4
                fake_result.comment_count = 23
                fake_result.morechildren_request_count = 2
                fake_result.raw_comment_artifacts = ["raw/comments/abc123.json"]
                fake_result.normalized_comment_artifacts = ["comments/abc123.json"]
                with patch(
                    "reddit_pain_agent.main.enrich_run_with_comments",
                    AsyncMock(return_value=fake_result),
                ):
                    with patch("sys.stdout.write") as stdout_write:
                        exit_code = main(["comments", "--run-dir", str(run_dir)])

        self.assertEqual(exit_code, 0)
        output = "".join(call.args[0] for call in stdout_write.call_args_list)
        self.assertIn("requested_submissions: 5", output)
        self.assertIn("saved_comments: 23", output)
        self.assertIn("morechildren_requests: 2", output)

    def test_rank_cli_prints_ranking_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-1"
            fake_result = AsyncMock()
            fake_result.run_dir = run_dir
            fake_result.candidate_count = 12
            fake_result.screened_candidate_count = 5
            fake_result.rejected_candidate_count = 7
            fake_result.rejection_counts = {"low_non_trivial_comments": 7}
            fake_result.selected_count = 5
            with patch("reddit_pain_agent.main.rank_run_candidates", return_value=fake_result):
                with patch("sys.stdout.write") as stdout_write:
                    exit_code = main(["rank", "--run-dir", str(run_dir), "--max-selected-posts", "5"])

        self.assertEqual(exit_code, 0)
        output = "".join(call.args[0] for call in stdout_write.call_args_list)
        self.assertIn("candidate_posts: 12", output)
        self.assertIn("rank_survivors: 5", output)
        self.assertIn("rank_rejected: 7", output)
        self.assertIn("rank_filtered: low_non_trivial_comments=7", output)
        self.assertIn("selected_posts: 5", output)
        self.assertIn("candidate_screening_json:", output)

    def test_cluster_cli_prints_cluster_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-1"
            fake_result = AsyncMock()
            fake_result.run_dir = run_dir
            fake_result.source_post_count = 8
            fake_result.cluster_count = 3
            fake_result.strongest_cluster_id = "cluster-1"
            fake_result.strongest_post_ids = ["a", "b"]
            fake_result.strongest_cluster_complaint_signal_post_count = 2
            fake_result.evidence_validation_passed = True
            fake_result.evidence_failure_reason = None
            with patch("reddit_pain_agent.main.cluster_run_posts", return_value=fake_result):
                with patch("sys.stdout.write") as stdout_write:
                    exit_code = main(["cluster", "--run-dir", str(run_dir)])

        self.assertEqual(exit_code, 0)
        output = "".join(call.args[0] for call in stdout_write.call_args_list)
        self.assertIn("clusters: 3", output)
        self.assertIn("strongest_cluster_id: cluster-1", output)
        self.assertIn("cluster_complaint_posts: 2", output)
        self.assertIn("cluster_evidence_valid: yes", output)
        self.assertIn("cluster_evidence_validation_json:", output)

    def test_llm_models_cli_prints_openai_models(self) -> None:
        class FakeLLMClient:
            def __init__(self, config):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def list_models(self):
                return [type("Model", (), {"id": "openai/gpt-oss-20b", "owned_by": "lmstudio"})()]

        with patch.dict(
            "os.environ",
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-key",
            },
            clear=True,
        ):
            with patch("reddit_pain_agent.main.LMStudioClient", FakeLLMClient):
                with patch("sys.stdout.write") as stdout_write:
                    exit_code = main(["llm", "models"])

        self.assertEqual(exit_code, 0)
        output = "".join(call.args[0] for call in stdout_write.call_args_list)
        self.assertIn("provider: openai", output)
        self.assertIn("openai/gpt-oss-20b", output)

    def test_llm_prompt_cli_prints_generated_text(self) -> None:
        class FakeLLMClient:
            def __init__(self, config):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def generate_text(self, prompt, model=None):
                return f"echo:{prompt}:{model or self.config.model}"

        with patch.dict(
            "os.environ",
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-key",
                "LLM_MODEL": "gpt-5.2",
            },
            clear=True,
        ):
            with patch("reddit_pain_agent.main.LMStudioClient", FakeLLMClient):
                with patch("sys.stdout.write") as stdout_write:
                    exit_code = main(["llm", "prompt", "--prompt", "test prompt"])

        self.assertEqual(exit_code, 0)
        output = "".join(call.args[0] for call in stdout_write.call_args_list)
        self.assertIn("provider: openai", output)
        self.assertIn("model: gpt-5.2", output)
        self.assertIn("echo:test prompt:gpt-5.2", output)

    def test_summarize_cli_writes_summary_artifacts(self) -> None:
        class FakeLMStudioClient:
            def __init__(self, config):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def generate_response(self, prompt, model=None):
                return type(
                    "Generation",
                    (),
                    {
                        "provider": "lmstudio",
                        "model": model or self.config.model,
                        "prompt": prompt,
                        "output_text": "## Repeated Complaints\n\nManual follow-up keeps showing up.",
                        "raw_response": {"id": "resp_1", "output_text": "Manual follow-up keeps showing up."},
                    },
                )()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-1"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "candidate_posts.json").write_text(
                (
                    '[{"id":"raw1","title":"Raw post that should be ignored",'
                    '"subreddit":"Entrepreneur","url":"https://reddit.com/raw",'
                    '"permalink":"/r/Entrepreneur/comments/raw/example/",'
                    '"selftext":"raw order item","score":1,"num_comments":1},'
                    '{"id":"abc123","title":"Manual client follow-up is eating my week",'
                    '"subreddit":"Entrepreneur","url":"https://reddit.com/example",'
                    '"permalink":"/r/Entrepreneur/comments/abc123/example/",'
                    '"selftext":"Still tracking leads in spreadsheets.","score":42,"num_comments":18}]'
                ),
                encoding="utf-8",
            )
            (run_dir / "selected_posts.json").write_text(
                (
                    '[{"candidate":{"id":"other1","title":"Another shortlisted post",'
                    '"subreddit":"Entrepreneur","url":"https://reddit.com/other1",'
                    '"permalink":"/r/Entrepreneur/comments/other1/example/",'
                    '"selftext":"not in strongest cluster","score":30,"num_comments":9},'
                    '"saved_comment_count":0,"breakdown":{"total_score":7.0},"rank":1},'
                    '{"candidate":{"id":"abc123","title":"Manual client follow-up is eating my week",'
                    '"subreddit":"Entrepreneur","url":"https://reddit.com/example",'
                    '"permalink":"/r/Entrepreneur/comments/abc123/example/",'
                    '"selftext":"Still tracking leads in spreadsheets.","score":42,"num_comments":18},'
                    '"saved_comment_count":1,"breakdown":{"total_score":9.0},"rank":2}]'
                ),
                encoding="utf-8",
            )
            (run_dir / "theme_summary.json").write_text(
                (
                    '{"run_dir":"x","generated_at":"2026-04-02T00:00:00Z","source_post_count":2,'
                    '"cluster_count":2,"strongest_cluster_id":"cluster-2","strongest_post_ids":["abc123"],'
                    '"clusters":[{"cluster_id":"cluster-2","label":"crm / follow-up","post_ids":["abc123"],"size":1}]}'
                ),
                encoding="utf-8",
            )
            (run_dir / "comments").mkdir(parents=True, exist_ok=True)
            (run_dir / "comments" / "abc123.json").write_text(
                (
                    '{"submission_id":"abc123","subreddit":"Entrepreneur","title":"Manual client follow-up is eating my week",'
                    '"fetched_comment_count":1,"comments":[{"id":"c1","body":"I am still doing this manually","score":6,"depth":0}]}'
                ),
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {
                    "LLM_PROVIDER": "lmstudio",
                    "LLM_BASE_URL": "http://127.0.0.1:1234/v1",
                    "LLM_MODEL": "openai/gpt-oss-20b",
                },
                clear=True,
            ):
                with patch("reddit_pain_agent.main.LMStudioClient", FakeLMStudioClient):
                    with patch("sys.stdout.write") as stdout_write:
                        exit_code = main(["summarize", "--run-dir", str(run_dir)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((run_dir / "comment_selection.json").exists())
            self.assertTrue((run_dir / "evidence_summary.json").exists())
            self.assertTrue((run_dir / "evidence_summary.md").exists())
            self.assertTrue((run_dir / "prompts" / "candidate-evidence-summary.txt").exists())
            self.assertTrue((run_dir / "raw" / "llm" / "candidate-evidence-summary.json").exists())
            registry = json.loads((run_dir / "asset_registry.json").read_text(encoding="utf-8"))
            registry_by_path = {item["artifact_path"]: item for item in registry}
            self.assertEqual(
                registry_by_path["evidence_summary.json"]["generation"],
                {
                    "provider": "lmstudio",
                    "model": "openai/gpt-oss-20b",
                    "prompt_artifact_path": "prompts/candidate-evidence-summary.txt",
                    "raw_response_artifact_path": "raw/llm/candidate-evidence-summary.json",
                },
            )

        output = "".join(call.args[0] for call in stdout_write.call_args_list)
        self.assertIn("comment_selection_json:", output)
        self.assertIn("summary_json:", output)
        self.assertIn("candidate_posts: 1", output)
        self.assertIn("saved_comments: 1", output)
        self.assertIn("selected_comments: 1", output)

    def test_memo_cli_writes_final_memo_artifacts(self) -> None:
        class FakeLMStudioClient:
            def __init__(self, config):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def generate_response(self, prompt, model=None):
                return type(
                    "Generation",
                    (),
                    {
                        "provider": "lmstudio",
                        "model": model or self.config.model,
                        "prompt": prompt,
                        "output_text": (
                            "# Executive Summary\n\nThe theme is strong.\n\n"
                            "## Research Takeaways\n\nPain is repeated.\n\n"
                            "## Top 5 Product Ideas\n\n1. Idea one\n2. Idea two\n3. Idea three\n4. Idea four\n5. Idea five\n\n"
                            "## Best Single Bet\n\nBuild the follow-up layer.\n\n"
                            "## 10 Content Hooks\n\n"
                            "1. Hook 1\n2. Hook 2\n3. Hook 3\n4. Hook 4\n5. Hook 5\n6. Hook 6\n7. Hook 7\n8. Hook 8\n9. Hook 9\n10. Hook 10\n\n"
                            "## Risks / Caveats\n\nReddit can over-index edge cases."
                        ),
                        "raw_response": {"id": "resp_memo_1"},
                    },
                )()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-1"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "selected_posts.json").write_text(
                (
                    '[{"candidate":{"id":"a","title":"Manual follow-up in spreadsheets is painful","subreddit":"Entrepreneur","url":"https://reddit.com/a","selftext":"CRM setup feels too heavy","score":30,"num_comments":11},"saved_comment_count":2,"breakdown":{"total_score":8.0},"rank":1},'
                    '{"candidate":{"id":"b","title":"Spreadsheet lead tracking keeps breaking follow-up","subreddit":"Entrepreneur","url":"https://reddit.com/b","selftext":"Still doing manual reminders every week","score":28,"num_comments":10},"saved_comment_count":1,"breakdown":{"total_score":7.8},"rank":2},'
                    '{"candidate":{"id":"c","title":"CRM follow-up falls apart without manual reminders","subreddit":"Entrepreneur","url":"https://reddit.com/c","selftext":"We keep missing replies","score":24,"num_comments":8},"saved_comment_count":1,"breakdown":{"total_score":7.4},"rank":3},'
                    '{"candidate":{"id":"d","title":"Manual lead follow-up is killing our process","subreddit":"Entrepreneur","url":"https://reddit.com/d","selftext":"Everything lives in a spreadsheet","score":26,"num_comments":9},"saved_comment_count":1,"breakdown":{"total_score":7.3},"rank":4},'
                    '{"candidate":{"id":"e","title":"Client follow-up still depends on spreadsheets and CRM hacks","subreddit":"Entrepreneur","url":"https://reddit.com/e","selftext":"Too much copy and paste","score":23,"num_comments":7},"saved_comment_count":1,"breakdown":{"total_score":7.1},"rank":5}]'
                ),
                encoding="utf-8",
            )
            (run_dir / "theme_summary.json").write_text(
                (
                    '{"run_dir":"x","generated_at":"2026-04-02T00:00:00Z","source_post_count":5,'
                    '"cluster_count":1,"strongest_cluster_id":"cluster-1","strongest_post_ids":["a","b","c","d","e"],'
                    '"clusters":[{"cluster_id":"cluster-1","label":"crm / follow-up / spreadsheet","post_ids":["a","b","c","d","e"],'
                    '"size":5,"average_post_score":7.52,"total_comment_count":45,'
                    '"top_terms":["crm","follow-up","spreadsheet"],"member_ranks":[1,2,3,4,5],"cohesion_score":0.34}]}'
                ),
                encoding="utf-8",
            )
            (run_dir / "evidence_summary.json").write_text(
                (
                    '{"run_dir":"x","generated_at":"2026-04-02T00:00:00Z","provider":"lmstudio",'
                    '"model":"openai/gpt-oss-20b","candidate_count":5,"comment_count":12,"selected_comment_count":6,'
                    '"max_posts_used":5,"prompt_artifact_path":"prompts/candidate-evidence-summary.txt",'
                    '"raw_response_artifact_path":"raw/llm/candidate-evidence-summary.json",'
                    '"summary_markdown_artifact_path":"evidence_summary.md",'
                    '"summary_text":"## Repeated Complaints\\n\\nManual follow-up and spreadsheet-heavy CRM workflows keep failing."}'
                ),
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {
                    "LLM_PROVIDER": "lmstudio",
                    "LLM_BASE_URL": "http://127.0.0.1:1234/v1",
                    "LLM_MODEL": "openai/gpt-oss-20b",
                },
                clear=True,
            ):
                with patch("reddit_pain_agent.main.LMStudioClient", FakeLMStudioClient):
                    with patch("sys.stdout.write") as stdout_write:
                        exit_code = main(["memo", "--run-dir", str(run_dir)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((run_dir / "final_memo.json").exists())
            self.assertTrue((run_dir / "final_memo.md").exists())
            self.assertTrue((run_dir / "prompts" / "final-memo.txt").exists())
            self.assertTrue((run_dir / "raw" / "llm" / "final-memo.json").exists())
            registry = json.loads((run_dir / "asset_registry.json").read_text(encoding="utf-8"))
            registry_by_path = {item["artifact_path"]: item for item in registry}
            self.assertEqual(
                registry_by_path["final_memo.md"]["generation"],
                {
                    "provider": "lmstudio",
                    "model": "openai/gpt-oss-20b",
                    "prompt_artifact_path": "prompts/final-memo.txt",
                    "raw_response_artifact_path": "raw/llm/final-memo.json",
                },
            )

        output = "".join(call.args[0] for call in stdout_write.call_args_list)
        self.assertIn("strongest_cluster_id: cluster-1", output)
        self.assertIn("strongest_cluster_size: 5", output)
        self.assertIn("final_memo_markdown:", output)

    def test_reply_drafts_cli_writes_reply_artifacts(self) -> None:
        class FakeLMStudioClient:
            def __init__(self, config):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def generate_response(self, prompt, model=None):
                if "Evaluate the reply drafts below" in prompt:
                    payload = {
                        "provider": "lmstudio",
                        "model": model or self.config.model,
                        "prompt": prompt,
                        "output_text": json.dumps(
                            {
                                "evaluations": [
                                    {
                                        "post_id": "abc123",
                                        "relevance_score": 4,
                                        "conversation_value_score": 4,
                                        "voice_match_score": 4,
                                        "reddit_friendliness_score": 4,
                                        "feedback": "Good.",
                                    }
                                ]
                            }
                        ),
                        "raw_response": {"id": "resp_eval_cli"},
                    }
                elif "Revise the reply drafts below" in prompt:
                    payload = {
                        "provider": "lmstudio",
                        "model": model or self.config.model,
                        "prompt": prompt,
                        "output_text": (
                            "## Post 1\n"
                            "post_id: abc123\n"
                            "reply: I can see why this is frustrating. When a process keeps collapsing back into manual follow-up, it usually means the system is still too brittle for day-to-day use.\n\n"
                            "The part I would focus on is the trust gap, because once people start keeping backup habits around the workflow the system is already losing.\n"
                        ),
                        "raw_response": {"id": "resp_reply_cli_2"},
                    }
                else:
                    payload = {
                        "provider": "lmstudio",
                        "model": model or self.config.model,
                        "prompt": prompt,
                        "output_text": (
                            "## Post 1\n"
                            "post_id: abc123\n"
                            "reply: I can see why this is frustrating.\n"
                        ),
                        "raw_response": {"id": "resp_reply_cli_1"},
                    }
                return type("Generation", (), payload)()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-1"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "selected_posts.json").write_text(
                (
                    '[{"candidate":{"id":"abc123","title":"Manual client follow-up is eating my week",'
                    '"subreddit":"Entrepreneur","url":"https://reddit.com/example","selftext":"Still tracking leads in spreadsheets."},'
                    '"saved_comment_count":0,"breakdown":{"total_score":7.0},"rank":1}]'
                ),
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {
                    "LLM_PROVIDER": "lmstudio",
                    "LLM_BASE_URL": "http://127.0.0.1:1234/v1",
                    "LLM_MODEL": "openai/gpt-oss-20b",
                },
                clear=True,
            ):
                with patch("reddit_pain_agent.main.LMStudioClient", FakeLMStudioClient):
                    with patch("sys.stdout.write") as stdout_write:
                        exit_code = main(
                            [
                                "reply-drafts",
                                "--run-dir",
                                str(run_dir),
                                "--voice",
                                "plainspoken founder",
                                "--max-posts",
                                "1",
                            ]
                        )

            self.assertEqual(exit_code, 0)
            self.assertTrue((run_dir / "reply_drafts.json").exists())
            self.assertTrue((run_dir / "reply_drafts.md").exists())
            registry = json.loads((run_dir / "asset_registry.json").read_text(encoding="utf-8"))
            registry_by_path = {item["artifact_path"]: item for item in registry}
            self.assertEqual(
                registry_by_path["reply_drafts.json"]["generation"],
                {
                    "provider": "lmstudio",
                    "model": "openai/gpt-oss-20b",
                    "prompt_artifact_path": "prompts/reply-drafts.txt",
                    "raw_response_artifact_path": "raw/llm/reply-drafts.json",
                },
            )

        output = "".join(call.args[0] for call in stdout_write.call_args_list)
        self.assertIn("manual_review_only: yes", output)
        self.assertIn("reply_drafts_json:", output)
        self.assertIn("selected_posts: 1", output)
        self.assertIn("passed_threshold: yes", output)

    def test_comment_opportunities_cli_writes_scored_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-1"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "selected_posts.json").write_text(
                (
                    '[{"candidate":{"id":"strong","title":"We still miss leads because manual follow-up is killing our workflow",'
                    '"subreddit":"Entrepreneur","url":"https://reddit.com/strong","selftext":"I need to fix this now.",'
                    '"source_queries":["manual follow-up pain"],"created_utc":1775450000},'
                    '"saved_comment_count":0,"breakdown":{"total_score":7.0},"rank":1}]'
                ),
                encoding="utf-8",
            )
            (run_dir / "comments").mkdir(parents=True, exist_ok=True)
            (run_dir / "comments" / "strong.json").write_text(
                (
                    '{"submission_id":"strong","subreddit":"Entrepreneur","title":"x","fetched_comment_count":2,'
                    '"comments":[{"id":"c1","body":"We still do this manually and it slows everything down.","score":5,"depth":0},'
                    '{"id":"c2","body":"Same issue here, we keep missing people because the workflow breaks.","score":4,"depth":0}]}'
                ),
                encoding="utf-8",
            )

            with patch("sys.stdout.write") as stdout_write:
                exit_code = main(
                    [
                        "comment-opportunities",
                        "--run-dir",
                        str(run_dir),
                        "--max-posts",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((run_dir / "comment_opportunities.json").exists())

        output = "".join(call.args[0] for call in stdout_write.call_args_list)
        self.assertIn("scored_posts: 1", output)
        self.assertIn("opportunity: post_id=strong", output)
        self.assertIn("comment_opportunities_json:", output)

    def test_comment_enrichment_expands_morechildren_bounded(self) -> None:
        class FakeRedditClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def search_subreddit(self, spec):
                raise AssertionError("not used")

            async def fetch_submission_comments(self, permalink, sort="best", limit=20, depth=3):
                payload = [
                    {"kind": "Listing", "data": {"children": []}},
                    {
                        "kind": "Listing",
                        "data": {
                            "children": [
                                {
                                    "kind": "t1",
                                    "data": {"id": "c1", "body": "Top level comment", "score": 5, "depth": 0},
                                },
                                {
                                    "kind": "more",
                                    "data": {"children": ["c2", "c3"]},
                                },
                            ]
                        },
                    },
                ]
                return payload, type(
                    "LogEntry",
                    (),
                    {
                        "raw_artifact_path": None,
                        "model_dump": lambda self, mode="json": {
                            "requested_at": "2026-04-02T00:00:00Z",
                            "request_name": "comments:test",
                            "method": "GET",
                            "url": "https://oauth.reddit.com/test",
                            "params": {},
                            "status_code": 200,
                            "duration_ms": 10.0,
                            "attempt": 1,
                            "rate_limit": None,
                            "raw_artifact_path": self.raw_artifact_path,
                            "error": None,
                        },
                    },
                )()

            async def fetch_more_children(self, *, link_id, children, sort="best", depth=3, limit_children=True):
                self.last_children = children
                payload = {
                    "json": {
                        "data": {
                            "things": [
                                {
                                    "kind": "t1",
                                    "data": {"id": "c2", "body": "Expanded reply", "score": 4, "depth": 1},
                                },
                                {
                                    "kind": "t1",
                                    "data": {"id": "c3", "body": "Another expanded reply", "score": 3, "depth": 1},
                                },
                            ]
                        }
                    }
                }
                return payload, type(
                    "LogEntry",
                    (),
                    {
                        "raw_artifact_path": None,
                        "model_dump": lambda self, mode="json": {
                            "requested_at": "2026-04-02T00:00:01Z",
                            "request_name": "morechildren:test",
                            "method": "GET",
                            "url": "https://oauth.reddit.com/api/morechildren",
                            "params": {"children": ",".join(children)},
                            "status_code": 200,
                            "duration_ms": 9.0,
                            "attempt": 1,
                            "rate_limit": None,
                            "raw_artifact_path": self.raw_artifact_path,
                            "error": None,
                        },
                    },
                )()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-1"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "candidate_posts.json").write_text(
                (
                    '[{"id":"abc123","title":"Manual client follow-up is eating my week",'
                    '"subreddit":"Entrepreneur","url":"https://reddit.com/example",'
                    '"permalink":"/r/Entrepreneur/comments/abc123/example/","score":42,"num_comments":18}]'
                ),
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {
                    "REDDIT_CLIENT_ID": "client-id",
                    "REDDIT_USER_AGENT": "script:test:v1 (by /u/example)",
                    "REDDIT_OUTPUT_ROOT": tmpdir,
                },
                clear=True,
            ):
                config = load_runtime_config()
                result = asyncio.run(
                    enrich_run_with_comments(
                        config=config,
                        run_dir=run_dir,
                        max_posts=1,
                        max_morechildren_requests=1,
                        morechildren_batch_size=10,
                        client=FakeRedditClient(),
                    )
                )

            self.assertEqual(result.comment_count, 3)
            self.assertEqual(result.morechildren_request_count, 1)
            self.assertTrue((run_dir / "comments" / "abc123.json").exists())


if __name__ == "__main__":
    unittest.main()
