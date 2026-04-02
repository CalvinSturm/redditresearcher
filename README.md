# Reddit Pain Agent

A CLI-first research agent that searches Reddit communities, finds repeated customer pain points from high-engagement discussions, groups related posts into themes, and turns them into founder-grade research memos.

## What this project does

For each research run, the agent:

1. Searches one or more subreddits for relevant discussions
2. Pulls posts plus comment evidence
3. Ranks candidates by relevance and engagement
4. Clusters them into common themes
5. Selects at least 5 strongly related posts from the best theme
6. Produces a memo with:
   - research takeaways
   - top 5 product ideas
   - best single bet
   - 10 content hooks
   - risks / caveats

This is not a generic Reddit summarizer.
It is a market research and product discovery agent.

## Primary use cases

- discover repeated customer pain points
- identify founder or business bottlenecks
- generate product wedges from public discussion
- generate content ideas grounded in real audience pain
- build thesis-level research from Reddit conversations

## Initial target subreddits

- r/vibecoding
- r/Entrepreneur

The system should be extensible to more subreddits later.

## Categories

The agent supports three initial categories:

- `software`
- `ergonomics`
- `business`

## Expected output

A successful run outputs:

- `outputs/runs/<run-slug>/run_report.json`
- `outputs/runs/<run-slug>/candidate_posts.json`
- `outputs/runs/<run-slug>/comment_enrichment.json`
- `outputs/runs/<run-slug>/candidate_screening.json`
- `outputs/runs/<run-slug>/selected_posts.json`
- `outputs/runs/<run-slug>/theme_summary.json`
- `outputs/runs/<run-slug>/cluster_evidence_validation.json`
- `outputs/runs/<run-slug>/final_memo.md`

## Recommended repo layout

```text
src/reddit_pain_agent/
tests/
research/
  briefs/
  queries/
templates/
outputs/
  runs/
```

This keeps:

- implementation code under `src/`
- repeatable run inputs under `research/`
- reusable output templates under `templates/`
- run-specific artifacts under `outputs/runs/<run-slug>/`

## Bootstrap a run

Create the repo layout:

```bash
python -m reddit_pain_agent.main layout
```

Create a new run scaffold:

```bash
python -m reddit_pain_agent.main init-run "vibecoding onboarding pain"
```

That generates:

- a run brief in `research/briefs/`
- a run output folder in `outputs/runs/`
- a memo shell in the run folder for the final write-up

## Quality bar

The memo should read like a pragmatic founder / product strategist:
- concise
- specific
- evidence-linked
- realistic
- useful for deciding what to build or write next

## Core design principles

- retrieval first
- deterministic where possible
- inspectable ranking and clustering
- LLM only for synthesis and optional semantic cleanup
- no fabricated Reddit data
- no fake “theme” grouping just to satisfy quotas

## Example CLI

### Search only

```bash
python -m reddit_pain_agent.main search \
  --subreddit vibecoding \
  --query "pain points in coding workflows" \
  --limit 25
```

## Retrieval slice

The current vertical slice implements:

- OAuth-authenticated subreddit search using the Reddit Data API
- Playwright capture of Reddit search and thread pages into the manual import schema
- manual or Playwright-collected JSON import as a first-class retrieval fallback
- paginated search retrieval across multiple result pages per subreddit/query variant
- deterministic query expansion for broader lexical coverage
- deterministic multi-sort and multi-time-filter search planning
- explicit comment enrichment for shortlisted submissions
- deterministic discussion-depth screening before ranking
- deterministic post ranking and shortlist selection
- deterministic theme clustering over the ranked shortlist
- deterministic strongest-cluster evidence validation before synthesis
- explicit end-to-end `run` orchestration from search through final memo generation
- raw search artifact persistence per request
- normalized `candidate_posts.json`
- run manifest and request log persistence

Install the package locally before using the module entrypoint:

```bash
python -m pip install -e .
```

If you want to use the Playwright capture command, install a browser once:

```bash
python -m playwright install chromium
```

Required environment variables:

- `REDDIT_CLIENT_ID`
- `REDDIT_USER_AGENT`

Optional environment variables:

- `REDDIT_CLIENT_SECRET`
- `REDDIT_OUTPUT_ROOT`
- `REDDIT_REQUEST_TIMEOUT_SECONDS`
- `REDDIT_MAX_RETRIES`
- `REDDIT_MAX_CONCURRENT_REQUESTS`

If you do not have Reddit API credentials yet, use the manual import path below instead of `search`.

Example:

```bash
python -m reddit_pain_agent.main search \
  --subreddit vibecoding \
  --subreddit Entrepreneur \
  --query "manual outreach pain" \
  --query "clients falling through cracks" \
  --additional-sort new \
  --additional-time-filter week \
  --pages-per-query 2 \
  --min-score 3 \
  --min-comments 5 \
  --filter-nsfw \
  --sort comments \
  --time-filter month \
  --limit 25
```

Search coverage defaults:

- query expansion is enabled by default for broader lexical coverage
- additional sorts and time filters are composed into a deterministic search plan
- pagination fetches up to `2` result pages per subreddit/query variant by default
- weak candidates can be rejected before ranking with `--min-score`, `--min-comments`, `--filter-nsfw`, `--allow-subreddit`, and `--deny-subreddit`
- use `--no-expand-queries` to search only the exact queries you provide
- use `--additional-sort <sort>` and `--additional-time-filter <window>` to widen coverage explicitly
- use `--pages-per-query <n>` to tighten or widen paginated retrieval depth
- filtered candidates are counted explicitly in the search summary and persisted in run artifacts for inspection

### Manual or Playwright fallback

If Reddit API access is unavailable, you can import manually collected posts and comments into the same run artifact format the rest of the pipeline expects.

```bash
python -m reddit_pain_agent.main manual-import \
  --input research/manual/manual-follow-up.json \
  --subreddit Entrepreneur \
  --query "manual follow-up pain" \
  --output-dir outputs/runs/manual-follow-up
```

Accepted input shape:

```json
{
  "posts": [
    {
      "id": "abc123",
      "title": "Manual follow-up is killing our process",
      "subreddit": "Entrepreneur",
      "url": "https://www.reddit.com/r/Entrepreneur/comments/abc123/example/",
      "permalink": "/r/Entrepreneur/comments/abc123/example/",
      "score": 42,
      "num_comments": 18,
      "created_utc": 1712073600,
      "selftext": "We still track leads in spreadsheets.",
      "comments": [
        {
          "id": "c1",
          "body": "Same issue here",
          "score": 6,
          "depth": 0
        }
      ]
    }
  ]
}
```

Notes:

- the file can also be a top-level JSON array of posts
- Tampermonkey/userscript exports are also accepted directly when they use fields like `post_id`, `body_full`, `comments_full`, and `comment_id`
- imported posts are still deduplicated and filtered with `--min-score`, `--min-comments`, `--filter-nsfw`, `--allow-subreddit`, and `--deny-subreddit`
- imported comments are written straight to `comments/*.json` plus `comment_enrichment.json`
- the raw input file is copied into `raw/manual/` for inspection
- this path does not fabricate request logs or API responses; it only preserves the supplied input

### Playwright capture

To collect a bundle directly from Reddit pages with Playwright:

```bash
python -m reddit_pain_agent.main capture \
  --subreddit Entrepreneur \
  --query "manual follow-up pain" \
  --sort comments \
  --time-filter month \
  --max-posts 5 \
  --max-comments 12
```

Useful flags:

- `--select-result <n>` captures specific 1-based search-result indices instead of the top results
- `--thread-url <url>` captures known thread URLs in addition to discovered search results
- `--skip-search` captures only the provided `--thread-url` values
- `--show-browser` launches Chromium visibly instead of headless mode
- `--output-json <path>` writes the import bundle to an explicit file

Handoff examples:

```bash
python -m reddit_pain_agent.main capture \
  --subreddit Entrepreneur \
  --query "manual follow-up pain" \
  --handoff manual-import \
  --output-dir outputs/runs/manual-follow-up
```

```bash
python -m reddit_pain_agent.main capture \
  --subreddit Entrepreneur \
  --query "manual follow-up pain" \
  --handoff run \
  --output-dir outputs/runs/manual-follow-up \
  --model openai/gpt-oss-20b
```

Capture behavior:

- the command visits subreddit search pages, extracts candidate thread links, then visits selected thread pages
- it saves a JSON bundle in the manual import schema, including captured comments when visible on the page
- it writes a persistent session log next to the bundle, for example `outputs/captures/<slug>.log`
- it writes HTML and PNG snapshots for each visited page under a sibling snapshot directory, for example `outputs/captures/<slug>/`
- when `--handoff manual-import` is used, the bundle is immediately imported into normal run artifacts
- when `--handoff run` is used, the bundle is passed straight into `run --manual-input ...`

Fetch comment evidence for the top candidate posts in a run:

```bash
python -m reddit_pain_agent.main comments \
  --run-dir outputs/runs/<run-slug> \
  --max-posts 5 \
  --comment-limit 20 \
  --comment-depth 3 \
  --max-morechildren-requests 3
```

Rank candidate posts and persist a shortlist:

```bash
python -m reddit_pain_agent.main rank \
  --run-dir outputs/runs/<run-slug> \
  --max-selected-posts 10 \
  --min-nontrivial-comments 2 \
  --min-complaint-signal-comments 2
```

Ranking discussion-depth gates:

- `--min-nontrivial-comments <n>` requires at least `n` saved comments that are more than low-signal agreement replies
- `--min-complaint-signal-comments <n>` requires at least `n` saved comments that contain first-person complaint/workflow language
- ranking writes `candidate_screening.json` before it writes `post_ranking.json`, so you can inspect which posts were rejected and why

Cluster the ranked shortlist into repeated pain themes:

```bash
python -m reddit_pain_agent.main cluster \
  --run-dir outputs/runs/<run-slug> \
  --similarity-threshold 0.22 \
  --min-shared-terms 2 \
  --min-cluster-complaint-posts 2
```

Cluster evidence validation:

- the cluster stage now writes `cluster_evidence_validation.json`
- it counts how many strongest-cluster posts have complaint-signal comment evidence from `candidate_screening.json`
- `run` stops before summarize/memo when the strongest cluster has too little complaint-signal evidence, even if the cluster is large enough by raw post count

Current limitations:

- comment enrichment only expands `MoreComments` in a bounded way
- query expansion is lexical and deterministic, not semantic
- multi-sort and multi-time-window planning increases coverage, but it can also increase duplicate retrieval volume
- retrieval-quality thresholds are blunt pre-ranking gates; they improve signal quality, but they can also hide edge-case posts if set too aggressively
- discussion-depth gating depends on saved comments, so standalone `rank` runs with no `comments/` artifacts will screen as if there is no discussion evidence
- cluster evidence validation depends on `candidate_screening.json`, so stale or missing ranking artifacts can invalidate synthesis even when `theme_summary.json` exists
- the Playwright capture stage relies on Reddit’s current DOM structure, so selectors may need periodic updates if the site changes
- Playwright capture saves only what is visible on the page; it does not expand every hidden or lazy-loaded comment branch
- capture logs and snapshots are written locally for debugging, but they can contain page text and should be treated as inspectable run data
- stored post text should be treated as inspectable run data, not permanent archival data

## LM Studio provider

The repo now includes a minimal LLM provider layer for `lmstudio`.

This support targets LM Studio's OpenAI-compatible local server endpoints:

- `GET /v1/models`
- `POST /v1/responses`

Recommended environment variables:

- `LLM_PROVIDER=lmstudio`
- `LLM_BASE_URL=http://127.0.0.1:1234/v1`
- `LLM_MODEL=<loaded-model-id>`
- `LLM_API_KEY=<optional if you enabled auth>`

List available models:

```bash
python -m reddit_pain_agent.main llm models
```

Run a one-shot prompt:

```bash
python -m reddit_pain_agent.main llm prompt \
  --prompt "Summarize the main complaint in one sentence."
```

Current scope:

- provider support is limited to `lmstudio`
- LLM usage is wired into `run`, `summarize`, and `memo`
- retrieval, ranking, and clustering remain deterministic code paths

## End-to-end run

To execute the full pipeline in one command:

```bash
python -m reddit_pain_agent.main run \
  --subreddit Entrepreneur \
  --query "manual follow-up pain" \
  --query "spreadsheet CRM pain" \
  --sort comments \
  --time-filter month \
  --limit 25
```

This orchestrates:

1. `search`
2. `comments`
3. `rank`
4. `cluster`
5. `summarize`
6. `memo`

To run the same loop from manual or Playwright-collected input instead of Reddit API search:

```bash
python -m reddit_pain_agent.main run \
  --manual-input research/manual/manual-follow-up.json \
  --output-dir outputs/runs/manual-follow-up \
  --subreddit Entrepreneur \
  --query "manual follow-up pain"
```

Important behavior:

- the run stops cleanly if the strongest cluster has fewer than `--min-cluster-posts` related posts
- the run also stops cleanly if the strongest cluster has too few complaint-signal posts for `--min-cluster-complaint-posts`
- when that happens, the deterministic artifacts are still written, but no memo is fabricated
- `run` uses the configured LM Studio provider only after the cluster passes that threshold
- `run` always writes `run_report.json` once a run directory exists, including per-stage status, timings, stop reason, and key artifact paths
- `run --manual-input ...` skips Reddit credential loading and reuses imported comment artifacts instead of calling the Reddit API

To resume an interrupted or stopped run in place:

```bash
python -m reddit_pain_agent.main run \
  --resume \
  --output-dir outputs/runs/<run-slug> \
  --subreddit Entrepreneur \
  --query "manual follow-up pain"
```

Resume behavior:

- completed stages are reused only when their artifacts still exist
- completed stages are reused only when the recorded stage parameters still match the current CLI flags
- completed stages are reused only when the recorded artifact fingerprints still match the current on-disk artifacts
- if a later-stage parameter changes, `run` resumes from that stage onward instead of trusting stale downstream artifacts
- older `run_report.json` files without stage fingerprints are treated as non-reusable and will rerun from the first affected stage

## Candidate summarization

After a `search` run creates `candidate_posts.json`, you can summarize that evidence into a grounded artifact with LM Studio.

```bash
python -m reddit_pain_agent.main summarize \
  --run-dir outputs/runs/<run-slug>
```

Optional flags:

- `--model <model-id>` to override `LLM_MODEL`
- `--max-posts 10` to cap how many candidate posts are included in the prompt

This writes:

- `comment_enrichment.json` if you ran the explicit `comments` stage
- `post_ranking.json`
- `selected_posts.json`
- `theme_summary.json`
- `comment_selection.json`
- `evidence_summary.json`
- `evidence_summary.md`
- `prompts/candidate-evidence-summary.txt`
- `raw/llm/candidate-evidence-summary.json`

Current limitations:

- this is an evidence extraction stage, not the final memo
- the summary uses a deterministic scored subset of saved comments when available, otherwise it falls back to post-only evidence
- the summary prefers the strongest theme cluster when present, then `selected_posts.json`, then raw candidate order

## Final memo generation

After `cluster` and `summarize`, you can write the grounded final memo:

```bash
python -m reddit_pain_agent.main memo \
  --run-dir outputs/runs/<run-slug>
```

Optional flags:

- `--model <model-id>` to override `LLM_MODEL`
- `--min-cluster-posts 5` to enforce the minimum themed post count before memo generation
- `--max-posts 8` to cap how many strongest-cluster posts are included in the memo prompt

This writes:

- `final_memo.json`
- `final_memo.md`
- `prompts/final-memo.txt`
- `raw/llm/final-memo.json`

Current limitations:

- memo generation fails if the strongest cluster has fewer than 5 related posts
- memo synthesis is LLM-backed, but cluster selection and post selection remain deterministic
