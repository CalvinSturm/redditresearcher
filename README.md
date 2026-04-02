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

- `outputs/runs/<run-slug>/candidate_posts.json`
- `outputs/runs/<run-slug>/selected_posts.json`
- `outputs/runs/<run-slug>/theme_summary.json`
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
- raw search artifact persistence per request
- normalized `candidate_posts.json`
- run manifest and request log persistence

Install the package locally before using the module entrypoint:

```bash
python -m pip install -e .
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

Example:

```bash
python -m reddit_pain_agent.main search \
  --subreddit vibecoding \
  --subreddit Entrepreneur \
  --query "manual outreach pain" \
  --query "clients falling through cracks" \
  --sort comments \
  --time-filter month \
  --limit 25
```

Current limitations:

- ranking is not implemented
- clustering is not implemented
- comments are not fetched yet
- stored post text should be treated as inspectable run data, not permanent archival data
