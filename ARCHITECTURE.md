# Architecture

## System goal

Build a Reddit research agent that finds repeated pain signals from high-engagement discussions and turns them into founder-grade product and content opportunities.

## High-level pipeline

The runtime pipeline is:

1. Query expansion
2. Reddit retrieval
3. Ranking
4. Theme clustering
5. Theme selection
6. Evidence extraction
7. Memo synthesis
8. Artifact persistence

## Guiding architecture principles

- Keep the pipeline explicit
- Separate retrieval from synthesis
- Make scoring and grouping inspectable
- Prefer deterministic transformations over opaque agent loops
- Use LLMs only where they add clear value
- Save intermediate artifacts for debugging and trust

## Module layout

Expected source layout:

```text
src/reddit_pain_agent/
  __init__.py
  main.py
  config.py
  models.py
  reddit_client.py
  artifact_store.py
  retrieval.py
  ranking.py
  clustering.py
  pain_analysis.py
  memo_writer.py
  prompts.py
  llm.py
  utils.py
```

## Working repo layout

Keep run-specific material outside the package code:

```text
research/
  briefs/
  queries/
templates/
outputs/
  runs/
tests/
```

This separation matters because:

* `research/` is analyst input
* `templates/` is reusable report structure
* `outputs/runs/` is persisted run evidence
* `tests/` validates the deterministic pipeline pieces

## Module responsibilities

### `config.py`

Owns runtime configuration:

* Reddit credentials
* LLM provider configuration
* default output paths
* ranking / clustering knobs
* environment loading

### `models.py`

Owns typed data models:

* post model
* comment model
* candidate post model
* scored post model
* cluster model
* run request
* run result
* memo sections
* artifact payloads

Use Pydantic models for typed flow.

### `reddit_client.py`

Owns direct Reddit access:

* API auth
* subreddit search
* post fetch
* comment fetch
* retries / error handling
* rate limiting

This module should not decide theme relevance.
It only retrieves data.

### `retrieval.py`

Owns query expansion and candidate collection:

* category-to-query expansion
* multi-subreddit retrieval orchestration
* paginated subreddit search retrieval
* deterministic search-plan expansion across sorts and time windows
* candidate normalization
* deterministic retrieval-quality gating before ranking
* filtering deleted / empty / low-signal entries with explicit rejection counts
* explicit comment enrichment for shortlisted submissions
* bounded `MoreComments` expansion to reduce truncated comment evidence

### `artifact_store.py`

Owns persisted run artifacts:

* run directory creation
* manifest persistence
* request log persistence
* raw response persistence
* normalized artifact writes
* top-level run report persistence

### `ranking.py`

Owns candidate scoring:

* textual relevance
* engagement weighting
* comment richness
* discussion-depth gating from saved comment evidence
* modest recency
* explainable score breakdown

It must be possible to inspect why a post ranked highly.

The current vertical slice uses this module to:

* screen posts deterministically before ranking using saved-comment discussion depth
* score candidate posts deterministically
* persist `candidate_screening.json` with explicit rejection reasons
* persist a full ranking artifact
* persist `selected_posts.json` as the shortlist for downstream stages

### `clustering.py`

Owns theme grouping:

* feature generation
* similarity computation
* clustering
* cluster summaries
* strongest theme selection

Initial implementation should be understandable and testable.
TF-IDF + cosine similarity + deterministic clustering is acceptable.

The current vertical slice uses this module to:

* cluster the ranked shortlist deterministically
* select the strongest repeated-pain cluster
* persist `theme_summary.json` for downstream stages
* persist `cluster_evidence_validation.json` so synthesis can reject theme clusters with weak complaint evidence

### `pain_analysis.py`

Owns evidence extraction:

* repeated complaints
* workflow bottlenecks
* unmet needs
* second-order pain
* representative excerpts
* grouping-level pain synthesis inputs

This stage prepares structured evidence for the memo.

The current vertical slice uses this module to:

* load `candidate_posts.json`
* score and select representative saved comments deterministically
* build a deterministic evidence prompt
* call the configured LLM provider for summary generation
* persist summary artifacts back into the run directory

### `llm.py`

Owns model access:

* provider abstraction
* prompt submission
* retries / timeouts
* structured response parsing if used

This should be a thin layer.
Do not bury business logic here.

### `prompts.py`

Owns synthesis prompts and templates:

* memo synthesis prompt
* evidence summarization prompt
* theme normalization prompt if needed

### `memo_writer.py`

Owns markdown generation:

* source post sections
* research takeaways
* top 5 product ideas
* best single bet
* 10 content hooks
* risks / caveats

It can combine deterministic formatting with LLM-generated content.

The current vertical slice uses this module to:

* load `theme_summary.json` and require a strongest cluster
* fail clearly when the strongest cluster has fewer than 5 related posts
* build a grounded memo prompt from the strongest cluster plus saved evidence summary
* persist `final_memo.md`, `final_memo.json`, prompt text, and raw LLM response artifacts

### `main.py`

Owns CLI entry points:

* `search`
* `run`

The current vertical slice exposes explicit stage commands plus an end-to-end `run`
command that orchestrates retrieval, comment enrichment, ranking, clustering,
summary generation, and final memo writing.

It also supports resumable runs by consulting `run_report.json`, validating that
completed-stage artifacts still exist, validating their recorded fingerprints,
and reusing only the completed stages whose recorded parameters still match the
current CLI invocation.

### `utils.py`

Owns small shared helpers only.
Do not let this become a dump.

## Data flow

### Input

User runs CLI with:

* subreddits
* topic or category
* min posts
* output path

### Retrieval output

Candidate posts with:

* title
* subreddit
* url
* score
* num_comments
* created_utc
* selftext
* retrieval provenance
* selected comment text

### Ranking output

Scored candidates with breakdown:

* relevance score
* engagement score
* richness score
* recency score
* combined score
* ranking explanation

### Clustering output

Clustered post groups with:

* cluster id
* member posts
* cluster size
* cluster summary
* confidence / cohesion metadata if available

### Final selection

One selected theme with at least 5 meaningfully related posts.

### Analysis output

Structured evidence:

* repeated pains
* representative quotes or excerpts
* problem taxonomy
* inferred opportunities
* caveats

### Final output

Markdown memo plus saved JSON artifacts.

## Why this architecture

This project should not rely on a freeform autonomous agent loop for core correctness.

The reason:

* retrieval must be trustworthy
* ranking must be inspectable
* clustering must be explainable
* memo generation must be grounded in evidence

So the architecture is mostly pipeline-oriented, with LLM help only at the synthesis layer.

## Failure modes to design for

* not enough relevant posts
* posts too weakly related to form a theme
* high engagement but low relevance
* noisy comments drowning the signal
* subreddit mismatch for requested category
* API failures or rate limits
* model synthesis drifting beyond evidence

## Correct failure behavior

The system should fail clearly when:

* fewer than the required number of related posts are found
* the strongest theme is too weak or incoherent
* Reddit retrieval fails
* output artifacts cannot be saved

Do not silently fabricate a theme.

## Testing strategy

Test the pipeline in layers:

1. ranking unit tests
2. clustering unit tests
3. memo formatting tests
4. CLI smoke test
5. optional integration test with mocked Reddit responses

## Extension path

After the vertical slice works:

* add more subreddits
* add embeddings-based grouping
* add better evidence extraction
* add richer run reports
* add saved run history
* add alternate sources later if needed

Do not expand before the core vertical slice works reliably.
