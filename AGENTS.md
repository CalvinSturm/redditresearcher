# AGENTS.md

This file tells coding agents how to work in this repository.

## Mission

Build and maintain a Reddit pain-research agent that finds repeated customer pain points from Reddit discussions and produces founder-grade product and content memos.

## Start here

When entering this repo with no context:

1. Read `README.md`
2. Read `ARCHITECTURE.md`
3. Read `SKILLS.md`
4. Read `SOUL.md`
5. Inspect the current source tree
6. Identify the smallest vertical slice that improves the project safely

Do not assume hidden context.
Work from the repo.

## Working style

You are expected to behave like a pragmatic senior engineer:
- keep changes small
- keep logic inspectable
- favor vertical slices over sprawling refactors
- prefer deterministic code paths for retrieval/ranking/clustering
- use LLMs narrowly and intentionally
- preserve clarity over cleverness

## What this repo is

This repo is:
- a research pipeline
- a CLI tool
- a synthesis system grounded in retrieved evidence

This repo is not:
- a generic autonomous agent playground
- a social media spam engine
- a vague “AI insights” demo
- an excuse to hide logic inside prompts

## Core invariants

Do not violate these unless the repo explicitly evolves and the docs are updated:

1. Retrieval is separate from synthesis
2. Reddit data must not be fabricated
3. Theme selection must be evidence-based
4. At least 5 related posts are required for a valid themed run
5. Ranking and clustering should remain inspectable
6. Intermediate artifacts should be saved
7. CLI should remain usable
8. Tests should cover the critical path

## Preferred execution order for any task

1. Understand the task
2. Read nearby code
3. Identify affected modules
4. Make the smallest coherent change
5. Add or update focused tests
6. Run the narrowest useful validation
7. Summarize exactly what changed

## Change boundaries

Prefer narrow edits over broad rewrites.

Good:
- implement one missing pipeline stage
- improve score normalization
- fix broken cluster selection
- add one well-scoped CLI feature
- improve memo formatting
- add a focused test

Bad:
- replacing the whole architecture without evidence
- moving all logic into an LLM prompt
- adding frameworks that do not materially help
- speculative abstractions for future scale
- changing many files when 2 would do

## If the task is ambiguous

Do not stall on ambiguity when a reasonable local decision is possible.
Make the best grounded decision based on:
- current docs
- current code
- current acceptance criteria

If tradeoffs matter, document them briefly in your final summary.

## Implementation priorities

Priority order:
1. correctness
2. inspectability
3. testability
4. developer usability
5. extensibility
6. sophistication

## LLM usage policy

Use LLMs for:
- memo synthesis
- optional semantic normalization
- tightly scoped textual analysis where deterministic logic is insufficient

Do not use LLMs to replace:
- retrieval
- candidate ranking
- engagement scoring
- cluster membership logic
- artifact persistence

## Testing expectations

For meaningful logic changes:
- update or add focused unit tests
- run the narrowest relevant test set
- avoid claiming tests passed if they were not run

If a dependency or environment issue prevents execution, say so explicitly.

## File discipline

Keep module responsibilities clean:
- no business logic in `utils.py`
- no retrieval logic in `memo_writer.py`
- no ranking logic hidden in prompts
- no config constants scattered everywhere

## Logging and artifacts

Preserve inspectability:
- save intermediate artifacts
- keep score breakdowns understandable
- do not hide key decisions in opaque strings

## Final output format for implementation tasks

When you finish a task, report:
1. what changed
2. exact files changed
3. validation run
4. known limitations or follow-ups

## If you need a guiding question

Ask:
“What is the smallest change that makes the core pipeline more correct or more usable?”