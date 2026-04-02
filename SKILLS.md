# SKILLS.md

This file defines the capabilities the agent should demonstrate while working in this repo.

## Primary engineering skills

### 1. Pipeline thinking
The agent should reason in explicit stages:
- retrieval
- ranking
- clustering
- evidence extraction
- synthesis
- persistence

It should avoid collapsing everything into one opaque step.

### 2. Typed Python implementation
The agent should be comfortable with:
- Python 3.11+
- Pydantic models
- typed function boundaries
- clear module responsibilities
- straightforward CLI design

### 3. Information retrieval judgment
The agent should understand:
- query expansion
- relevance vs engagement tradeoffs
- why comments matter
- why post titles alone are insufficient
- why not all high-score posts are thematically useful

### 4. Evidence-grounded synthesis
The agent should be able to turn structured evidence into:
- research takeaways
- product ideas
- positioning angles
- content hooks
- caveats

without drifting into generic filler.

### 5. Small-scope engineering discipline
The agent should:
- prefer a working vertical slice
- avoid speculative architecture
- add tests with logic changes
- keep code readable

## Domain skills

### Reddit-native research sense
The agent should understand:
- Reddit has noisy but useful pain-language
- high engagement is helpful but not enough by itself
- comments often contain the strongest signal
- community norms matter
- theme validity matters more than raw post count

### Founder / product strategy sense
The agent should be able to translate repeated discussion patterns into:
- concrete pain statements
- likely ICPs
- product wedges
- realistic MVP directions
- content angles

### Business pain taxonomy
The agent should recognize common classes of pain:
- tooling pain
- workflow pain
- ergonomics pain
- trust pain
- GTM pain
- hiring / ops pain
- speed-to-lead or distribution pain

## Output skills

### Memo quality
Outputs should be:
- concise
- sharp
- evidence-aware
- useful for decision-making
- free of hype

### Debuggability
The agent should preserve:
- score breakdowns
- selected theme rationale
- saved artifacts
- deterministic intermediate outputs where feasible

## Minimum competence bar

A competent agent in this repo should be able to:
1. trace the full run path
2. implement a missing stage cleanly
3. diagnose why the wrong posts were selected
4. improve theme quality without overhauling the whole system
5. write or update tests for its changes
6. explain the tradeoffs in plain English

## Anti-skills to avoid

The following are not desirable behaviors here:
- “just let the model decide everything”
- giant abstractions before the pipeline works
- rewriting many modules for style alone
- burying key logic in prompts
- pretending uncertainty does not exist
- picking weakly related posts to hit quotas

## If the repo is mostly empty

Start with these skills in order:
1. define models
2. define config
3. implement retrieval client
4. implement ranking
5. implement clustering
6. implement memo writer
7. wire CLI
8. add tests

That sequence is preferred because it yields a working vertical slice quickly.