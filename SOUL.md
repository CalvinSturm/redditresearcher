# SOUL.md

This file defines what the agent should not do, plus the repo's practical constraints and red lines.

You can think of this as:
- “should not”
- “must avoid”
- “do not fool yourself”

## Hard constraints

### 1. Do not fabricate data
Never invent:
- Reddit posts
- scores
- comment counts
- user comments
- URLs
- selected themes
- source evidence

If the evidence is weak, say it is weak.

### 2. Do not fake a valid theme
If you cannot find at least 5 meaningfully related posts with sufficient engagement, the run should fail clearly.

Do not pad the selection with marginally related posts just to satisfy the requirement.

### 3. Do not replace core logic with prompt magic
This repo should not become:
- “retrieve vaguely, ask LLM to guess the theme”
- “rank with vibes”
- “let the model pick anything”

Retrieval, ranking, and clustering need real code paths.

### 4. Do not overengineer early
Avoid:
- plugin systems before the core works
- orchestration frameworks for no reason
- distributed architecture fantasies
- premature microservices
- heavy databases before run artifacts justify them

### 5. Do not silently swallow uncertainty
If:
- clustering is weak
- subreddit coverage is poor
- comments are missing
- API retrieval failed
- ranking is noisy

surface that honestly.

## Practical repo constraints

### Keep the vertical slice small
The first useful version only needs:
- CLI
- Reddit retrieval
- ranking
- clustering
- memo output
- saved artifacts
- tests

Everything else is secondary.

### Keep modules honest
Do not let:
- `utils.py` become the real codebase
- `prompts.py` become the ranking engine
- `memo_writer.py` do retrieval
- `reddit_client.py` decide strategy

### Keep the system inspectable
Save artifacts.
Expose why posts ranked highly.
Explain theme grouping.
Make the output debuggable.

## Behavior pitfalls to avoid

### 1. Chasing breadth over quality
More subreddits, more features, and more categories are not wins if the chosen theme quality is weak.

### 2. Confusing engagement with relevance
A high-comment thread is not automatically the right thread.

### 3. Confusing relevance with opportunity
Repeated discussion can still be a bad product opportunity.
The memo should be thoughtful, not automatic.

### 4. Turning the product into spam tooling
This repo is for research.
Not for automated posting, growth hacking spam, or manipulative engagement tooling.

### 5. Writing generic memos
Bad memo language:
- “users want better UX”
- “there is demand for AI solutions”
- “businesses need automation”
- “founders struggle with challenges”

Good memo language is sharper and grounded.

## Decision rules when unsure

If unsure, prefer the option that is:
1. more truthful
2. more inspectable
3. more local in scope
4. easier to test
5. easier to explain

## Smells that mean you are drifting

If you find yourself doing any of these, stop and reassess:
- adding major abstractions before the CLI works
- skipping tests because “the design is obvious”
- hiding ranking logic in prose
- using an LLM because deterministic code feels annoying
- broad refactors not tied to acceptance criteria
- claiming success without a real run path

## Final reminder

This repo wins by being:
- grounded
- inspectable
- sharp
- useful

It loses when it becomes:
- vague
- magical
- overbuilt
- untestable