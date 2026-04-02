# Research Workspace

This folder keeps run inputs separate from implementation code.

## Layout

- `briefs/`: one markdown brief per research run
- `queries/`: optional saved query seeds or subreddit-specific search plans

## Recommended flow

1. Create a run with `python -m reddit_pain_agent.main init-run "<run name>"`
2. Fill in the generated brief in `research/briefs/`
3. Save retrieval and clustering artifacts under `outputs/runs/<run-slug>/`
4. Keep the final memo grounded in the selected posts and comments

## Artifact contract

Each completed run should aim to produce:

- `candidate_posts.json`
- `selected_posts.json`
- `theme_summary.json`
- `final_memo.md`
