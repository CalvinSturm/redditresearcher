from __future__ import annotations

from .models import CandidatePost, Comment, RankedCandidatePost, RunManifest, ThemeCluster


def build_candidate_evidence_prompt(
    posts: list[CandidatePost],
    comments_by_submission: dict[str, list[Comment]] | None = None,
    research_context: RunManifest | None = None,
    max_posts: int = 10,
    max_selftext_chars: int = 600,
    max_comments_per_post: int = 5,
    max_comment_chars: int = 280,
) -> str:
    if not posts:
        raise ValueError("at least one candidate post is required")

    selected_posts = posts[:max_posts]
    evidence_blocks = []
    for index, post in enumerate(selected_posts, start=1):
        selftext = (post.selftext or "").strip()
        trimmed_selftext = selftext[:max_selftext_chars].strip()
        if selftext and len(selftext) > max_selftext_chars:
            trimmed_selftext = f"{trimmed_selftext}..."
        evidence_blocks.append(
            "\n".join(
                [
                    f"Post {index}",
                    f"Subreddit: {post.subreddit}",
                    f"Title: {post.title}",
                    f"URL: {post.url}",
                    f"Score: {post.score if post.score is not None else 'unknown'}",
                    f"Comments: {post.num_comments if post.num_comments is not None else 'unknown'}",
                    f"Queries: {', '.join(post.source_queries) if post.source_queries else 'unknown'}",
                    f"Body: {trimmed_selftext or '[no selftext]'}",
                    _format_comment_block(
                        comments_by_submission.get(post.id, []) if comments_by_submission else [],
                        max_comments=max_comments_per_post,
                        max_comment_chars=max_comment_chars,
                    ),
                ]
            )
        )

    instructions = "\n".join(
        [
            f"Research Reddit for repeated pain points around this topic: {_topic_label(research_context)}.",
            "Your job is to identify real, repeated frustrations, bottlenecks, failed workflows, workarounds, and unmet needs from Reddit discussions.",
            "Operate like a researcher, not a content writer. Prioritize evidence density, repetition, and specificity.",
            "Work only from the evidence provided below.",
            "Do not fabricate Reddit data, quotes, themes, users, or conclusions.",
            "Prefer repeated pains over one-off opinions.",
            "Do not force a conclusion if evidence is weak or fragmented.",
            "Separate true pain points from general preferences, ideology, or low-stakes complaints.",
            "Do not overcount duplicates, reposts, or multiple comments from the same discussion as separate evidence.",
            "If the evidence is mixed, say so explicitly.",
            "Return markdown with these exact headings:",
            "## Topic Overview",
            "## Candidate Pain Themes",
            "## Evidence Strength",
            "## Open Questions",
        ]
    )
    return "\n\n".join(
        [
            instructions,
            _format_research_context(research_context),
            "# Candidate Post Evidence",
            "\n\n".join(evidence_blocks),
        ]
    )


def build_final_memo_prompt(
    theme_cluster: ThemeCluster,
    posts: list[CandidatePost],
    evidence_summary_text: str,
    research_context: RunManifest | None = None,
    *,
    max_posts: int = 8,
    max_selftext_chars: int = 450,
) -> str:
    if not posts:
        raise ValueError("at least one post is required to build a final memo prompt")
    if not evidence_summary_text.strip():
        raise ValueError("evidence_summary_text is required")

    selected_posts = posts[:max_posts]
    post_blocks = []
    for index, post in enumerate(selected_posts, start=1):
        selftext = (post.selftext or "").strip()
        trimmed_selftext = selftext[:max_selftext_chars].strip()
        if selftext and len(selftext) > max_selftext_chars:
            trimmed_selftext = f"{trimmed_selftext}..."
        post_blocks.append(
            "\n".join(
                [
                    f"Post {index}",
                    f"Subreddit: {post.subreddit}",
                    f"Title: {post.title}",
                    f"URL: {post.url}",
                    f"Score: {post.score if post.score is not None else 'unknown'}",
                    f"Comments: {post.num_comments if post.num_comments is not None else 'unknown'}",
                    f"Body: {trimmed_selftext or '[no selftext]'}",
                ]
            )
        )

    instructions = "\n".join(
        [
            f"Research Reddit for repeated pain points around this topic: {_topic_label(research_context)}.",
            "Your job is to identify real, repeated frustrations, bottlenecks, failed workflows, workarounds, and unmet needs from Reddit discussions, then turn them into a grounded research memo.",
            "Operate like a researcher, not a content writer. Prioritize evidence density, repetition, and specificity.",
            "Work only from the evidence summary and source posts below.",
            "Do not fabricate Reddit data, quotes, themes, users, or conclusions.",
            "Prefer repeated pains over one-off opinions.",
            "Do not force a conclusion if evidence is weak or fragmented.",
            "Separate true pain points from general preferences, ideology, or low-stakes complaints.",
            "A valid pain theme must be supported by at least 5 distinct, meaningfully related posts or threads.",
            "Do not overcount duplicates, reposts, or multiple comments from the same discussion as separate evidence.",
            "If the evidence is mixed, say so explicitly.",
            "For each theme, capture: theme name, why it matters, who feels it most, supporting evidence count, representative evidence, common workarounds or failed solutions, why current solutions seem insufficient, and confidence level.",
            "Use markdown with these exact headings:",
            "## Topic Overview",
            "## Top Repeated Pain Themes",
            "## Product Opportunities",
            "## Best Single Opportunity",
            "## Risks and Caveats",
            "Under '## Product Opportunities', provide flat bullets. Each bullet must state which theme it maps to and why it might work.",
        ]
    )

    cluster_block = "\n".join(
        [
            "# Selected Theme",
            f"- cluster_id: {theme_cluster.cluster_id}",
            f"- label: {theme_cluster.label}",
            f"- post_count: {theme_cluster.size}",
            f"- average_post_score: {theme_cluster.average_post_score}",
            f"- total_comment_count: {theme_cluster.total_comment_count}",
            f"- top_terms: {', '.join(theme_cluster.top_terms) if theme_cluster.top_terms else 'unknown'}",
        ]
    )

    return "\n\n".join(
        [
            instructions,
            _format_research_context(research_context),
            cluster_block,
            "# Evidence Summary",
            evidence_summary_text.strip(),
            "# Source Posts",
            "\n\n".join(post_blocks),
        ]
    )


def _topic_label(research_context: RunManifest | None) -> str:
    if research_context and research_context.topic:
        return research_context.topic
    if research_context and research_context.queries:
        return research_context.queries[0]
    return "the requested topic"


def _format_research_context(research_context: RunManifest | None) -> str:
    if research_context is None:
        return "# Research Context\n- topic: unknown"

    subreddits = ", ".join(research_context.subreddits) if research_context.subreddits else "unknown"
    preferred = ", ".join(research_context.subreddits) if research_context.subreddits else "unknown"
    avoided = ", ".join(research_context.denied_subreddits) if research_context.denied_subreddits else "none"
    return "\n".join(
        [
            "# Research Context",
            f"- topic: {research_context.topic or _topic_label(research_context)}",
            f"- target_audience: {research_context.target_audience or 'unspecified'}",
            f"- category: {research_context.category or 'unspecified'}",
            f"- time_horizon: {research_context.time_horizon or research_context.time_filter or 'unspecified'}",
            f"- subreddits_reviewed: {subreddits}",
            f"- preferred_subreddits: {preferred}",
            f"- avoided_subreddits: {avoided}",
        ]
    )


def build_reply_drafts_prompt(
    posts: list[RankedCandidatePost],
    *,
    voice: str,
    max_posts: int,
    max_selftext_chars: int = 350,
) -> str:
    if not posts:
        raise ValueError("at least one post is required to build reply drafts")
    if not voice.strip():
        raise ValueError("voice is required")

    selected_posts = posts[:max_posts]
    post_blocks = []
    for index, post in enumerate(selected_posts, start=1):
        candidate = post.candidate
        selftext = candidate.selftext.strip()
        trimmed_selftext = selftext[:max_selftext_chars].strip()
        if selftext and len(selftext) > max_selftext_chars:
            trimmed_selftext = f"{trimmed_selftext}..."
        post_blocks.append(
            "\n".join(
                [
                    f"Post {index}",
                    f"post_id: {candidate.id}",
                    f"rank: {post.rank}",
                    f"subreddit: {candidate.subreddit}",
                    f"title: {candidate.title}",
                    f"url: {candidate.url}",
                    f"body: {trimmed_selftext or '[no selftext]'}",
                ]
            )
        )

    instructions = "\n".join(
        [
            "You are drafting Reddit-friendly reply suggestions for manual review only.",
            "Do not claim to be the original poster.",
            "Do not use marketing language, sales CTAs, outreach hooks, or ask to DM.",
            "Keep each reply Reddit-friendly, natural, respectful, and worth posting in-thread.",
            "Each reply should respond to the actual post, add something to the conversation, and include a clear take or vibe from the user instead of generic agreement.",
            "Write each reply in the user's voice described below.",
            "Each reply must be plain text only and must be 1 to 3 short paragraphs.",
            "Do not use bullet points, numbered lists, hashtags, headings, markdown emphasis, or sign-offs inside the reply.",
            "Return markdown using exactly this structure for every post:",
            "## Post <n>",
            "post_id: <post_id>",
            "reply: <1-3 paragraph reply>",
            "",
            f"User voice: {voice.strip()}",
        ]
    )
    return "\n\n".join([instructions, "# Posts To Reply To", "\n\n".join(post_blocks)])


def build_reply_improvement_prompt(
    posts: list[RankedCandidatePost],
    initial_output: str,
    *,
    voice: str,
    max_posts: int,
    evaluation_feedback: dict[str, str] | None = None,
    round_number: int = 1,
) -> str:
    if not posts:
        raise ValueError("at least one post is required to improve reply drafts")
    if not initial_output.strip():
        raise ValueError("initial_output is required")
    if not voice.strip():
        raise ValueError("voice is required")

    selected_posts = posts[:max_posts]
    post_blocks = []
    for index, post in enumerate(selected_posts, start=1):
        candidate = post.candidate
        feedback = (evaluation_feedback or {}).get(candidate.id, "").strip()
        post_blocks.append(
            "\n".join(
                [
                    f"Post {index}",
                    f"post_id: {candidate.id}",
                    f"subreddit: {candidate.subreddit}",
                    f"title: {candidate.title}",
                    f"url: {candidate.url}",
                    f"body: {candidate.selftext.strip() or '[no selftext]'}",
                    f"feedback: {feedback or '[no targeted feedback]'}",
                ]
            )
        )

    instructions = "\n".join(
        [
            f"Revise the reply drafts below so they feel native to Reddit. This is revision round {round_number}.",
            "For each draft, check these criteria before rewriting:",
            "1. It is directly relevant to the specific post.",
            "2. It adds something to the conversation beyond agreement or summary.",
            "3. It reflects the user's voice, take, and vibe.",
            "4. It stays Reddit-friendly and does not sound promotional, scripted, or outreach-heavy.",
            "5. The final reply is plain text only and is 1 to 3 short paragraphs.",
            "6. There are no bullet points, numbered lists, headings, hashtags, emojis, or sign-offs inside the reply.",
            "Return markdown using exactly this structure for every post:",
            "## Post <n>",
            "post_id: <post_id>",
            "reply: <1-3 paragraph revised reply>",
            "",
            f"User voice: {voice.strip()}",
        ]
    )
    return "\n\n".join(
        [
            instructions,
            "# Source Posts",
            "\n\n".join(post_blocks),
            "# Initial Drafts",
            initial_output.strip(),
        ]
    )


def build_reply_evaluation_prompt(
    posts: list[RankedCandidatePost],
    draft_output: str,
    *,
    voice: str,
    max_posts: int,
) -> str:
    if not posts:
        raise ValueError("at least one post is required to evaluate reply drafts")
    if not draft_output.strip():
        raise ValueError("draft_output is required")
    if not voice.strip():
        raise ValueError("voice is required")

    selected_posts = posts[:max_posts]
    post_blocks = []
    for index, post in enumerate(selected_posts, start=1):
        candidate = post.candidate
        post_blocks.append(
            "\n".join(
                [
                    f"Post {index}",
                    f"post_id: {candidate.id}",
                    f"subreddit: {candidate.subreddit}",
                    f"title: {candidate.title}",
                    f"url: {candidate.url}",
                    f"body: {candidate.selftext.strip() or '[no selftext]'}",
                ]
            )
        )

    instructions = "\n".join(
        [
            "Evaluate the reply drafts below against a strict Reddit-friendly rubric.",
            "Score each dimension from 1 to 5, where 5 is strongest.",
            "Dimensions:",
            "1. relevance",
            "2. conversation_value",
            "3. voice_match",
            "4. reddit_friendliness",
            "Be tough on generic agreement, weak relevance, promotional tone, and replies that do not add a real take.",
            "User voice must match this description:",
            voice.strip(),
            'Return valid JSON only, with this exact shape:',
            '{"evaluations":[{"post_id":"...","relevance_score":4,"conversation_value_score":4,"voice_match_score":4,"reddit_friendliness_score":5,"feedback":"..."}]}',
        ]
    )

    return "\n\n".join(
        [
            instructions,
            "# Source Posts",
            "\n\n".join(post_blocks),
            "# Draft Replies",
            draft_output.strip(),
        ]
    )


def _format_comment_block(
    comments: list[Comment],
    *,
    max_comments: int,
    max_comment_chars: int,
) -> str:
    if not comments:
        return "Representative Comments: [no saved comments]"

    lines = ["Representative Comments:"]
    for comment in comments[:max_comments]:
        body = comment.body.strip()
        trimmed = body[:max_comment_chars].strip()
        if len(body) > max_comment_chars:
            trimmed = f"{trimmed}..."
        lines.append(
            f"- score={comment.score if comment.score is not None else 'unknown'} depth={comment.depth if comment.depth is not None else 'unknown'} body={trimmed}"
        )
    return "\n".join(lines)
