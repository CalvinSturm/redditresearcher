from __future__ import annotations

from .models import CandidatePost, Comment, ThemeCluster


def build_candidate_evidence_prompt(
    posts: list[CandidatePost],
    comments_by_submission: dict[str, list[Comment]] | None = None,
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
            "You are analyzing Reddit candidate posts for repeated pain signals.",
            "Work only from the evidence provided below.",
            "Do not invent complaints, user types, workarounds, or certainty.",
            "If the evidence is weak or mixed, say so clearly.",
            "Return markdown with these sections and exact headings:",
            "## Repeated Complaints",
            "## Who Feels The Pain",
            "## Current Workarounds",
            "## Evidence Strength",
            "## Open Questions",
        ]
    )
    return f"{instructions}\n\n# Candidate Post Evidence\n\n" + "\n\n".join(evidence_blocks)


def build_final_memo_prompt(
    theme_cluster: ThemeCluster,
    posts: list[CandidatePost],
    evidence_summary_text: str,
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
            "You are writing a founder-grade Reddit research memo.",
            "Work only from the evidence summary and source posts below.",
            "Do not invent post counts, user types, pain points, quotes, or workarounds.",
            "If the evidence is mixed or weak, say so explicitly.",
            "Use markdown with these exact headings:",
            "# Executive Summary",
            "## Research Takeaways",
            "## Top 5 Product Ideas",
            "## Best Single Bet",
            "## 10 Content Hooks",
            "## Risks / Caveats",
            "Under 'Top 5 Product Ideas', provide exactly 5 numbered items.",
            "Under '10 Content Hooks', provide exactly 10 numbered items.",
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
            cluster_block,
            "# Evidence Summary",
            evidence_summary_text.strip(),
            "# Source Posts",
            "\n\n".join(post_blocks),
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
