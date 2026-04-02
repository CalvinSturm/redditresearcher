from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import tempfile
from typing import Any, Protocol
from urllib.parse import quote_plus, urlparse, urlunparse

from .config import DEFAULT_OUTPUT_ROOT, DEFAULT_SORT, DEFAULT_TIME_FILTER, build_search_run_slug
from .models import Comment, ManualImportPost


SEARCH_PAGE_EXTRACTION_SCRIPT = """
() => {
  const cleanText = (value) => (value || "").replace(/\\s+/g, " ").trim();
  const normalizeUrl = (href) => {
    if (!href) {
      return "";
    }
    try {
      const url = new URL(href, window.location.origin);
      url.search = "";
      url.hash = "";
      return url.toString().replace(/\\/$/, "");
    } catch {
      return "";
    }
  };

  const nodes = Array.from(document.querySelectorAll('a[href*="/comments/"]'));
  const results = [];
  for (const anchor of nodes) {
    const url = normalizeUrl(anchor.href || anchor.getAttribute("href") || "");
    if (!url || !/\\/comments\\//i.test(url)) {
      continue;
    }
    const title = cleanText(anchor.textContent);
    if (!title) {
      continue;
    }
    const root = anchor.closest("article, shreddit-post, faceplate-tracker, li, div");
    let subreddit = "";
    if (root) {
      const subredditAnchor = root.querySelector('a[href^="/r/"]');
      if (subredditAnchor) {
        subreddit = cleanText(subredditAnchor.textContent).replace(/^r\\//i, "");
      }
    }
    results.push({ url, title, subreddit });
  }
  return results;
}
"""


THREAD_PAGE_EXTRACTION_SCRIPT = """
() => {
  const cleanText = (value) => (value || "").replace(/\\s+/g, " ").trim();
  const canonical = document.querySelector('link[rel="canonical"]')?.href || window.location.href;
  const normalizedCanonical = (() => {
    try {
      const url = new URL(canonical, window.location.origin);
      url.search = "";
      url.hash = "";
      return url.toString().replace(/\\/$/, "");
    } catch {
      return canonical;
    }
  })();
  const postNode = document.querySelector("shreddit-post, article");
  const title =
    cleanText(postNode?.getAttribute("post-title")) ||
    cleanText(document.querySelector('meta[property="og:title"]')?.content) ||
    cleanText(document.querySelector("h1")?.textContent);
  const permalinkMatch =
    normalizedCanonical.match(/https?:\\/\\/[^/]+(\\/r\\/[^/]+\\/comments\\/[^/]+\\/[^?#]*)/i) ||
    window.location.pathname.match(/(\\/r\\/[^/]+\\/comments\\/[^/]+\\/[^?#]*)/i);
  const permalink = permalinkMatch ? permalinkMatch[1] : window.location.pathname;
  const idMatch =
    normalizedCanonical.match(/\\/comments\\/([^/]+)/i) ||
    window.location.pathname.match(/\\/comments\\/([^/]+)/i);
  const subreddit =
    cleanText(postNode?.getAttribute("subreddit-prefixed-name")).replace(/^r\\//i, "") ||
    cleanText(document.querySelector('a[href^="/r/"]')?.textContent).replace(/^r\\//i, "");
  const selftext =
    cleanText(postNode?.querySelector('[slot="text-body"]')?.innerText) ||
    cleanText(postNode?.querySelector('[data-post-click-location="text-body"]')?.innerText) ||
    cleanText(document.querySelector('[slot="text-body"]')?.innerText) ||
    "";
  const bodyText = cleanText(postNode?.innerText || document.body.innerText || "");
  const commentNodes = Array.from(
    document.querySelectorAll("shreddit-comment, article[id^='t1_'], div[id^='t1_'], [data-testid='comment']")
  );
  const comments = [];
  for (const node of commentNodes) {
    const id =
      cleanText(node.getAttribute("thingid")).replace(/^t1_/i, "") ||
      cleanText(node.getAttribute("comment-id")) ||
      cleanText(node.id).replace(/^t1_/i, "");
    const body =
      cleanText(node.querySelector('[slot="comment"]')?.innerText) ||
      cleanText(node.querySelector('[data-testid="comment"]')?.innerText) ||
      cleanText(node.querySelector("p")?.innerText) ||
      "";
    if (!id || !body) {
      continue;
    }
    const author =
      cleanText(node.getAttribute("author")) ||
      cleanText(node.querySelector('a[href^="/user/"], a[href*="/u/"]')?.textContent) ||
      null;
    const permalinkHref = node.querySelector('a[href*="/comments/"]')?.href || null;
    comments.push({
      id,
      body,
      author,
      permalink: permalinkHref,
      depth: null
    });
  }
  return {
    id: idMatch ? idMatch[1] : "",
    title,
    subreddit,
    url: normalizedCanonical,
    permalink,
    selftext,
    num_comments: comments.length,
    over_18: /nsfw/i.test(bodyText) || /over18/i.test(document.body.innerHTML),
    comments
  };
}
"""


class BrowserPage(Protocol):
    async def goto(self, url: str, *, wait_until: str, timeout: int) -> Any: ...
    async def wait_for_timeout(self, timeout_ms: int) -> Any: ...
    async def evaluate(self, expression: str) -> Any: ...
    async def content(self) -> str: ...
    async def screenshot(self, *, path: str, full_page: bool) -> Any: ...


@dataclass(frozen=True)
class SearchResultPreview:
    title: str
    url: str
    subreddit: str | None
    source_search_url: str


@dataclass(frozen=True)
class PlaywrightCaptureResult:
    output_path: Path
    log_path: Path
    snapshot_dir: Path
    search_url_count: int
    discovered_thread_count: int
    selected_thread_count: int
    captured_post_count: int
    captured_comment_count: int
    html_snapshot_count: int
    screenshot_count: int
    page_error_count: int
    selected_thread_urls: list[str]


@dataclass(frozen=True)
class CaptureSessionPaths:
    output_path: Path
    log_path: Path
    snapshot_dir: Path


async def capture_reddit_threads(
    *,
    subreddits: list[str],
    queries: list[str],
    sort: str = DEFAULT_SORT,
    time_filter: str = DEFAULT_TIME_FILTER,
    thread_urls: list[str] | None = None,
    select_results: list[int] | None = None,
    max_posts: int = 5,
    max_comments: int = 12,
    headless: bool = True,
    page_timeout_seconds: float = 20.0,
    page_wait_ms: int = 1000,
    output_path: Path | None = None,
    skip_search: bool = False,
) -> PlaywrightCaptureResult:
    if max_posts <= 0:
        raise ValueError("max_posts must be greater than 0")
    if max_comments < 0:
        raise ValueError("max_comments must be 0 or greater")
    if page_timeout_seconds <= 0:
        raise ValueError("page_timeout_seconds must be greater than 0")
    if page_wait_ms < 0:
        raise ValueError("page_wait_ms must be 0 or greater")

    session_paths = resolve_capture_session_paths(
        output_path=output_path,
        subreddits=subreddits,
        queries=queries,
    )
    search_urls = [] if skip_search else build_reddit_search_urls(
        subreddits=subreddits,
        queries=queries,
        sort=sort,
        time_filter=time_filter,
    )
    direct_thread_urls = [
        normalize_thread_url(url)
        for url in (thread_urls or [])
        if normalize_thread_url(url)
    ]
    discovered_results: list[SearchResultPreview] = []
    selected_thread_urls: list[str] = []
    log_entries: list[str] = []
    html_snapshot_count = 0
    screenshot_count = 0
    page_error_count = 0
    captured_posts: list[ManualImportPost] = []

    _log_capture_event(
        log_entries,
        level="info",
        event="capture_started",
        output_path=str(session_paths.output_path),
        log_path=str(session_paths.log_path),
        snapshot_dir=str(session_paths.snapshot_dir),
        headless=headless,
        skip_search=skip_search,
        search_url_count=len(search_urls),
    )
    try:
        async with _open_playwright_page(headless=headless) as page:
            for index, search_url in enumerate(search_urls, start=1):
                snapshot_name = f"search-{index:03d}"
                try:
                    previews = await extract_search_results_from_page(
                        page,
                        search_url,
                        page_timeout_seconds=page_timeout_seconds,
                        page_wait_ms=page_wait_ms,
                        snapshot_dir=session_paths.snapshot_dir,
                        snapshot_name=snapshot_name,
                    )
                    html_path, png_path = capture_snapshot_paths(session_paths.snapshot_dir, snapshot_name)
                    html_snapshot_count += int(html_path.exists())
                    screenshot_count += int(png_path.exists())
                    _log_capture_event(
                        log_entries,
                        level="info",
                        event="search_page_extracted",
                        url=search_url,
                        result_count=len(previews),
                        html_snapshot=str(html_path) if html_path.exists() else None,
                        screenshot=str(png_path) if png_path.exists() else None,
                    )
                    discovered_results.extend(previews)
                except Exception as exc:
                    page_error_count += 1
                    html_path, png_path = capture_snapshot_paths(session_paths.snapshot_dir, snapshot_name)
                    html_snapshot_count += int(html_path.exists())
                    screenshot_count += int(png_path.exists())
                    _log_capture_event(
                        log_entries,
                        level="error",
                        event="search_page_failed",
                        url=search_url,
                        error=str(exc),
                        html_snapshot=str(html_path) if html_path.exists() else None,
                        screenshot=str(png_path) if png_path.exists() else None,
                    )

            selected_previews = select_search_results(
                discovered_results,
                select_results=select_results or [],
                max_posts=max_posts,
            )
            selected_thread_urls = _dedupe_preserve_order(
                [*direct_thread_urls, *(item.url for item in selected_previews)]
            )[:max_posts]
            _log_capture_event(
                log_entries,
                level="info",
                event="thread_selection_completed",
                selected_thread_count=len(selected_thread_urls),
                selected_thread_urls=selected_thread_urls,
            )
            for index, thread_url in enumerate(selected_thread_urls, start=1):
                snapshot_name = f"thread-{index:03d}"
                try:
                    captured_post = await extract_thread_post_from_page(
                        page,
                        thread_url,
                        max_comments=max_comments,
                        page_timeout_seconds=page_timeout_seconds,
                        page_wait_ms=page_wait_ms,
                        queries=queries,
                        subreddits=subreddits,
                        sort=sort,
                        time_filter=time_filter,
                        snapshot_dir=session_paths.snapshot_dir,
                        snapshot_name=snapshot_name,
                    )
                    html_path, png_path = capture_snapshot_paths(session_paths.snapshot_dir, snapshot_name)
                    html_snapshot_count += int(html_path.exists())
                    screenshot_count += int(png_path.exists())
                    if captured_post is None:
                        page_error_count += 1
                        _log_capture_event(
                            log_entries,
                            level="warning",
                            event="thread_page_empty",
                            url=thread_url,
                            html_snapshot=str(html_path) if html_path.exists() else None,
                            screenshot=str(png_path) if png_path.exists() else None,
                        )
                        continue
                    _log_capture_event(
                        log_entries,
                        level="info",
                        event="thread_page_extracted",
                        url=thread_url,
                        submission_id=captured_post.id,
                        comment_count=len(captured_post.comments),
                        html_snapshot=str(html_path) if html_path.exists() else None,
                        screenshot=str(png_path) if png_path.exists() else None,
                    )
                    captured_posts.append(captured_post)
                except Exception as exc:
                    page_error_count += 1
                    html_path, png_path = capture_snapshot_paths(session_paths.snapshot_dir, snapshot_name)
                    html_snapshot_count += int(html_path.exists())
                    screenshot_count += int(png_path.exists())
                    _log_capture_event(
                        log_entries,
                        level="error",
                        event="thread_page_failed",
                        url=thread_url,
                        error=str(exc),
                        html_snapshot=str(html_path) if html_path.exists() else None,
                        screenshot=str(png_path) if png_path.exists() else None,
                    )
    finally:
        _log_capture_event(
            log_entries,
            level="info",
            event="capture_finished",
            discovered_thread_count=len(_dedupe_preserve_order([item.url for item in discovered_results])),
            selected_thread_count=len(selected_thread_urls),
            captured_post_count=len(captured_posts),
            captured_comment_count=sum(len(item.comments) for item in captured_posts),
            html_snapshot_count=html_snapshot_count,
            screenshot_count=screenshot_count,
            page_error_count=page_error_count,
        )
        _atomic_write_text(session_paths.log_path, "\n".join(log_entries))

    merged_posts = merge_captured_posts(captured_posts)
    payload = {
        "source": "playwright",
        "captured_at": datetime.now(UTC).isoformat(),
        "search_urls": search_urls,
        "selected_thread_urls": selected_thread_urls,
        "posts": [item.model_dump(mode="json") for item in merged_posts],
    }
    _atomic_write_json(session_paths.output_path, payload)
    return PlaywrightCaptureResult(
        output_path=session_paths.output_path,
        log_path=session_paths.log_path,
        snapshot_dir=session_paths.snapshot_dir,
        search_url_count=len(search_urls),
        discovered_thread_count=len(_dedupe_preserve_order([item.url for item in discovered_results])),
        selected_thread_count=len(selected_thread_urls),
        captured_post_count=len(merged_posts),
        captured_comment_count=sum(len(item.comments) for item in merged_posts),
        html_snapshot_count=html_snapshot_count,
        screenshot_count=screenshot_count,
        page_error_count=page_error_count,
        selected_thread_urls=selected_thread_urls,
    )


def build_reddit_search_urls(
    *,
    subreddits: list[str],
    queries: list[str],
    sort: str,
    time_filter: str,
) -> list[str]:
    if not subreddits:
        raise ValueError("at least one subreddit is required")
    if not queries:
        raise ValueError("at least one query is required")
    urls: list[str] = []
    for subreddit in subreddits:
        for query in queries:
            normalized_query = " ".join(query.split()).strip()
            if not normalized_query:
                continue
            urls.append(
                "https://www.reddit.com"
                f"/r/{subreddit}/search/?q={quote_plus(normalized_query)}"
                f"&sort={sort}&t={time_filter}&type=link"
            )
    return urls


def select_search_results(
    previews: list[SearchResultPreview],
    *,
    select_results: list[int],
    max_posts: int,
) -> list[SearchResultPreview]:
    if max_posts <= 0:
        raise ValueError("max_posts must be greater than 0")
    unique_previews = _dedupe_previews(previews)
    if not unique_previews:
        return []
    if not select_results:
        return unique_previews[:max_posts]
    selected: list[SearchResultPreview] = []
    for index in select_results:
        if index <= 0 or index > len(unique_previews):
            raise ValueError(
                f"select-result index {index} is out of range for {len(unique_previews)} discovered results"
            )
        selected.append(unique_previews[index - 1])
    return _dedupe_previews(selected)[:max_posts]


def merge_captured_posts(posts: list[ManualImportPost]) -> list[ManualImportPost]:
    merged: dict[str, ManualImportPost] = {}
    for post in posts:
        if not post.id:
            continue
        existing = merged.get(post.id)
        if existing is None:
            merged[post.id] = post
            continue
        seen_comment_ids = {item.id for item in existing.comments}
        merged_comments = list(existing.comments)
        for comment in post.comments:
            if comment.id and comment.id not in seen_comment_ids:
                merged_comments.append(comment)
                seen_comment_ids.add(comment.id)
        merged[post.id] = existing.model_copy(
            update={
                "score": post.score if (post.score or 0) > (existing.score or 0) else existing.score,
                "num_comments": post.num_comments if (post.num_comments or 0) > (existing.num_comments or 0) else existing.num_comments,
                "selftext": existing.selftext or post.selftext,
                "comments": merged_comments,
                "source_queries": _dedupe_preserve_order([*existing.source_queries, *post.source_queries]),
                "source_subreddits": _dedupe_preserve_order([*existing.source_subreddits, *post.source_subreddits]),
                "source_sorts": _dedupe_preserve_order([*existing.source_sorts, *post.source_sorts]),
                "source_time_filters": _dedupe_preserve_order([*existing.source_time_filters, *post.source_time_filters]),
                "retrieval_requests": _dedupe_preserve_order([*existing.retrieval_requests, *post.retrieval_requests]),
            }
        )
    return list(merged.values())


def resolve_capture_session_paths(
    *,
    output_path: Path | None,
    subreddits: list[str],
    queries: list[str],
) -> CaptureSessionPaths:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path = output_path
    else:
        capture_slug = build_search_run_slug(subreddits, queries).replace("search-", "capture-", 1)
        captures_dir = DEFAULT_OUTPUT_ROOT.parent / "captures"
        captures_dir.mkdir(parents=True, exist_ok=True)
        resolved_output_path = captures_dir / f"{capture_slug}.json"
    log_path = resolved_output_path.with_suffix(".log")
    snapshot_dir = resolved_output_path.with_suffix("")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return CaptureSessionPaths(
        output_path=resolved_output_path,
        log_path=log_path,
        snapshot_dir=snapshot_dir,
    )


def resolve_capture_output_path(
    *,
    output_path: Path | None,
    subreddits: list[str],
    queries: list[str],
) -> Path:
    return resolve_capture_session_paths(
        output_path=output_path,
        subreddits=subreddits,
        queries=queries,
    ).output_path


def normalize_thread_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme:
        parsed = urlparse(f"https://www.reddit.com{url if url.startswith('/') else '/' + url}")
    cleaned = parsed._replace(query="", fragment="")
    return urlunparse(cleaned).rstrip("/")


async def extract_search_results_from_page(
    page: BrowserPage,
    search_url: str,
    *,
    page_timeout_seconds: float,
    page_wait_ms: int,
    snapshot_dir: Path | None = None,
    snapshot_name: str | None = None,
) -> list[SearchResultPreview]:
    await page.goto(search_url, wait_until="domcontentloaded", timeout=int(page_timeout_seconds * 1000))
    if page_wait_ms:
        await page.wait_for_timeout(page_wait_ms)
    if snapshot_dir is not None and snapshot_name is not None:
        await save_page_snapshot(page, snapshot_dir=snapshot_dir, snapshot_name=snapshot_name)
    payload = await page.evaluate(SEARCH_PAGE_EXTRACTION_SCRIPT)
    results: list[SearchResultPreview] = []
    if not isinstance(payload, list):
        return results
    for item in payload:
        if not isinstance(item, dict):
            continue
        url = normalize_thread_url(str(item.get("url") or ""))
        title = str(item.get("title") or "").strip()
        subreddit = str(item.get("subreddit") or "").strip() or None
        if not url or not title:
            continue
        results.append(
            SearchResultPreview(
                title=title,
                url=url,
                subreddit=subreddit,
                source_search_url=search_url,
            )
        )
    return _dedupe_previews(results)


async def extract_thread_post_from_page(
    page: BrowserPage,
    thread_url: str,
    *,
    max_comments: int,
    page_timeout_seconds: float,
    page_wait_ms: int,
    queries: list[str],
    subreddits: list[str],
    sort: str,
    time_filter: str,
    snapshot_dir: Path | None = None,
    snapshot_name: str | None = None,
) -> ManualImportPost | None:
    await page.goto(thread_url, wait_until="domcontentloaded", timeout=int(page_timeout_seconds * 1000))
    if page_wait_ms:
        await page.wait_for_timeout(page_wait_ms)
    if snapshot_dir is not None and snapshot_name is not None:
        await save_page_snapshot(page, snapshot_dir=snapshot_dir, snapshot_name=snapshot_name)
    payload = await page.evaluate(THREAD_PAGE_EXTRACTION_SCRIPT)
    if not isinstance(payload, dict):
        return None
    post_id = str(payload.get("id") or "").strip()
    title = str(payload.get("title") or "").strip()
    url = normalize_thread_url(str(payload.get("url") or thread_url))
    if not post_id or not title or not url:
        return None

    raw_comments = payload.get("comments") or []
    comments: list[Comment] = []
    if isinstance(raw_comments, list):
        for raw_comment in raw_comments[:max_comments]:
            if not isinstance(raw_comment, dict):
                continue
            body = str(raw_comment.get("body") or "").strip()
            comment_id = str(raw_comment.get("id") or "").strip()
            if not comment_id or not body:
                continue
            comments.append(
                Comment(
                    id=comment_id,
                    body=body,
                    author=str(raw_comment.get("author")) if raw_comment.get("author") else None,
                    score=_coerce_int(raw_comment.get("score")),
                    created_utc=_coerce_float(raw_comment.get("created_utc")),
                    permalink=str(raw_comment.get("permalink")) if raw_comment.get("permalink") else None,
                    parent_id=str(raw_comment.get("parent_id")) if raw_comment.get("parent_id") else None,
                    link_id=str(raw_comment.get("link_id")) if raw_comment.get("link_id") else None,
                    depth=_coerce_int(raw_comment.get("depth")),
                )
            )

    return ManualImportPost(
        id=post_id,
        title=title,
        subreddit=str(payload.get("subreddit") or "").strip() or (subreddits[0] if subreddits else ""),
        url=url,
        permalink=str(payload.get("permalink") or "") or None,
        score=_coerce_int(payload.get("score")),
        num_comments=_coerce_int(payload.get("num_comments")) or len(comments),
        created_utc=_coerce_float(payload.get("created_utc")),
        selftext=str(payload.get("selftext") or "").strip(),
        author=str(payload.get("author")) if payload.get("author") else None,
        over_18=bool(payload.get("over_18")) if payload.get("over_18") is not None else None,
        source_queries=list(queries),
        source_subreddits=_dedupe_preserve_order([*(subreddits or []), str(payload.get("subreddit") or "").strip()]),
        source_sorts=[sort] if sort else [],
        source_time_filters=[time_filter] if time_filter else [],
        retrieval_requests=[f"playwright:{thread_url}"],
        comments=comments,
    )


class _PlaywrightPageContext:
    def __init__(self, *, headless: bool) -> None:
        self._headless = headless
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def __aenter__(self) -> BrowserPage:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required for capture. Install dependencies and run "
                "`python -m playwright install chromium`."
            ) from exc

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()
        return self._page

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._page is not None:
            await self._page.close()
        if self._context is not None:
            await self._context.close()
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()


def _open_playwright_page(*, headless: bool) -> _PlaywrightPageContext:
    return _PlaywrightPageContext(headless=headless)


def capture_snapshot_paths(snapshot_dir: Path, snapshot_name: str) -> tuple[Path, Path]:
    safe_name = _safe_name(snapshot_name)
    return snapshot_dir / f"{safe_name}.html", snapshot_dir / f"{safe_name}.png"


async def save_page_snapshot(
    page: BrowserPage,
    *,
    snapshot_dir: Path,
    snapshot_name: str,
) -> tuple[Path, Path]:
    html_path, screenshot_path = capture_snapshot_paths(snapshot_dir, snapshot_name)
    html_content = await page.content()
    _atomic_write_text(html_path, html_content)
    try:
        await page.screenshot(path=str(screenshot_path), full_page=True)
    except Exception:
        if screenshot_path.exists():
            screenshot_path.unlink(missing_ok=True)
    return html_path, screenshot_path


def _log_capture_event(
    entries: list[str],
    *,
    level: str,
    event: str,
    **fields: Any,
) -> None:
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level,
        "event": event,
        **fields,
    }
    entries.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _dedupe_previews(previews: list[SearchResultPreview]) -> list[SearchResultPreview]:
    seen: set[str] = set()
    deduped: list[SearchResultPreview] = []
    for preview in previews:
        if preview.url in seen:
            continue
        seen.add(preview.url)
        deduped.append(preview)
    return deduped


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        handle.write(content)
        if not content.endswith("\n"):
            handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _safe_name(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-") or "value"
