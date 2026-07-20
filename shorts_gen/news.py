"""AI News Collector — shorts_gen/news.py

Automatically fetches the latest AI news from multiple free, public sources
using only Python stdlib (urllib, xml.etree.ElementTree, json, html.parser).
No extra pip packages required.

Sources
-------
RSS feeds  : OpenAI Blog, Anthropic, Google AI Blog, HuggingFace Blog,
             NVIDIA AI Blog, Microsoft AI Blog
JSON APIs  : Hacker News (Algolia free API), Reddit (r/OpenAI, r/LocalLLaMA,
             r/MachineLearning, r/artificial)

Pipeline
--------
fetch_news()          → raw Story list from all sources
deduplicate()         → remove near-identical titles
score_and_classify()  → keyword pre-score → LLM score + classify (or heuristic)
top_stories()         → top N by score, valid categories only
print_news_board()    → formatted terminal output
best_story()          → single highest-scoring valid pick

Story categories
----------------
✅ Valid (generate Shorts for these):
   Breaking News | Product Launch | AI Tool | Research Breakthrough | Business/Funding

❌ Invalid (skip these):
   Opinion | Tutorial | Weekly Roundup | Minor Patch | Other

Public API used by main.py
---------------------------
  fetch_and_score(provider, model, verbose) -> list[Story]
  print_news_board(stories, top_n)
  best_story(stories) -> Story | None
"""
from __future__ import annotations

import hashlib
import html.parser
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {
    "Breaking News",
    "Product Launch",
    "AI Tool",
    "Research Breakthrough",
    "Business/Funding",
}

INVALID_CATEGORIES = {
    "Opinion",
    "Tutorial",
    "Weekly Roundup",
    "Minor Patch",
    "Other",
}

ALL_CATEGORIES = VALID_CATEGORIES | INVALID_CATEGORIES

# RSS sources: (display name, feed URL)
RSS_SOURCES: list[tuple[str, str]] = [
    ("OpenAI Blog",    "https://openai.com/blog/rss.xml"),
    ("Anthropic",      "https://www.anthropic.com/rss.xml"),
    ("Google AI Blog", "https://blog.research.google/feeds/posts/default"),
    ("HuggingFace",    "https://huggingface.co/blog/feed.xml"),
    ("NVIDIA AI",      "https://blogs.nvidia.com/blog/category/generative-ai/feed/"),
    ("Microsoft AI",   "https://blogs.microsoft.com/ai/feed/"),
]

# Hacker News via Algolia free search API — no key required
_HN_API = (
    "https://hn.algolia.com/api/v1/search"
    "?query=artificial+intelligence+AI+LLM+machine+learning"
    "&tags=story"
    "&numericFilters=points%3E15"
    "&hitsPerPage=25"
)

# Reddit subreddits to scrape (JSON API, no auth required)
REDDIT_SUBS = ["OpenAI", "LocalLLaMA", "MachineLearning", "artificial"]

# HTTP request settings
_USER_AGENT = (
    "Mozilla/5.0 (compatible; yt-gen-news/1.0; "
    "+https://github.com/yt-gen)"
)
_REDDIT_UA = "yt-gen-news/1.0 by /u/yt_gen_shorts"
_FETCH_TIMEOUT = 15  # seconds per request

# Atom XML namespace
_ATOM_NS = "http://www.w3.org/2005/Atom"


# ---------------------------------------------------------------------------
# Story dataclass
# ---------------------------------------------------------------------------

@dataclass
class Story:
    """A single AI news story collected from any source."""

    title: str
    source: str
    url: str
    published: datetime
    summary: str
    keyword_score: int = 0         # fast keyword-based pre-score (0-100)
    llm_score: Optional[int] = None  # LLM viral-potential score (0-100), or None
    category: str = "Unknown"      # one of ALL_CATEGORIES
    is_valid: bool = True          # False for Opinion/Tutorial/Roundup/Patch

    # Grounding fields — populated by ground_story()
    article_text: str = ""         # raw extracted article text
    article_notes: str = ""        # LLM-generated factual summary + top paragraphs
    article_chars: int = 0         # character count of extracted text
    article_extracted: bool = False  # True only when extraction succeeded
    article_paragraphs: str = ""   # top 5 relevant paragraphs from the article

    @property
    def score(self) -> int:
        """Effective score: LLM score if available, else keyword score."""
        return self.llm_score if self.llm_score is not None else self.keyword_score

    @property
    def age_hours(self) -> float:
        """How many hours ago this story was published."""
        now = datetime.now(timezone.utc)
        pub = self.published
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        delta = now - pub
        return delta.total_seconds() / 3600


# ---------------------------------------------------------------------------
# HTTP + parsing helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, *, timeout: int = _FETCH_TIMEOUT, ua: str = _USER_AGENT) -> bytes:
    """Download URL and return raw bytes. Raises on any network error."""
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _parse_date(raw: str) -> datetime:
    """Parse RFC 2822 (RSS) or ISO 8601 (Atom) date strings → UTC datetime."""
    if not raw:
        return datetime.now(timezone.utc)
    raw = raw.strip()
    # RFC 2822
    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc)
    except Exception:
        pass
    # ISO 8601 variants — try progressively shorter formats
    iso_formats = [
        ("%Y-%m-%dT%H:%M:%S%z", 25),   # 2026-01-02T03:04:05+00:00
        ("%Y-%m-%dT%H:%M:%SZ",  20),   # 2026-01-02T03:04:05Z
        ("%Y-%m-%dT%H:%M:%S",   19),   # 2026-01-02T03:04:05
        ("%Y-%m-%d",            10),   # 2026-01-02
    ]
    for fmt, length in iso_formats:
        try:
            dt = datetime.strptime(raw[:length], fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    return datetime.now(timezone.utc)


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", " ", text or "").strip()


def _truncate(text: str, max_chars: int = 250) -> str:
    """Truncate text to max_chars at a word boundary."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:") + "…"


# ---------------------------------------------------------------------------
# RSS / Atom feed parser
# ---------------------------------------------------------------------------

def _parse_feed(data: bytes, source_name: str, max_items: int) -> list[Story]:
    """Parse RSS 2.0 or Atom feed bytes into a list of Story objects."""
    stories: list[Story] = []
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return stories

    ns_tag = root.tag
    is_atom = "atom" in ns_tag.lower() or ns_tag == f"{{{_ATOM_NS}}}feed"

    if is_atom:
        # --- Atom feed ---
        entries = root.findall(f"{{{_ATOM_NS}}}entry")
        for entry in entries[:max_items]:
            title_el = entry.find(f"{{{_ATOM_NS}}}title")
            title = (title_el.text or "").strip() if title_el is not None else ""

            link_el = entry.find(f"{{{_ATOM_NS}}}link")
            url = ""
            if link_el is not None:
                url = link_el.get("href", "") or (link_el.text or "")

            pub_el = (
                entry.find(f"{{{_ATOM_NS}}}published")
                or entry.find(f"{{{_ATOM_NS}}}updated")
            )
            pub = _parse_date(pub_el.text if pub_el is not None else "")

            sum_el = entry.find(f"{{{_ATOM_NS}}}summary") or entry.find(
                f"{{{_ATOM_NS}}}content"
            )
            summary = _truncate(_strip_html((sum_el.text or "") if sum_el is not None else ""))

            if title:
                stories.append(Story(
                    title=title, source=source_name,
                    url=url.strip(), published=pub, summary=summary,
                ))
    else:
        # --- RSS 2.0 ---
        channel = root.find("channel") or root
        items = channel.findall("item")
        for item in items[:max_items]:
            title_el = item.find("title")
            title = (title_el.text or "").strip() if title_el is not None else ""

            link_el = item.find("link")
            url = (link_el.text or "").strip() if link_el is not None else ""

            pub_el = item.find("pubDate") or item.find("published")
            pub = _parse_date(pub_el.text if pub_el is not None else "")

            desc_el = item.find("description")
            summary = _truncate(_strip_html((desc_el.text or "") if desc_el is not None else ""))

            if title:
                stories.append(Story(
                    title=title, source=source_name,
                    url=url, published=pub, summary=summary,
                ))

    return stories


# ---------------------------------------------------------------------------
# Individual source fetchers
# ---------------------------------------------------------------------------

def _fetch_rss(name: str, url: str, max_items: int, *, verbose: bool) -> list[Story]:
    try:
        data = _http_get(url)
        stories = _parse_feed(data, name, max_items)
        if verbose:
            print(f"    {name}: {len(stories)} stories")
        return stories
    except Exception as exc:
        if verbose:
            print(f"    {name}: ✗ ({exc})")
        return []


def _fetch_hackernews(max_items: int, *, verbose: bool) -> list[Story]:
    try:
        data = _http_get(_HN_API)
        body = json.loads(data)
        stories: list[Story] = []
        for hit in body.get("hits", [])[:max_items]:
            title = (hit.get("title") or "").strip()
            if not title:
                continue
            url = hit.get("url") or (
                f"https://news.ycombinator.com/item?id={hit.get('objectID','')}"
            )
            ts = hit.get("created_at_i")
            pub = (
                datetime.fromtimestamp(ts, tz=timezone.utc)
                if ts
                else datetime.now(timezone.utc)
            )
            summary = _truncate(hit.get("story_text") or "")
            stories.append(Story(
                title=title, source="Hacker News",
                url=url, published=pub, summary=summary,
            ))
        if verbose:
            print(f"    Hacker News: {len(stories)} stories")
        return stories
    except Exception as exc:
        if verbose:
            print(f"    Hacker News: ✗ ({exc})")
        return []


def _fetch_reddit(subreddit: str, max_items: int, *, verbose: bool) -> list[Story]:
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={max_items}"
    try:
        data = _http_get(url, ua=_REDDIT_UA)
        body = json.loads(data)
        posts = body.get("data", {}).get("children", [])
        stories: list[Story] = []
        for post in posts[:max_items]:
            d = post.get("data", {})
            if d.get("stickied") or d.get("is_self") is False and not d.get("url"):
                continue
            title = (d.get("title") or "").strip()
            if not title:
                continue
            permalink = f"https://reddit.com{d.get('permalink', '')}"
            pub = datetime.fromtimestamp(
                d.get("created_utc", time.time()), tz=timezone.utc
            )
            summary = _truncate(d.get("selftext", "") or d.get("url_overridden_by_dest", ""))
            stories.append(Story(
                title=title, source=f"r/{subreddit}",
                url=permalink, published=pub, summary=summary,
            ))
        if verbose:
            print(f"    r/{subreddit}: {len(stories)} stories")
        return stories
    except Exception as exc:
        if verbose:
            print(f"    r/{subreddit}: ✗ ({exc})")
        return []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _normalize(title: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    t = title.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _token_overlap(a: str, b: str) -> float:
    """Jaccard-like overlap on word tokens."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def deduplicate(stories: list[Story], threshold: float = 0.65) -> list[Story]:
    """Remove near-duplicate stories (≥ threshold token overlap in normalised title).

    Within each duplicate group the freshest story is kept.
    """
    norms = [_normalize(s.title) for s in stories]
    keep: list[Story] = []
    skip: set[int] = set()

    for i in range(len(stories)):
        if i in skip:
            continue
        group = [i]
        for j in range(i + 1, len(stories)):
            if j not in skip and _token_overlap(norms[i], norms[j]) >= threshold:
                group.append(j)
                skip.add(j)
        # Keep the most recently published story from the group
        best_idx = max(group, key=lambda idx: stories[idx].published)
        keep.append(stories[best_idx])

    return keep


# ---------------------------------------------------------------------------
# Keyword pre-scoring (fast, no LLM)
# ---------------------------------------------------------------------------

_BOOSTS: list[tuple[list[str], int]] = [
    (["openai", "gpt", "chatgpt", "o3", "o4", "o1"], 22),
    (["anthropic", "claude"], 20),
    (["google", "gemini", "deepmind"], 19),
    (["xai", "grok"], 17),
    (["nvidia", "cuda"], 16),
    (["launches", "releases", "introduces", "unveils", "announces", "new model"], 15),
    (["billion", "funding", "acquisition", "acquired", "investment", "raises"], 14),
    (["breakthrough", "state-of-the-art", "sota", "beats", "surpasses", "first ever"], 16),
    (["ai agent", "agentic", "autonomous agent"], 14),
    (["robotics", "robot", "humanoid"], 12),
    (["coding", "code generation", "copilot", "cursor", "devin"], 11),
    (["open source", "open-source", "free tier", "available now"], 8),
    (["consumer", "everyone", "public", "anyone can"], 7),
]

_PENALTIES: list[tuple[list[str], int]] = [
    (["tutorial", "how to", "how-to", "guide", "learn", "course", "beginner"], -20),
    (["opinion", "commentary", "my take", "i think", "perspective", "viewpoint"], -15),
    (["weekly", "roundup", "digest", "newsletter", "recap", "wrap-up"], -18),
    (["minor update", "small fix", "patch", "bug fix", "hotfix"], -14),
    (["job listing", "hiring", "career", "we're hiring"], -12),
    (["paper review", "explained", "deep dive"], -8),
]


def _keyword_score(story: Story) -> int:
    """Fast heuristic score based on keyword presence in title + summary."""
    text = f"{story.title} {story.summary}".lower()
    score = 45  # neutral baseline

    for keywords, pts in _BOOSTS:
        if any(kw in text for kw in keywords):
            score += pts
    for keywords, pts in _PENALTIES:
        if any(kw in text for kw in keywords):
            score += pts  # pts already negative

    # Age penalty
    age = story.age_hours
    if age > 168:    # > 7 days
        score -= 22
    elif age > 72:   # > 3 days
        score -= 12
    elif age > 48:   # > 2 days
        score -= 6

    return max(0, min(100, score))


# ---------------------------------------------------------------------------
# Story category classification (heuristic fallback)
# ---------------------------------------------------------------------------

def _heuristic_category(story: Story) -> str:
    """Rule-based category from title + summary keywords."""
    text = f"{story.title} {story.summary}".lower()

    # Invalids first (higher false-positive risk if checked last)
    if any(kw in text for kw in ["tutorial", "how to", "how-to", "guide", "learn how", "step by step", "beginner"]):
        return "Tutorial"
    if any(kw in text for kw in ["opinion", "commentary", "my take", "i think", "perspective", "essay", "viewpoint"]):
        return "Opinion"
    if any(kw in text for kw in ["weekly", "roundup", "digest", "newsletter", "this week in", "recap"]):
        return "Weekly Roundup"
    if any(kw in text for kw in ["minor update", "small fix", "patch notes", "bug fix", "hotfix", "maintenance"]):
        return "Minor Patch"

    # Valids
    if any(kw in text for kw in ["funding", "billion", "million", "raises", "acquisition", "acquired", "valuation", "ipo", "investment"]):
        return "Business/Funding"
    if any(kw in text for kw in ["research", "paper", "arxiv", "study", "benchmark", "breakthrough", "proposed method", "outperforms"]):
        return "Research Breakthrough"
    if any(kw in text for kw in ["tool", "app", "plugin", "extension", "platform", "sdk", "api available", "open source"]):
        return "AI Tool"
    if any(kw in text for kw in ["launch", "release", "releases", "introduces", "unveils", "announces", "new model", "gpt", "claude", "gemini", "llama", "mistral"]):
        return "Product Launch"
    if any(kw in text for kw in ["breaking", "just in", "developing", "urgent", "major"]):
        return "Breaking News"

    return "Other"


# ---------------------------------------------------------------------------
# LLM scoring + classification
# ---------------------------------------------------------------------------

_LLM_SYSTEM = """\
You are an expert YouTube Shorts content strategist for the AI news niche.
You evaluate AI news stories for viral potential and classify them accurately.
Always return valid JSON only — no prose, no markdown fences.
"""

_LLM_PROMPT_TEMPLATE = """\
Evaluate this AI news story for a YouTube Shorts channel.

Title:   {title}
Source:  {source}
Summary: {summary}

Return ONLY a JSON object with exactly these keys:
  "score":    integer 0-100
              (viral potential: novelty, consumer impact, curiosity, controversy,
               broad appeal; major launches/breakthroughs get 80+; minor patches get <30)
  "category": exactly one of:
              "Breaking News", "Product Launch", "AI Tool",
              "Research Breakthrough", "Business/Funding",
              "Opinion", "Tutorial", "Weekly Roundup", "Minor Patch", "Other"
  "is_valid": true if category is one of the first five above, false otherwise

JSON only. No prose. No markdown.
"""


def _llm_evaluate(
    chat_fn,
    story: Story,
    model: str | None,
) -> tuple[Optional[int], str, bool]:
    """Ask the LLM to score and classify one story.

    Returns (score, category, is_valid).
    Returns (None, "Unknown", True) on any failure so caller falls back.
    """
    prompt = _LLM_PROMPT_TEMPLATE.format(
        title=story.title,
        source=story.source,
        summary=story.summary or "(no summary available)",
    )
    try:
        raw = chat_fn(_LLM_SYSTEM, prompt, model)
        # Strip any markdown fences the model might add
        raw = re.sub(r"```[^\n]*\n?|```", "", raw).strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return None, "Unknown", True
        data = json.loads(raw[start : end + 1])

        score = int(data.get("score", 0) or 0)
        score = max(0, min(100, score))

        category = str(data.get("category", "Other")).strip()
        if category not in ALL_CATEGORIES:
            category = "Other"

        is_valid = bool(data.get("is_valid", category in VALID_CATEGORIES))
        return score, category, is_valid

    except Exception:
        return None, "Unknown", True


# ---------------------------------------------------------------------------
# Main scoring + classification entry point
# ---------------------------------------------------------------------------

def score_and_classify(
    stories: list[Story],
    *,
    provider=None,
    model: str | None = None,
    verbose: bool = True,
) -> list[Story]:
    """Score and classify all stories; sort by effective score descending.

    If a provider with a ``chat()`` method is available the top 30
    keyword-pre-scored candidates are sent to the LLM for smarter scoring
    and classification.  All others use the keyword heuristic.

    Modifies stories in-place and also returns them.
    """
    # Step 1: fast keyword pre-score for everyone
    for story in stories:
        story.keyword_score = _keyword_score(story)

    chat = getattr(provider, "chat", None) if provider is not None else None
    use_llm = callable(chat)

    if use_llm:
        # Only send the top 30 pre-scored stories to the LLM (saves time/cost)
        ordered = sorted(stories, key=lambda s: s.keyword_score, reverse=True)
        llm_pool = ordered[:30]
        rest = ordered[30:]

        # Heuristic-only for the tail
        for story in rest:
            story.category = _heuristic_category(story)
            story.is_valid = story.category in VALID_CATEGORIES

        if verbose:
            print(f"  LLM scoring top {len(llm_pool)} candidates (keyword fallback for rest)…")

        for idx, story in enumerate(llm_pool):
            if verbose:
                short_title = story.title[:55] + ("…" if len(story.title) > 55 else "")
                print(f"  [{idx + 1:2}/{len(llm_pool)}] {short_title}", end="\r", flush=True)
            llm_sc, cat, valid = _llm_evaluate(chat, story, model)
            if llm_sc is not None:
                story.llm_score = llm_sc
                story.category = cat
                story.is_valid = valid
            else:
                # LLM failed for this story — fall back to heuristic
                story.category = _heuristic_category(story)
                story.is_valid = story.category in VALID_CATEGORIES

        if verbose:
            print()  # clear the \r progress line

    else:
        if verbose:
            print(f"  keyword scoring {len(stories)} stories (no LLM provider)…")
        for story in stories:
            story.category = _heuristic_category(story)
            story.is_valid = story.category in VALID_CATEGORIES

    stories.sort(key=lambda s: s.score, reverse=True)
    return stories


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def top_stories(
    stories: list[Story],
    n: int = 10,
    *,
    valid_only: bool = True,
) -> list[Story]:
    """Return the top *n* stories by effective score.

    Parameters
    ----------
    valid_only:
        If True (default) only stories whose category is in VALID_CATEGORIES
        are considered.  Set False to include all categories.
    """
    pool = [s for s in stories if s.is_valid] if valid_only else stories
    return sorted(pool, key=lambda s: s.score, reverse=True)[:n]


def best_story(stories: list[Story]) -> Optional[Story]:
    """Return the single highest-scoring valid story, or None if none exist."""
    valid = [s for s in stories if s.is_valid]
    return valid[0] if valid else None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_CATEGORY_EMOJI = {
    "Breaking News":         "🔴",
    "Product Launch":        "🚀",
    "AI Tool":               "🛠",
    "Research Breakthrough": "🔬",
    "Business/Funding":      "💰",
    "Opinion":               "💬",
    "Tutorial":              "📖",
    "Weekly Roundup":        "📋",
    "Minor Patch":           "🔧",
    "Other":                 "•",
    "Unknown":               "•",
}


def print_news_board(stories: list[Story], *, top_n: int = 10) -> None:
    """Print a formatted leaderboard of today's top AI news stories."""
    valid = top_stories(stories, n=top_n, valid_only=True)

    width = 62
    print("\n" + "═" * width)
    print("  TODAY'S AI NEWS".center(width))
    print("═" * width)

    if not valid:
        print("\n  No valid stories found.\n")
        print("═" * width)
        return

    for i, story in enumerate(valid, 1):
        age = story.age_hours
        age_str = f"{int(age)}h ago" if age < 48 else f"{int(age / 24)}d ago"
        emoji = _CATEGORY_EMOJI.get(story.category, "•")
        # Truncate title to fit nicely
        title = story.title if len(story.title) <= 56 else story.title[:53] + "…"
        print(f"\n  {i:2}. {title}")
        print(f"      Score: {story.score:3d}  {emoji} {story.category}")
        print(f"      {story.source}  ·  {age_str}")

    print("\n" + "═" * width + "\n")


# ---------------------------------------------------------------------------
# Article extraction + grounding pipeline
# ---------------------------------------------------------------------------

_ARTICLE_FETCH_TIMEOUT = 20   # seconds
_ARTICLE_MAX_CHARS     = 8_000  # truncate very long articles
_ARTICLE_CACHE_DIR     = Path("output") / "_articles_cache"
_ARTICLE_MAX_AGE_HOURS = 24.0

# HTML tags whose text content we always discard
_SKIP_TAGS = {
    "script", "style", "noscript", "header", "footer",
    "nav", "aside", "form", "button", "svg", "iframe",
}


class _ArticleHTMLParser(html.parser.HTMLParser):
    """Minimal HTML → plain-text extractor (stdlib only)."""

    def __init__(self):
        super().__init__()
        self._skip_depth: int = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        raw = " ".join(self._parts)
        # Collapse whitespace runs
        return re.sub(r"\s{2,}", " ", raw).strip()


def _url_cache_key(url: str) -> str:
    """Return a filename-safe hash for *url*."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def _cache_path(url: str, cache_dir: Path) -> Path:
    return cache_dir / f"{_url_cache_key(url)}.txt"


def _load_cached_article(url: str, cache_dir: Path) -> Optional[str]:
    """Return cached article text if it exists and is < 24 h old, else None."""
    p = _cache_path(url, cache_dir)
    if not p.is_file():
        return None
    age_hours = (time.time() - p.stat().st_mtime) / 3600
    if age_hours > _ARTICLE_MAX_AGE_HOURS:
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def _save_cached_article(url: str, text: str, cache_dir: Path) -> None:
    """Write *text* to the article cache."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_path(url, cache_dir).write_text(text, encoding="utf-8")
    except Exception:
        pass  # caching failure is non-fatal


def extract_article_text(
    url: str,
    cache_dir: Path | None = None,
    *,
    verbose: bool = False,
) -> str:
    """Download *url* and return clean plain-text content.

    Results are cached in *cache_dir* (default: ``output/_articles_cache/``).
    A cached copy younger than 24 h is returned without a network request.

    Returns empty string on any failure — callers must check.
    """
    if cache_dir is None:
        cache_dir = _ARTICLE_CACHE_DIR

    # --- cache hit ---
    cached = _load_cached_article(url, cache_dir)
    if cached is not None:
        if verbose:
            print(f"    [article] cache hit: {len(cached)} chars")
        return cached

    # --- fetch ---
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=_ARTICLE_FETCH_TIMEOUT) as resp:
            raw_bytes = resp.read()
    except Exception as exc:
        if verbose:
            print(f"    [article] fetch failed: {exc}")
        return ""

    # --- decode ---
    try:
        html_text = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        html_text = raw_bytes.decode("latin-1", errors="replace")

    # --- strip HTML ---
    parser = _ArticleHTMLParser()
    try:
        parser.feed(html_text)
        text = parser.get_text()
    except Exception:
        # Final fallback: crude tag-strip regex
        text = re.sub(r"<[^>]+>", " ", html_text)
        text = re.sub(r"\s{2,}", " ", text).strip()

    # Truncate to avoid sending huge articles to the LLM
    text = text[:_ARTICLE_MAX_CHARS]

    if not text:
        if verbose:
            print("    [article] could not extract text from page")
        return ""

    _save_cached_article(url, text, cache_dir)
    if verbose:
        print(f"    [article] extracted {len(text)} chars")
    return text


_SUMMARISE_SYSTEM = """\
You are a factual notes writer for a YouTube Shorts news channel.
Your job is to read an AI news article and extract ONLY verified facts.
Do NOT invent, infer, or embellish.
Do NOT add opinions or predictions.
Return a compact bulleted list of facts (max 15 bullets, max 200 words total).
Each bullet must be directly supported by the article text.
"""

_SUMMARISE_PROMPT = """\
Extract factual notes from this article for use in a YouTube Short narration.

Article title: {title}

Article text:
\"\"\"
{text}
\"\"\"

Return ONLY a bulleted list of concrete facts from this article.
- Start each bullet with a dash (-)
- Stick to what the article actually states
- Max 15 bullets, max 200 words
- No invented details, no speculation
"""


def _extract_top_paragraphs(text: str, max_paragraphs: int = 5, min_len: int = 60) -> str:
    """Extract the top *max_paragraphs* longest/most informative sentence groups.

    Splits the article into sentences, groups into paragraphs by punctuation,
    and returns the most content-dense ones joined by newlines.
    """
    # Split on common sentence boundaries to get candidate paragraphs
    raw_chunks = re.split(r"\.\s+|\!\s+|\?\s+", text)
    # Keep chunks that are long enough to be informative
    chunks = [c.strip() for c in raw_chunks if len(c.strip()) >= min_len]
    # Sort by length descending (longer = more informative), take top N
    top = sorted(chunks, key=len, reverse=True)[:max_paragraphs]
    # Re-order by original position to preserve narrative flow
    ordered = [c for c in chunks if c in top]
    return "\n\n".join(f"• {c}." for c in ordered[:max_paragraphs])


def summarize_article(
    title: str,
    text: str,
    provider=None,
    model: str | None = None,
) -> str:
    """Ask the LLM to distil *text* into factual bullet notes.

    Falls back to a truncated excerpt when the LLM is unavailable.
    The returned string always includes:
      1. A structured bullet list of facts (LLM or excerpt fallback)
      2. The top 5 most informative paragraphs from the article
    This dual structure ensures the script LLM has both curated summaries
    and verbatim article text to draw from.
    """
    chat = getattr(provider, "chat", None)
    top_paragraphs = _extract_top_paragraphs(text)

    if callable(chat) and text:
        prompt = _SUMMARISE_PROMPT.format(title=title, text=text[:4000])
        try:
            notes = chat(_SUMMARISE_SYSTEM, prompt, model).strip()
            if notes:
                # Combine LLM bullets with top paragraphs for richer context
                return (
                    f"FACTUAL BULLETS (verified against article):\n{notes}\n\n"
                    f"TOP ARTICLE PARAGRAPHS (use verbatim product/tool names from here):\n"
                    f"{top_paragraphs}"
                )
        except Exception:
            pass

    # Heuristic fallback: raw excerpt + top paragraphs
    excerpt = text[:500].strip()
    notes_section = f"[article excerpt — LLM unavailable]\n{excerpt}" if excerpt else ""
    if top_paragraphs:
        return f"{notes_section}\n\nTOP ARTICLE PARAGRAPHS:\n{top_paragraphs}"
    return notes_section


def ground_story(
    story: "Story",
    provider=None,
    model: str | None = None,
    cache_dir: Path | None = None,
    *,
    verbose: bool = True,
) -> "Story":
    """Populate *story* with extracted article text + factual notes.

    Modifies *story* in-place and also returns it.
    Sets ``story.article_extracted = False`` on any failure so callers can skip.
    """
    if not story.url:
        if verbose:
            print("    [ground] no URL — cannot extract article")
        return story

    if verbose:
        print(f"    [ground] downloading article: {story.url[:80]}")

    text = extract_article_text(story.url, cache_dir, verbose=verbose)
    if not text:
        if verbose:
            print("    [ground] extraction failed — story will be skipped")
        story.article_extracted = False
        return story

    story.article_text = text
    story.article_chars = len(text)
    story.article_extracted = True

    if verbose:
        print(f"    [ground] summarising {len(text)} chars → factual notes…")

    notes = summarize_article(story.title, text, provider, model)
    story.article_notes = notes

    if verbose:
        note_words = len(notes.split()) if notes else 0
        print(f"    [ground] notes: {note_words} words")

    return story


# ---------------------------------------------------------------------------
# Convenience top-level functions (called by main.py)
# ---------------------------------------------------------------------------

def fetch_news(max_per_source: int = 15, *, verbose: bool = True) -> list[Story]:
    """Fetch from all sources, deduplicate, and return raw Story list.

    Each source that fails is skipped silently (printed if *verbose*).
    """
    all_stories: list[Story] = []

    if verbose:
        print("  [news] fetching RSS feeds…")
    for name, url in RSS_SOURCES:
        all_stories.extend(_fetch_rss(name, url, max_per_source, verbose=verbose))

    if verbose:
        print("  [news] fetching Hacker News…")
    all_stories.extend(_fetch_hackernews(max_per_source, verbose=verbose))

    if verbose:
        print("  [news] fetching Reddit…")
    for sub in REDDIT_SUBS:
        all_stories.extend(_fetch_reddit(sub, max_per_source, verbose=verbose))

    before = len(all_stories)
    all_stories = deduplicate(all_stories)
    if verbose:
        print(f"  [news] deduplication: {before} → {len(all_stories)} unique stories")

    return all_stories


def fetch_and_score(
    *,
    provider=None,
    model: str | None = None,
    max_per_source: int = 15,
    verbose: bool = True,
) -> list[Story]:
    """Full pipeline: fetch → deduplicate → score → classify → sort.

    Parameters
    ----------
    provider:
        An LLM provider object (OllamaProvider, OpenAIProvider, …) with a
        ``chat(system, user, model)`` method.  Pass None to use keyword
        heuristic scoring only.
    model:
        Optional model name passed through to the provider.
    max_per_source:
        Maximum items fetched per source.
    verbose:
        Print progress to stdout.

    Returns
    -------
    list[Story]
        All stories sorted by effective score descending.
    """
    stories = fetch_news(max_per_source=max_per_source, verbose=verbose)
    return score_and_classify(stories, provider=provider, model=model, verbose=verbose)
