"""Quality assurance pipeline — shorts_gen/quality.py

All quality gates, scoring, validation, sanitization, and reporting for yt-gen.
Every LLM call degrades gracefully to a heuristic fallback when the provider
is unavailable or the call fails.

Public API (consumed by main.py)
---------------------------------
  story_passes_gate(story)                          → (bool, reasons)
  score_audience_appeal(story, *, provider, model)  → AudienceScores
  combined_story_score(story, audience)             → float
  validate_script(text, topic, *, provider, model)  → ScriptValidation
  fact_safety_check(text, topic, *, provider, model)→ (bool, invented_list)
  sanitize_visual_prompts(prompts, *, provider, model, verbose)
                                                    → (prompts, num_rewritten)
  build_production_report(...)                      → ProductionReport
  write_production_report(report, path)             → Path
  print_quality_report(report)
  build_history_entry(report, video_path)           → dict
  append_history(entry, history_path)

Dependency tree (no circular imports)
--------------------------------------
  quality.py → news.py  (Story type, TYPE_CHECKING guard only)
  quality.py → stdlib only at runtime
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .news import Story  # only used in type hints; no runtime circular import


# ---------------------------------------------------------------------------
# 1.  Story quality gate
# ---------------------------------------------------------------------------

GATE_MIN_SCORE: int = 85          # minimum viral score to proceed
GATE_MAX_AGE_HOURS: float = 48.0  # reject stories older than this

GATE_VALID_CATEGORIES: set[str] = {
    "Breaking News",
    "Product Launch",
    "AI Tool",
    "Research Breakthrough",
    "Business/Funding",
}

GATE_INVALID_CATEGORIES: set[str] = {
    "Opinion",
    "Tutorial",
    "Weekly Roundup",
    "Minor Patch",
    "Other",
    "Unknown",
}


def story_passes_gate(story: "Story") -> tuple[bool, list[str]]:
    """Check whether a story passes ALL quality gate criteria.

    Returns
    -------
    (passes, rejection_reasons)
        passes           True only when every criterion is satisfied.
        rejection_reasons Human-readable list explaining each failure.
    """
    reasons: list[str] = []

    if story.score < GATE_MIN_SCORE:
        reasons.append(
            f"viral score {story.score}/100 is below the minimum {GATE_MIN_SCORE}"
        )

    if story.age_hours > GATE_MAX_AGE_HOURS:
        reasons.append(
            f"story is {story.age_hours:.0f}h old "
            f"(limit: {int(GATE_MAX_AGE_HOURS)}h)"
        )

    if story.category not in GATE_VALID_CATEGORIES:
        reasons.append(
            f"category '{story.category}' is excluded "
            f"(valid: {', '.join(sorted(GATE_VALID_CATEGORIES))})"
        )

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# 2.  Audience appeal scoring
# ---------------------------------------------------------------------------

@dataclass
class AudienceScores:
    """Five-dimension + overall audience appeal score for a news story."""

    curiosity: int            = 5   # 0–10: makes viewer curious
    shock_value: int          = 5   # 0–10: surprising or alarming
    mass_appeal: int          = 5   # 0–10: how many non-technical people care
    shareability: int         = 5   # 0–10: would viewers share this Short
    retention_potential: int  = 5   # 0–10: would viewers watch all the way through
    overall: int              = 50  # 0–100: combined audience score
    reasoning: str            = ""  # one-sentence explanation

    @classmethod
    def neutral(cls) -> "AudienceScores":
        """Return a neutral 50/100 set — used when LLM is unavailable."""
        return cls()


_AUDIENCE_SYSTEM = """\
You are a YouTube Shorts content strategist specialising in AI news virality.
Score news stories for their audience appeal on short-form video platforms.
Return valid JSON only — no prose, no markdown fences.
"""

_AUDIENCE_PROMPT = """\
Evaluate this AI news story for YouTube Shorts audience appeal.

Title:    {title}
Category: {category}
Score:    {score}/100 (viral potential)
Summary:  {summary}

Return ONLY a JSON object with these exact keys:
  "curiosity":           integer 0-10  (makes a viewer intensely curious)
  "shock_value":         integer 0-10  (surprising, alarming, or jaw-dropping)
  "mass_appeal":         integer 0-10  (how many non-technical people care)
  "shareability":        integer 0-10  (would average viewers share this Short)
  "retention_potential": integer 0-10  (would viewers watch all 40 seconds)
  "overall":             integer 0-100 (holistic audience score)
  "reasoning":           string        (one sentence explaining the overall score)

Be harsh. Most stories score 40–65. Only genuinely viral news gets 80+.
JSON only. No prose.
"""


def score_audience_appeal(
    story: "Story",
    *,
    provider=None,
    model: str | None = None,
) -> AudienceScores:
    """Ask the LLM to score the story on five audience-appeal dimensions.

    Falls back to neutral (50/100) if the LLM is unavailable or fails.
    """
    chat = getattr(provider, "chat", None)
    if not callable(chat):
        return AudienceScores.neutral()

    prompt = _AUDIENCE_PROMPT.format(
        title=story.title,
        category=story.category,
        score=story.score,
        summary=story.summary or "(no summary)",
    )
    try:
        raw = chat(_AUDIENCE_SYSTEM, prompt, model)
        raw = re.sub(r"```[^\n]*\n?|```", "", raw).strip()
        s, e = raw.find("{"), raw.rfind("}")
        if s == -1 or e <= s:
            return AudienceScores.neutral()
        d = json.loads(raw[s : e + 1])

        def _int(key: str, default: int, lo: int, hi: int) -> int:
            return max(lo, min(hi, int(d.get(key, default) or default)))

        return AudienceScores(
            curiosity=_int("curiosity", 5, 0, 10),
            shock_value=_int("shock_value", 5, 0, 10),
            mass_appeal=_int("mass_appeal", 5, 0, 10),
            shareability=_int("shareability", 5, 0, 10),
            retention_potential=_int("retention_potential", 5, 0, 10),
            overall=_int("overall", 50, 0, 100),
            reasoning=str(d.get("reasoning", "") or ""),
        )
    except Exception:
        return AudienceScores.neutral()


def combined_story_score(
    story: "Story",
    audience: Optional[AudienceScores],
) -> float:
    """Weighted score: 60 % viral potential + 40 % audience appeal."""
    viral = float(story.score)
    aud = float(audience.overall) if audience is not None else 50.0
    return 0.6 * viral + 0.4 * aud


# ---------------------------------------------------------------------------
# 3.  Script validation
# ---------------------------------------------------------------------------

MIN_WORDS: int = 90
MAX_WORDS: int = 120
MIN_SECONDS: float = 35.0
MAX_SECONDS: float = 45.0
_WORDS_PER_SECOND: float = 150 / 60  # ~2.5 words / second

# How many outer regeneration attempts after the first script generation
MAX_SCRIPT_REGEN_ATTEMPTS: int = 1

# Hard minimum for the structure_score before a script is accepted for publishing.
# Scripts scoring below this are rejected and trigger regeneration.
SCRIPT_MIN_STRUCTURE_SCORE: int = 8

# Maximum per-story regeneration rounds before the story is abandoned.
SCRIPT_MAX_REGEN_ROUNDS: int = 5


@dataclass
class ScriptValidation:
    """Full quality audit result for a generated script."""

    word_count: int          = 0
    estimated_seconds: float = 0.0
    has_hook: bool           = False
    has_curiosity_gap: bool  = False
    has_payoff: bool         = False
    has_why_it_matters: bool = False
    has_cta: bool            = False
    has_invented_facts: bool = False
    hook_score: int          = 0   # 1–10
    structure_score: int     = 0   # 1–10
    issues: list[str]        = field(default_factory=list)

    @property
    def passes(self) -> bool:
        """True only when ALL mandatory checks are satisfied."""
        return (
            MIN_WORDS <= self.word_count <= MAX_WORDS
            and MIN_SECONDS <= self.estimated_seconds <= MAX_SECONDS
            and self.has_hook
            and self.has_curiosity_gap
            and self.has_payoff
            and self.has_why_it_matters
            and self.has_cta
            and not self.has_invented_facts
        )

    @property
    def score(self) -> int:
        return self.structure_score


_VALIDATION_SYSTEM = """\
You are a YouTube Shorts script quality auditor for AI news content.
Analyse the script strictly against its topic and return a JSON report.
Return valid JSON only — no prose, no markdown fences.
"""

_VALIDATION_PROMPT = """\
Audit this YouTube Shorts narration script.

Topic (ALL allowed facts come from here ONLY):
{topic}

Script:
\"\"\"
{script}
\"\"\"

Return ONLY this JSON object (no extra keys, no prose):
{{
  "word_count":         <integer — exact word count>,
  "has_hook":           <boolean — strong scroll-stopping first sentence, ≤15 words?>,
  "has_curiosity_gap":  <boolean — a sentence that teases the payoff without revealing it?>,
  "has_payoff":         <boolean — main information clearly delivered?>,
  "has_why_it_matters": <boolean — impact or consequence of the news explained?>,
  "has_cta":            <boolean — ends with a follow/subscribe call to action?>,
  "has_invented_facts": <boolean — does the script use names/stats/quotes/dates NOT in topic?>,
  "invented_items":     <array of strings — specific invented items, or []>,
  "hook_score":         <integer 1-10 — how scroll-stopping is the opening line?>,
  "structure_score":    <integer 1-10 — overall structure and flow quality?>,
  "issues":             <array of up to 5 specific, actionable problems>
}}
"""


def _heuristic_validate(script_text: str) -> ScriptValidation:
    """Keyword-based fallback when no LLM provider is available."""
    wc = len(re.findall(r"\b[\w'-]+\b", script_text))
    est = round(wc / _WORDS_PER_SECOND, 1)
    lower = script_text.lower()
    first = (re.split(r"(?<=[.!?])\s+", script_text.strip(), maxsplit=1)[:1] or [""])[0]
    has_hook = 4 <= len(re.findall(r"\b[\w'-]+\b", first)) <= 20
    has_cta = any(
        kw in lower
        for kw in ("follow", "subscribe", "hit follow", "tap follow", "more ai")
    )
    issues: list[str] = []
    if not (MIN_WORDS <= wc <= MAX_WORDS):
        issues.append(f"word count {wc} outside {MIN_WORDS}–{MAX_WORDS}")
    if not has_hook:
        issues.append("weak or missing hook sentence")
    if not has_cta:
        issues.append("missing follow/subscribe CTA")

    ok = MIN_WORDS <= wc <= MAX_WORDS and has_hook and has_cta
    return ScriptValidation(
        word_count=wc,
        estimated_seconds=est,
        has_hook=has_hook,
        has_curiosity_gap=True,   # cannot reliably detect heuristically
        has_payoff=True,
        has_why_it_matters=True,
        has_cta=has_cta,
        has_invented_facts=False,  # handled by fact_safety_check()
        hook_score=7 if has_hook else 3,
        structure_score=7 if ok else 4,
        issues=issues,
    )


def validate_script(
    script_text: str,
    topic: str,
    *,
    provider=None,
    model: str | None = None,
    article_text: str = "",
) -> ScriptValidation:
    """Validate a generated script against all quality criteria.

    Uses an LLM for structural analysis when available; falls back to heuristic.
    Word count and duration are always computed heuristically (more reliable than
    asking the LLM to count words).

    When *article_text* is provided the LLM uses the full article to verify
    whether specific names and claims are supported — reducing false positives
    that occur when checking only against a title.

    Note: for the dedicated final fact-safety gate, call ``fact_safety_check()``.
    """
    wc = len(re.findall(r"\b[\w'-]+\b", script_text))
    est = round(wc / _WORDS_PER_SECOND, 1)

    chat = getattr(provider, "chat", None)
    if not callable(chat):
        val = _heuristic_validate(script_text)
        val.word_count = wc
        val.estimated_seconds = est
        return val

    # Build topic context: article title + truncated article text if available
    if article_text:
        grounding_ctx = (
            f"Article title: {topic}\n\n"
            f"Article content (first 2000 chars):\n{article_text[:2000]}"
        )
    else:
        grounding_ctx = f"Topic: {topic}"

    prompt = _VALIDATION_PROMPT.format(topic=grounding_ctx, script=script_text)
    try:
        raw = chat(_VALIDATION_SYSTEM, prompt, model)
        raw = re.sub(r"```[^\n]*\n?|```", "", raw).strip()
        s, e = raw.find("{"), raw.rfind("}")
        if s == -1 or e <= s:
            return _heuristic_validate(script_text)
        d = json.loads(raw[s : e + 1])

        def _b(key: str) -> bool:
            return bool(d.get(key, False))

        def _i(key: str, default: int) -> int:
            return max(1, min(10, int(d.get(key, default) or default)))

        return ScriptValidation(
            word_count=wc,                       # trust heuristic count
            estimated_seconds=est,
            has_hook=_b("has_hook"),
            has_curiosity_gap=_b("has_curiosity_gap"),
            has_payoff=_b("has_payoff"),
            has_why_it_matters=_b("has_why_it_matters"),
            has_cta=_b("has_cta"),
            has_invented_facts=_b("has_invented_facts"),
            hook_score=_i("hook_score", 5),
            structure_score=_i("structure_score", 5),
            issues=[str(x) for x in (d.get("issues") or []) if x][:5],
        )
    except Exception:
        return _heuristic_validate(script_text)


# ---------------------------------------------------------------------------
# 4.  Fact safety gate  (final check before TTS)
# ---------------------------------------------------------------------------

_FACT_CHECK_SYSTEM = """\
You are a strict fact-checker protecting an AI news channel from misinformation.
Your role is to identify fabricated or invented content in scripts.
Return valid JSON only — no prose, no markdown fences.
"""

_FACT_CHECK_PROMPT = """\
Fact-check this AI news narration against its source topic.

Source topic (ONLY facts stated here are permitted in the script):
{topic}

Script to fact-check:
\"\"\"
{script}
\"\"\"

Identify ANY content that was NOT stated or directly implied by the topic above.
Specifically check for:
  1. Product or model names not in the topic
  2. Numbers, percentages, or statistics not in the topic
  3. Quoted speech or attributed statements not in the topic
  4. Specific features or technical capabilities not mentioned
  5. Dates, timelines, or deadlines not in the topic
  6. Third-party company names or partnerships not mentioned
  7. People's names, titles, or roles not mentioned

Return ONLY JSON:
{{
  "safe":     <boolean — true ONLY if ZERO invented facts were found>,
  "invented": <array of strings — each invented item; empty array if none>
}}
"""


def fact_safety_check(
    script_text: str,
    topic: str,
    *,
    provider=None,
    model: str | None = None,
    article_text: str = "",
) -> tuple[bool, list[str]]:
    """Dedicated final hallucination gate before script goes to TTS.

    Returns
    -------
    (is_safe, invented_items)
        is_safe        True if no invented facts were detected.
        invented_items Specific invented items found (empty when safe).

    When *article_text* is provided the checker verifies claims against the
    full article rather than only the title — eliminating false positives for
    real product names that appear in the article but not the title.

    Fails-open (returns True, []) when no LLM is available, so the pipeline
    is not blocked — the anti-hallucination prompts in script.py are the
    primary protection.
    """
    chat = getattr(provider, "chat", None)
    if not callable(chat):
        return True, []

    # Provide richer grounding context when article text is available
    if article_text:
        grounding = (
            f"Article title: {topic}\n\n"
            f"Article content (first 3000 chars — use this as the source of truth):\n"
            f"{article_text[:3000]}"
        )
    else:
        grounding = f"Source topic (ONLY facts stated here are permitted in the script):\n{topic}"

    prompt = _FACT_CHECK_PROMPT.format(topic=grounding, script=script_text)
    try:
        raw = chat(_FACT_CHECK_SYSTEM, prompt, model)
        raw = re.sub(r"```[^\n]*\n?|```", "", raw).strip()
        s, e = raw.find("{"), raw.rfind("}")
        if s == -1 or e <= s:
            return True, []
        d = json.loads(raw[s : e + 1])
        safe = bool(d.get("safe", True))
        invented = [str(x) for x in (d.get("invented") or []) if x]
        return safe, invented
    except Exception:
        return True, []


# ---------------------------------------------------------------------------
# 5.  Visual prompt sanitisation
# ---------------------------------------------------------------------------

# Multi-word phrases that signal non-cinematic content in an image prompt.
_BAD_VISUAL_PHRASES: list[str] = [
    "with text", "showing text", "displaying text", "text overlay",
    "text on screen", "with words", "with letters", "with captions",
    "subtitle", "with logo", "showing logo", "brand logo", "watermark",
    "user interface", "ui element", "app interface", "mobile app screen",
    "screenshot", "screen capture", "screengrab",
    "mockup", "wireframe", "infographic", "with overlay",
    "dashboard view", "showing the app", "website screenshot",
    "with popup", "notification badge", "app icon",
]

_VISUAL_FIX_SYSTEM = """\
You are an AI image prompt engineer specialising in cinematic visual scenes.
Rewrite rejected prompts to be purely visual — no text, no UI, no logos.
Return only the rewritten prompt, no commentary or explanation.
"""

_VISUAL_FIX_PROMPT = """\
This image-generation prompt was rejected for containing text, logos, UI elements,
screenshots, or other non-cinematic content.

Rejected prompt:
{prompt}

Rewrite it as a pure cinematic visual scene following ALL these rules:
- NO text, letters, words, captions, or subtitles of any kind
- NO logos, brand marks, or watermarks
- NO UI elements, app screens, dashboards, or wireframes
- NO screenshots or screen captures
- FOCUS ON: people, physical environments, technology hardware, dramatic abstract
  compositions, atmospheric lighting
- Keep the core AI / technology concept but express it through physical imagery
- Style: cinematic, ultra-detailed, dramatic lighting, high contrast,
  9:16 vertical composition, photorealistic, 8k

Return ONLY the rewritten prompt. No explanation.
"""


def _prompt_has_bad_visual(prompt: str) -> bool:
    lower = prompt.lower()
    return any(phrase in lower for phrase in _BAD_VISUAL_PHRASES)


def _fix_visual_prompt(
    prompt: str,
    *,
    provider=None,
    model: str | None = None,
) -> str:
    """Rewrite a flagged prompt via LLM, or apply simple keyword stripping."""
    chat = getattr(provider, "chat", None)
    if callable(chat):
        try:
            fixed = chat(
                _VISUAL_FIX_SYSTEM,
                _VISUAL_FIX_PROMPT.format(prompt=prompt),
                model,
            ).strip()
            if fixed and len(fixed) > 20:
                return fixed
        except Exception:
            pass

    # Heuristic fallback: strip bad phrases and re-wrap in cinematic framing
    cleaned = prompt
    for phrase in _BAD_VISUAL_PHRASES:
        cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(".,;:")
    return (
        f"Cinematic scene: {cleaned}. "
        "Dramatic lighting, ultra-detailed, photorealistic, 9:16 vertical."
    )


def sanitize_visual_prompts(
    prompts: list[str],
    *,
    provider=None,
    model: str | None = None,
    verbose: bool = True,
) -> tuple[list[str], int]:
    """Rewrite any prompts that contain text, logos, UI, or screenshots.

    Returns
    -------
    (sanitized_prompts, num_rewritten)
    """
    result: list[str] = []
    rewritten = 0

    for i, prompt in enumerate(prompts):
        if _prompt_has_bad_visual(prompt):
            if verbose:
                print(
                    f"  [quality] prompt {i + 1} contains non-cinematic elements "
                    f"→ rewriting"
                )
            fixed = _fix_visual_prompt(prompt, provider=provider, model=model)
            result.append(fixed)
            rewritten += 1
        else:
            result.append(prompt)

    return result, rewritten


# ---------------------------------------------------------------------------
# 6.  Production report
# ---------------------------------------------------------------------------

@dataclass
class ProductionReport:
    """Complete quality metrics for a single Short generation run."""

    # Run metadata
    generated_at: str              = ""
    generation_time_seconds: float = 0.0
    topic: str                     = ""

    # Story (populated only in --auto mode)
    story_source: str              = ""
    story_url: str                 = ""
    story_score: Optional[int]     = None
    story_age_hours: Optional[float] = None
    story_category: str            = ""

    # Audience appeal (populated only in --auto mode)
    audience_curiosity: Optional[int]   = None
    audience_shock_value: Optional[int] = None
    audience_mass_appeal: Optional[int] = None
    audience_shareability: Optional[int]= None
    audience_retention: Optional[int]   = None
    audience_overall: Optional[int]     = None
    audience_reasoning: str             = ""

    # Script quality
    script_score: int            = 0    # structure score 1–10
    hook_score: int              = 0    # hook quality 1–10
    word_count: int              = 0
    estimated_duration: float    = 0.0
    has_hook: bool               = False
    has_curiosity_gap: bool      = False
    has_payoff: bool             = False
    has_why_it_matters: bool     = False
    has_cta: bool                = False
    fact_safe: bool              = True
    invented_items: list[str]    = field(default_factory=list)

    # Images
    image_count: int             = 0
    prompts_sanitized: int       = 0

    # Summary
    warnings: list[str]          = field(default_factory=list)
    final_pass: bool             = True


def build_production_report(
    *,
    topic: str,
    start_time: float,                      # time.time() at pipeline start
    story: "Story | None"              = None,
    audience: Optional[AudienceScores] = None,
    validation: Optional[ScriptValidation] = None,
    fact_safe: bool                    = True,
    invented_items: Optional[list[str]]= None,
    image_count: int                   = 0,
    prompts_sanitized: int             = 0,
) -> ProductionReport:
    """Assemble a ``ProductionReport`` from all pipeline run data."""
    now = datetime.now(timezone.utc)
    gen_time = round(time.time() - start_time, 1)
    warnings: list[str] = []
    final_pass = True

    r = ProductionReport(
        generated_at=now.isoformat(),
        generation_time_seconds=gen_time,
        topic=topic,
    )

    # --- Story ---
    if story is not None:
        r.story_source   = story.source
        r.story_url      = story.url
        r.story_score    = story.score
        r.story_age_hours = round(story.age_hours, 1)
        r.story_category = story.category

    # --- Audience ---
    if audience is not None:
        r.audience_curiosity   = audience.curiosity
        r.audience_shock_value = audience.shock_value
        r.audience_mass_appeal = audience.mass_appeal
        r.audience_shareability= audience.shareability
        r.audience_retention   = audience.retention_potential
        r.audience_overall     = audience.overall
        r.audience_reasoning   = audience.reasoning
        if audience.overall < 55:
            warnings.append(f"low audience score: {audience.overall}/100")

    # --- Script ---
    if validation is not None:
        r.script_score       = validation.structure_score
        r.hook_score         = validation.hook_score
        r.word_count         = validation.word_count
        r.estimated_duration = validation.estimated_seconds
        r.has_hook           = validation.has_hook
        r.has_curiosity_gap  = validation.has_curiosity_gap
        r.has_payoff         = validation.has_payoff
        r.has_why_it_matters = validation.has_why_it_matters
        r.has_cta            = validation.has_cta
        r.fact_safe          = fact_safe
        r.invented_items     = invented_items or []

        if not validation.passes:
            final_pass = False
            for issue in validation.issues:
                warnings.append(f"script: {issue}")
        if not (MIN_WORDS <= validation.word_count <= MAX_WORDS):
            warnings.append(
                f"word count {validation.word_count} outside "
                f"{MIN_WORDS}–{MAX_WORDS}"
            )
        if not fact_safe and invented_items:
            final_pass = False
            warnings.append(
                f"invented facts: {', '.join(invented_items[:3])}"
            )

    # --- Images ---
    r.image_count      = image_count
    r.prompts_sanitized = prompts_sanitized
    if prompts_sanitized > 0:
        warnings.append(
            f"{prompts_sanitized} image prompt(s) contained non-cinematic "
            "elements and were rewritten"
        )

    r.warnings    = warnings
    r.final_pass  = final_pass
    return r


def _report_to_dict(r: ProductionReport) -> dict:
    return {
        "generated_at":              r.generated_at,
        "generation_time_seconds":   r.generation_time_seconds,
        "topic":                     r.topic,
        "story": {
            "source":     r.story_source,
            "url":        r.story_url,
            "viral_score": r.story_score,
            "age_hours":  r.story_age_hours,
            "category":   r.story_category,
        },
        "audience": {
            "curiosity":           r.audience_curiosity,
            "shock_value":         r.audience_shock_value,
            "mass_appeal":         r.audience_mass_appeal,
            "shareability":        r.audience_shareability,
            "retention_potential": r.audience_retention,
            "overall":             r.audience_overall,
            "reasoning":           r.audience_reasoning,
        },
        "script": {
            "structure_score":    r.script_score,
            "hook_score":         r.hook_score,
            "word_count":         r.word_count,
            "estimated_seconds":  r.estimated_duration,
            "has_hook":           r.has_hook,
            "has_curiosity_gap":  r.has_curiosity_gap,
            "has_payoff":         r.has_payoff,
            "has_why_it_matters": r.has_why_it_matters,
            "has_cta":            r.has_cta,
            "fact_safe":          r.fact_safe,
            "invented_items":     r.invented_items,
        },
        "images": {
            "count":              r.image_count,
            "prompts_sanitized":  r.prompts_sanitized,
        },
        "warnings":    r.warnings,
        "final_pass":  r.final_pass,
    }


def write_production_report(report: ProductionReport, path: Path) -> Path:
    """Write ``quality_report.json`` to *path* (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_report_to_dict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def print_quality_report(report: ProductionReport) -> None:
    """Print a concise quality-report summary to stdout."""
    status = "✅ PASS" if report.final_pass else "❌ FAIL"
    width = 64
    bar = lambda n, hi: "█" * n + "░" * (hi - n)

    print("\n" + "─" * width)
    print(f"  QUALITY REPORT  {status}".center(width))
    print("─" * width)

    if report.story_score is not None:
        print(
            f"  Story    │ Viral: {report.story_score}/100  "
            f"│  {report.story_category}  │  {report.story_age_hours:.0f}h ago"
        )
    if report.audience_overall is not None:
        dim = (
            f"curiosity={report.audience_curiosity} "
            f"shock={report.audience_shock_value} "
            f"appeal={report.audience_mass_appeal} "
            f"share={report.audience_shareability} "
            f"retention={report.audience_retention}"
        )
        print(
            f"  Audience │ Overall: {report.audience_overall}/100  │  {dim}"
        )
        if report.audience_reasoning:
            print(f"           │ {report.audience_reasoning}")

    hook_bar = bar(report.hook_score, 10)
    struct_bar = bar(report.script_score, 10)
    print(f"  Script   │ Hook:   {hook_bar} {report.hook_score}/10")
    print(f"           │ Struct: {struct_bar} {report.script_score}/10")
    print(
        f"           │ Words: {report.word_count}  "
        f"│  ~{report.estimated_duration:.0f}s"
    )

    checks = [
        ("Hook",           report.has_hook),
        ("Curiosity Gap",  report.has_curiosity_gap),
        ("Payoff",         report.has_payoff),
        ("Why It Matters", report.has_why_it_matters),
        ("CTA",            report.has_cta),
        ("Fact Safe",      report.fact_safe),
    ]
    row = "  " + "   ".join(
        f"{'✓' if ok else '✗'} {name}" for name, ok in checks
    )
    print(row)

    print(
        f"  Images   │ {report.image_count} generated  "
        f"│  {report.prompts_sanitized} sanitized"
    )

    if report.warnings:
        print("  Warnings │")
        for w in report.warnings:
            print(f"    ⚠  {w}")

    print(f"  Time     │ {report.generation_time_seconds:.0f}s total")
    print("─" * width + "\n")


# ---------------------------------------------------------------------------
# 7.  Generation history
# ---------------------------------------------------------------------------

def build_history_entry(
    report: ProductionReport,
    video_path: Optional[Path] = None,
) -> dict:
    """Build a single history.json entry from a ``ProductionReport``."""
    date_str = report.generated_at[:10] if report.generated_at else ""
    return {
        "date":                     date_str,
        "timestamp":                report.generated_at,
        "topic":                    report.topic,
        "source":                   report.story_source or "manual",
        "url":                      report.story_url or "",
        "viral_score":              report.story_score,
        "audience_score":           report.audience_overall,
        "script_score":             report.script_score,
        "hook_score":               report.hook_score,
        "word_count":               report.word_count,
        "estimated_seconds":        report.estimated_duration,
        "fact_safe":                report.fact_safe,
        "final_pass":               report.final_pass,
        "warnings":                 report.warnings,
        "video_path":               str(video_path) if video_path else "",
        "generation_time_seconds":  report.generation_time_seconds,
    }


def append_history(entry: dict, history_path: Path) -> None:
    """Append *entry* to the generation history JSON array at *history_path*.

    Creates the file with an empty array if it does not exist yet.
    Safe for single-process use.
    """
    history_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if history_path.is_file():
        try:
            existing = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

    existing.append(entry)
    history_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
