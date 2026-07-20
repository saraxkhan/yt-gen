"""Pluggable LLM script generation.

Providers (priority order):
  1. ollama    - local Ollama daemon (default, fully offline, free)
  2. openai    - any OpenAI-compatible HTTP API (optional)
  3. file      - read script from a text file (no LLM at all)

Script quality pipeline:
  - Structured Hook -> Curiosity Gap -> Main Info -> Why It Matters -> CTA
  - Target 90-120 words for a 35-45s Short
  - Generate NUM_CANDIDATES=5 scripts independently, score all, select the best
  - No rewrite loop — each candidate is independent; best wins outright
  - FileProvider bypasses the review loop (user-provided text is final)
  - When article_notes are provided, the prompt is GROUNDED to those notes;
    the LLM may NOT invent any fact beyond what the notes contain.

Anti-hallucination:
  - The system prompt explicitly forbids inventing names, products, stats,
    or facts not present in the topic / article notes.
  - Hard length gate: word_count < 90 → score capped at 3, treated as a fail.

Scoring criteria (6):
  - Hook Strength       – scroll-stopping first sentence, no AI buzzwords
  - Curiosity Gap       – payoff promise that makes scrolling away costly
  - Clarity             – punchy, TTS-friendly sentences, no jargon
  - Retention Potential – pacing that holds attention for 35-45 s
  - Novelty             – fresh angle or surprising framing of the topic
  - Emotional Impact    – evokes curiosity, excitement, or urgency in viewer
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Protocol

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# CRITICAL: The anti-hallucination block is first and unconditional so it
# cannot be overridden by stylistic instructions that follow.
_ANTI_HALLUCINATION_BLOCK = """\
CRITICAL — FACTUAL ACCURACY RULES (non-negotiable, highest priority):
- You MUST NOT invent, fabricate, or assume ANY facts not explicitly present
  in the topic string provided by the user.
- You MUST NOT invent product names, company names, feature names, version
  numbers, statistics, quotes, dates, or any other specific details.
- If the topic mentions "OpenAI launches a new AI agent", write ONLY about
  that. Do NOT invent a name like "CLARITY" or any other name for the agent.
- If a detail is not in the topic, either omit it or use a generic
  description ("a new AI agent", "this tool", "the update").
- When in doubt, stay vague and accurate rather than specific and invented.
- Violating these rules produces misinformation. This is never acceptable.

"""

SYSTEM_PROMPT = _ANTI_HALLUCINATION_BLOCK + """\
You write punchy, scroll-stopping YouTube Shorts narration
for the AI News + AI Tools niche.

Hard rules:
- Plain spoken text only. No stage directions, no headings, no labels
  (do NOT write "Hook:", "CTA:", etc.), no emojis, no markdown, no lists.
- Output ONLY the narration that will be read aloud.
- 90-120 words total. This is a hard window.
- Short, punchy sentences a TTS engine can read naturally.
- Assume ~150 words per minute when pacing.
- Use ONLY information present in the topic. Never add invented details.

Required structure (flow naturally, no labels in the output):
1. HOOK - A single, scroll-stopping first sentence. Bold claim,
   surprising stat, sharp question, or pattern interrupt. Under 15 words.
2. CURIOSITY GAP - One sentence that promises a payoff and makes
   scrolling away feel costly. Tease, do not reveal yet.
3. MAIN INFORMATION - 2-4 sentences delivering the actual substance:
   the news, the tool, what it does, what changed. Concrete and specific,
   but ONLY using details given in the topic.
4. WHY IT MATTERS - 1-2 sentences on impact: who it affects, what it
   unlocks, what becomes possible or obsolete.
5. CTA - One final line telling viewers to follow for more AI news /
   AI tools. Keep it natural, not salesy.

Quality targets:
- Aim for a NOVEL angle — avoid the obvious, clichéd framing.
- Create EMOTIONAL IMPACT: make the viewer feel excited, curious, or
  slightly alarmed by what they are missing.
"""

REVIEW_SYSTEM_PROMPT = """\
You are a ruthless YouTube Shorts script editor for
the AI News + AI Tools niche. You score scripts honestly and harshly.

Return ONLY a single JSON object, no prose, no markdown fences, with keys:
  "score":             integer 1-10 (overall quality for a 35-45s Short)
  "word_count":        integer (actual words in the script)
  "hook_strength":     integer 1-10 (how scroll-stopping the first sentence is)
  "curiosity_gap":     integer 1-10 (how effectively it teases the payoff)
  "clarity":           integer 1-10 (punchy, TTS-friendly, easy to follow)
  "retention_potential": integer 1-10 (pacing holds attention for 35-45 s)
  "novelty":           integer 1-10 (fresh angle; not clichéd or generic framing)
  "emotional_impact":  integer 1-10 (evokes curiosity, excitement, or urgency)
  "has_hook":          boolean (strong scroll-stopping first sentence)
  "has_curiosity":     boolean (a real curiosity gap before the payoff)
  "has_main":          boolean (concrete substance, not vague filler)
  "has_why":           boolean (clear "why it matters")
  "has_cta":           boolean (a final follow / subscribe line)
  "has_invented":      boolean (true if script contains invented names/facts not in topic)
  "issues":            array of short strings (max 5)
  "rewrite_notes":     string (concrete instructions to fix it; empty if score>=8)

Scoring guide:
- 90-120 words, all 5 sections present and strong, NO invented facts,
  high novelty and emotional impact: 8-10
- Missing CTA, weak hook, off length, vague middle, generic framing,
  low emotional impact: 5-7
- Generic, labeled, contains markdown, or wrong topic: 1-4
- Contains invented product names or facts NOT in the topic/notes: score capped at 3
- Word count BELOW 90 or ABOVE 120: score capped at 3 (hard length gate)

HOOK RULES (applied before hook_strength scoring):
- PENALISE hooks using AI buzzwords: "revolutionary", "game-changing",
  "unprecedented", "groundbreaking", "mind-blowing", "next-level",
  "game changer", "transform", "disrupting", "next generation".
  A hook containing any of these words should score no higher than 5 on hook_strength.
- PENALISE exaggerated claims that cannot be verified from the topic alone.
- PENALISE invented facts or names not present in the topic.
- REWARD hooks that create genuine curiosity through a specific, surprising
  fact or a sharp question grounded in the topic.

The "score" field MUST reflect novelty and emotional_impact heavily.
A technically correct but boring/clichéd script should score no higher than 6.
"""

REWRITE_SYSTEM_PROMPT = _ANTI_HALLUCINATION_BLOCK + """\
You write punchy, scroll-stopping YouTube Shorts narration
for the AI News + AI Tools niche.

You are REWRITING a previous draft that failed quality review.
Fix every issue the editor listed. Keep what worked. Do not add labels.
Output ONLY the new narration.
Use ONLY information present in the original topic. Never invent details.

Push for a FRESH ANGLE. If the previous draft was generic or clichéd,
find a more surprising, emotionally engaging way to open and frame the story.
Aim to evoke genuine curiosity or excitement in the viewer.
"""

DEFAULT_OLLAMA_MODEL = "qwen2.5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

MIN_WORDS = 90
MAX_WORDS = 120
TARGET_SCORE = 8
NUM_CANDIDATES = 5        # number of candidate scripts generated and scored independently
# Note: The rewrite/refine loop has been removed.
# Each candidate is generated independently; the best one wins outright.


def _user_message(topic: str, duration_seconds: int, article_notes: str = "") -> str:
    if article_notes:
        return (
            f"GROUNDED FACTS — use ONLY these facts in your narration. "
            f"Do not invent or add anything beyond this list:\n"
            f"{article_notes}\n\n"
            f"Article title: {topic}\n"
            f"Target spoken length: ~{duration_seconds} seconds (90-120 words). "
            f"THIS IS A HARD REQUIREMENT: you MUST write between 90 and 120 words.\n"
            f"IMPORTANT: Use ONLY the grounded facts above. "
            f"Do not invent names, features, or details not in the list.\n"
            f"Write the narration now, following every rule."
        )
    return (
        f"Topic: {topic}\n"
        f"Target spoken length: ~{duration_seconds} seconds (90-120 words). "
        f"THIS IS A HARD REQUIREMENT: you MUST write between 90 and 120 words.\n"
        f"IMPORTANT: Use only the information in the topic above. "
        f"Do not invent names, features, or details.\n"
        f"Write the narration now, following every rule."
    )


def _regen_user_message(
    topic: str, duration_seconds: int, attempt: int, article_notes: str = ""
) -> str:
    """Fresh generation prompt — varied angle nudge helps break repetition."""
    angle_nudges = [
        "Try a bold, counterintuitive hook that challenges a common assumption.",
        "Open with a provocative question that immediately creates stakes.",
        "Start with the most surprising consequence or implication of this topic.",
        "Open with the most concrete, specific fact from the grounded notes.",
        "Frame it from the perspective of what this means for everyday users.",
    ]
    nudge = angle_nudges[(attempt - 1) % len(angle_nudges)]
    if article_notes:
        return (
            f"GROUNDED FACTS — use ONLY these facts. Do not invent anything:\n"
            f"{article_notes}\n\n"
            f"Article title: {topic}\n"
            f"Target spoken length: ~{duration_seconds} seconds (90-120 words). "
            f"THIS IS A HARD REQUIREMENT: you MUST write between 90 and 120 words.\n"
            f"IMPORTANT: Use ONLY the grounded facts above. "
            f"Do not invent names, features, or details not in the list.\n"
            f"Angle hint (use your own words, do not copy this): {nudge}\n"
            f"Write the narration now, following every rule."
        )
    return (
        f"Topic: {topic}\n"
        f"Target spoken length: ~{duration_seconds} seconds (90-120 words). "
        f"THIS IS A HARD REQUIREMENT: you MUST write between 90 and 120 words.\n"
        f"IMPORTANT: Use only the information in the topic above. "
        f"Do not invent names, features, or details.\n"
        f"Angle hint (use your own words, do not copy this): {nudge}\n"
        f"Write the narration now, following every rule."
    )


def _rewrite_user_message(
    topic: str, duration_seconds: int, previous: str, review: dict
) -> str:
    issues = "; ".join(review.get("issues") or []) or "(none listed)"
    notes = review.get("rewrite_notes") or ""
    sub_scores = (
        f"  hook_strength={review.get('hook_strength', '?')}/10, "
        f"curiosity_gap={review.get('curiosity_gap', '?')}/10, "
        f"clarity={review.get('clarity', '?')}/10, "
        f"retention_potential={review.get('retention_potential', '?')}/10, "
        f"novelty={review.get('novelty', '?')}/10, "
        f"emotional_impact={review.get('emotional_impact', '?')}/10"
    )
    return (
        f"Topic: {topic}\n"
        f"Target spoken length: ~{duration_seconds} seconds (90-120 words).\n"
        f"IMPORTANT: Use only the information in the topic above. "
        f"Do not invent names, features, or details.\n\n"
        f"Previous draft (score {review.get('score','?')}/10):\n"
        f'"""\n{previous}\n"""\n\n'
        f"Sub-scores:\n{sub_scores}\n\n"
        f"Editor issues: {issues}\n"
        f"Editor rewrite notes: {notes}\n\n"
        f"Rewrite the narration. Fix every issue. Output narration only."
    )


def _review_user_message(topic: str, script_text: str) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"Script to review:\n\"\"\"\n{script_text}\n\"\"\"\n\n"
        f"Check carefully: does the script invent any names, product names, "
        f"statistics, or specific facts that are NOT present in the topic above? "
        f"If so, set has_invented=true and cap the score at 3.\n"
        f"Also score novelty (1-10) and emotional_impact (1-10) harshly — "
        f"generic or clichéd framing should score 4 or below on those dimensions.\n"
        f"Return the JSON object now."
    )


# ---------------------------------------------------------------------------
# Cleaning + parsing helpers
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(
    r"^\s*(hook|curiosity(?:\s*gap)?|main(?:\s*info(?:rmation)?)?|why(?:\s*it\s*matters)?|cta|call\s*to\s*action)\s*[:\-]\s*",
    re.IGNORECASE | re.MULTILINE,
)
_MD_BULLET_RE = re.compile(r"^\s*[\-\*\d+\.]+\s+", re.MULTILINE)
_MD_FENCE_RE = re.compile(r"```[^\n]*\n?|```")


def _clean_script(text: str) -> str:
    text = _MD_FENCE_RE.sub("", text)
    text = _LABEL_RE.sub("", text)
    text = _MD_BULLET_RE.sub("", text)
    text = text.strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _extract_json(raw: str) -> dict | None:
    if not raw:
        return None
    raw = _MD_FENCE_RE.sub("", raw).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def _heuristic_review(script_text: str) -> dict:
    """Fallback review when the LLM review fails to return valid JSON."""
    wc = _word_count(script_text)
    first = (re.split(r"(?<=[.!?])\s+", script_text.strip(), maxsplit=1)[:1] or [""])[0]
    has_hook = 4 <= _word_count(first) <= 20
    lower = script_text.lower()
    has_cta = any(
        kw in lower
        for kw in ("follow", "subscribe", "hit follow", "tap follow", "more ai")
    )
    in_range = MIN_WORDS <= wc <= MAX_WORDS
    score = 5 + (2 if in_range else 0) + (1 if has_hook else 0) + (1 if has_cta else 0)
    issues = []
    if not in_range:
        issues.append(f"word count {wc} outside {MIN_WORDS}-{MAX_WORDS}")
    if not has_hook:
        issues.append("weak or missing hook sentence")
    if not has_cta:
        issues.append("missing follow/subscribe CTA")
    return {
        "score": score,
        "word_count": wc,
        "hook_strength": 5,
        "curiosity_gap": 5,
        "clarity": 5,
        "retention_potential": 5,
        "novelty": 5,
        "emotional_impact": 5,
        "has_hook": has_hook,
        "has_curiosity": True,
        "has_main": True,
        "has_why": True,
        "has_cta": has_cta,
        "has_invented": False,
        "issues": issues,
        "rewrite_notes": "Fix listed issues; keep 90-120 words; end with a CTA.",
    }


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

class Provider(Protocol):
    def generate(self, topic: str, duration_seconds: int, model: str | None) -> str: ...
    def chat(self, system: str, user: str, model: str | None) -> str: ...


# ---------------------------------------------------------------------------
# Ollama (default, fully offline)
# ---------------------------------------------------------------------------

class OllamaProvider:
    """Calls a local Ollama daemon via its native HTTP API.

    Install:    https://ollama.com/download
    Pull model: `ollama pull llama3.1`   (or qwen2.5, gemma2, mistral, ...)
    """

    def __init__(self, host: str | None = None):
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")

    def _chat(self, messages: list[dict], model: str | None, temperature: float) -> str:
        model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": messages,
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise RuntimeError(
                    f"Ollama model \"{model}\" not found (HTTP 404). "
                    f"Run `ollama pull {model}` to download it, "
                    f"or pass --model <name> with a model you have installed "
                    f"(`ollama list` shows available models)."
                ) from e
            raise RuntimeError(
                f"Ollama HTTP error {e.code} at {self.host}/api/chat: {e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. "
                f"Is the daemon running? (`ollama serve`)  Underlying error: {e}"
            ) from e
        text = (body.get("message") or {}).get("content", "").strip()
        if not text:
            raise RuntimeError(f"Ollama returned empty response: {body}")
        return text

    def generate(
        self, topic: str, duration_seconds: int, model: str | None,
        article_notes: str = ""
    ) -> str:
        return self._chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_message(topic, duration_seconds, article_notes)},
            ],
            model,
            temperature=0.8,
        )

    def generate_with_nudge(
        self, topic: str, duration_seconds: int, model: str | None,
        attempt: int, article_notes: str = ""
    ) -> str:
        """Generate a fresh candidate with a varied angle nudge."""
        return self._chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _regen_user_message(topic, duration_seconds, attempt, article_notes)},
            ],
            model,
            temperature=0.9,  # slightly higher temp for more varied candidates
        )

    def chat(self, system: str, user: str, model: str | None) -> str:
        return self._chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model,
            temperature=0.4,
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible (optional)
# ---------------------------------------------------------------------------

class OpenAIProvider:
    """Any OpenAI-compatible endpoint. Reads OPENAI_API_KEY / OPENAI_BASE_URL."""

    def _client_and_model(self, model: str | None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. `pip install openai` or use --llm-provider ollama."
            ) from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL") or None)
        model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        return client, model

    def _chat(self, messages: list[dict], model: str | None, temperature: float) -> str:
        client, model = self._client_and_model(model)
        resp = client.chat.completions.create(
            model=model, temperature=temperature, messages=messages,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("OpenAI returned empty response")
        return text

    def generate(
        self, topic: str, duration_seconds: int, model: str | None,
        article_notes: str = ""
    ) -> str:
        return self._chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_message(topic, duration_seconds, article_notes)},
            ],
            model,
            temperature=0.8,
        )

    def generate_with_nudge(
        self, topic: str, duration_seconds: int, model: str | None,
        attempt: int, article_notes: str = ""
    ) -> str:
        """Generate a fresh candidate with a varied angle nudge."""
        return self._chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _regen_user_message(topic, duration_seconds, attempt, article_notes)},
            ],
            model,
            temperature=0.9,
        )

    def chat(self, system: str, user: str, model: str | None) -> str:
        return self._chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model,
            temperature=0.4,
        )


# ---------------------------------------------------------------------------
# File fallback
# ---------------------------------------------------------------------------

class FileProvider:
    """Reads the script straight from a text file. No LLM."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def generate(self, topic: str, duration_seconds: int, model: str | None) -> str:
        if not self.path.is_file():
            raise FileNotFoundError(f"Script file not found: {self.path}")
        text = self.path.read_text(encoding="utf-8").strip()
        if not text:
            raise RuntimeError(f"Script file is empty: {self.path}")
        return text

    # No chat() -> review loop is skipped for file-provided scripts.


# ---------------------------------------------------------------------------
# Self-review helpers
# ---------------------------------------------------------------------------

def _review(provider: Provider, topic: str, script_text: str, model: str | None) -> dict:
    chat = getattr(provider, "chat", None)
    if not callable(chat):
        return _heuristic_review(script_text)
    try:
        raw = chat(REVIEW_SYSTEM_PROMPT, _review_user_message(topic, script_text), model)
        parsed = _extract_json(raw)
        if not parsed or "score" not in parsed:
            return _heuristic_review(script_text)
        parsed["word_count"] = _word_count(script_text)

        # Hard length gate: word count outside 90-120 → cap score at 3
        wc = parsed["word_count"]
        if not (MIN_WORDS <= wc <= MAX_WORDS):
            parsed["score"] = min(int(parsed.get("score") or 3), 3)
            issues = list(parsed.get("issues") or [])
            length_msg = f"word count {wc} outside {MIN_WORDS}-{MAX_WORDS} (hard gate)"
            if length_msg not in issues:
                issues.insert(0, length_msg)
            parsed["issues"] = issues[:5]

        # If the reviewer detected invented facts, cap score at 3
        if parsed.get("has_invented"):
            parsed["score"] = min(int(parsed.get("score") or 3), 3)
            if "invented facts or names not present in topic" not in str(parsed.get("issues", [])):
                issues = list(parsed.get("issues") or [])
                issues.insert(0, "invented facts or names not present in topic")
                parsed["issues"] = issues[:5]
            parsed.setdefault("rewrite_notes",
                "Remove ALL invented product names, company names, and specific "
                "details not given in the topic. Use only what the topic states.")
        return parsed
    except Exception:
        return _heuristic_review(script_text)


def _rewrite(
    provider: Provider,
    topic: str,
    duration_seconds: int,
    previous: str,
    review: dict,
    model: str | None,
) -> str:
    chat = getattr(provider, "chat", None)
    if not callable(chat):
        return previous
    return chat(
        REWRITE_SYSTEM_PROMPT,
        _rewrite_user_message(topic, duration_seconds, previous, review),
        model,
    )


def _regenerate(
    provider: Provider,
    topic: str,
    duration_seconds: int,
    model: str | None,
    attempt: int,
    article_notes: str = "",
) -> str:
    """Full fresh generation with a varied angle nudge."""
    gen_with_nudge = getattr(provider, "generate_with_nudge", None)
    if callable(gen_with_nudge):
        return gen_with_nudge(topic, duration_seconds, model, attempt, article_notes)
    # Fallback for providers that don't implement generate_with_nudge
    return provider.generate(topic, duration_seconds, model, article_notes)


def _score_int(review: dict) -> int:
    return int(review.get("score", 0) or 0)


def _format_sub_scores(review: dict) -> str:
    return (
        f"hook={review.get('hook_strength','?')} "
        f"curiosity={review.get('curiosity_gap','?')} "
        f"clarity={review.get('clarity','?')} "
        f"retention={review.get('retention_potential','?')} "
        f"novelty={review.get('novelty','?')} "
        f"emotion={review.get('emotional_impact','?')}"
    )


# ---------------------------------------------------------------------------
# Multi-candidate selection (no rewrite loop)
# ---------------------------------------------------------------------------

def _generate_candidates(
    provider: Provider,
    topic: str,
    duration_seconds: int,
    model: str | None,
    *,
    count: int = NUM_CANDIDATES,
    article_notes: str = "",
    verbose: bool = True,
) -> tuple[str, dict]:
    """Generate *count* independent candidate scripts, score each, return the best.

    Each candidate is generated independently with a different angle nudge.
    No rewriting — the best score wins outright.
    """
    grounded = " (grounded)" if article_notes else ""
    if verbose:
        print(f"  generating {count} independent candidate scripts{grounded}...")

    best_text: str | None = None
    best_review: dict | None = None

    for i in range(count):
        try:
            if i == 0:
                raw = provider.generate(topic, duration_seconds, model, article_notes)
            else:
                raw = _regenerate(provider, topic, duration_seconds, model, attempt=i,
                                  article_notes=article_notes)
            text = _clean_script(raw)
            review = _review(provider, topic, text, model)
            invented_flag = " [INVENTED FACTS]" if review.get("has_invented") else ""
            length_flag = " [SHORT]" if review.get("word_count", 0) < MIN_WORDS else ""
            if verbose:
                print(
                    f"  candidate {i+1}/{count}: score={review.get('score')}/10 "
                    f"words={review.get('word_count')} "
                    f"[{_format_sub_scores(review)}]{invented_flag}{length_flag}"
                )
            if best_review is None or _score_int(review) > _score_int(best_review):
                best_text, best_review = text, review
        except Exception as e:
            if verbose:
                print(f"  candidate {i+1}/{count}: generation failed ({e}); skipping")

    if best_text is None:
        raise RuntimeError("All candidate generations failed.")

    if verbose:
        print(
            f"  best candidate: score={best_review.get('score')}/10 "
            f"[{_format_sub_scores(best_review)}]"
        )
    return best_text, best_review


# ---------------------------------------------------------------------------
# Refine loop (rewrite-or-regen, no repetitive loops)
# ---------------------------------------------------------------------------

def _refine(
    provider: Provider,
    topic: str,
    duration_seconds: int,
    initial_text: str,
    initial_review: dict,
    model: str | None,
    *,
    verbose: bool = True,
) -> str:
    """Improve the best candidate if it scores below TARGET_SCORE.

    Strategy per attempt:
      1. Try a rewrite (guided by the editor's notes).
      2. If the rewrite does NOT improve the score → do a full regeneration
         with a fresh angle nudge instead of another futile rewrite.
    """
    best_text = initial_text
    best_review = initial_review

    for attempt in range(1, MAX_REFINE_ATTEMPTS + 1):
        if _score_int(best_review) >= TARGET_SCORE:
            break

        if verbose:
            print(
                f"  score {_score_int(best_review)}/10 < {TARGET_SCORE} "
                f"→ refine attempt {attempt}/{MAX_REFINE_ATTEMPTS} (rewrite)"
            )

        # --- Rewrite attempt ---
        try:
            rw_raw = _rewrite(provider, topic, duration_seconds, best_text, best_review, model)
            rw_text = _clean_script(rw_raw)
            rw_review = _review(provider, topic, rw_text, model)
            invented_flag = " [INVENTED FACTS]" if rw_review.get("has_invented") else ""
            if verbose:
                print(
                    f"  rewrite: score={rw_review.get('score')}/10 "
                    f"words={rw_review.get('word_count')} "
                    f"[{_format_sub_scores(rw_review)}]{invented_flag}"
                )
        except Exception as e:
            if verbose:
                print(f"  rewrite failed ({e}); skipping to regen")
            rw_review = {"score": 0}
            rw_text = best_text

        rewrite_improved = _score_int(rw_review) > _score_int(best_review)

        if rewrite_improved:
            best_text, best_review = rw_text, rw_review
            if verbose:
                print(f"  rewrite improved score → keeping rewrite")
        else:
            # Rewrite stalled → full regeneration with fresh angle
            if verbose:
                print(
                    f"  rewrite did NOT improve score "
                    f"({_score_int(rw_review)}/10 vs {_score_int(best_review)}/10) "
                    f"→ switching to full regeneration"
                )
            try:
                regen_raw = _regenerate(provider, topic, duration_seconds, model, attempt=attempt)
                regen_text = _clean_script(regen_raw)
                regen_review = _review(provider, topic, regen_text, model)
                invented_flag = " [INVENTED FACTS]" if regen_review.get("has_invented") else ""
                if verbose:
                    print(
                        f"  regen: score={regen_review.get('score')}/10 "
                        f"words={regen_review.get('word_count')} "
                        f"[{_format_sub_scores(regen_review)}]{invented_flag}"
                    )
                if _score_int(regen_review) > _score_int(best_review):
                    best_text, best_review = regen_text, regen_review
                    if verbose:
                        print(f"  regen improved score → keeping regen")
                else:
                    if verbose:
                        print(f"  regen also did not improve; keeping previous best")
            except Exception as e:
                if verbose:
                    print(f"  regen failed ({e}); keeping previous best")

    if verbose:
        print(
            f"  final: score={best_review.get('score')}/10 "
            f"words={best_review.get('word_count')} "
            f"[{_format_sub_scores(best_review)}]"
        )
    return best_text


# ---------------------------------------------------------------------------
# Factory + public API
# ---------------------------------------------------------------------------

PROVIDERS = ("ollama", "openai", "file")


def get_provider(name: str, *, script_file: str | None = None) -> Provider:
    name = (name or "ollama").lower()
    if name == "ollama":
        return OllamaProvider()
    if name == "openai":
        return OpenAIProvider()
    if name == "file":
        if not script_file:
            raise ValueError("--llm-provider file requires --script-file PATH")
        return FileProvider(script_file)
    raise ValueError(f"Unknown provider '{name}'. Choose one of: {', '.join(PROVIDERS)}")


def generate_script(
    topic: str,
    duration_seconds: int = 40,
    *,
    provider: str = "ollama",
    model: str | None = None,
    script_file: str | None = None,
    article_notes: str = "",
    review: bool = True,
    verbose: bool = True,
) -> str:
    """Generate a Shorts narration with 5 independent candidates, best wins.

    Pipeline (for ollama / openai providers):
      1. Generate NUM_CANDIDATES (5) scripts independently with different angle nudges.
         When article_notes are provided, every candidate is grounded to those facts.
      2. Score all candidates independently (hard length gate: < 90 words → score 3).
      3. Select the highest-scoring candidate.
      4. Return it — no rewriting.

    The file provider bypasses the review loop (user-supplied text is final).
    """
    prov = get_provider(provider, script_file=script_file)

    # FileProvider: return as-is, no review loop
    if not review or isinstance(prov, FileProvider):
        raw = prov.generate(topic, duration_seconds, model)
        return _clean_script(raw)

    # Generate 5 independent candidates, score each, pick the best
    best_text, _best_review = _generate_candidates(
        prov, topic, duration_seconds, model,
        count=NUM_CANDIDATES,
        article_notes=article_notes,
        verbose=verbose,
    )

    return best_text
