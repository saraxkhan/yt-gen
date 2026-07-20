"""benchmark_models.py — Rigorous controlled LLM benchmark for yt-gen.

Compares multiple Ollama models against a FROZEN dataset of cached articles,
using the identical pipeline: same prompts, grounding, quality gates, scoring.

The ONLY changing variable is the LLM.

Benchmark phases:
  Phase 0  — Prepare frozen dataset (once): article text + pre-computed notes
  Phase 1  — Per-model testing: 5 sequential attempts per (model, story)
  Phase 2  — Report: 9-column comparison table + ranked recommendation

Metrics per model:
  Avg Script Score        — mean structure_score across all valid runs
  Pass Rate               — % of runs that pass all 3 hard gates
  First-Pass Success Rate — % of attempt-1 scripts that pass (no regen needed)
  Avg Regen Rounds        — mean attempts until first pass (or 5 if none pass)
  Fact Safety Rate        — % of runs where fact_safe = True
  Hallucination Rate      — 100 - fact_safety_rate
  Hook Score              — mean hook_score
  Word Count Compliance   — % of runs with word count in 90-120
  Avg Generation Time     — mean wall-clock time per attempt (s)

Usage:
    py benchmark_models.py
    py benchmark_models.py --models qwen2.5 qwen3:8b --attempts 5
    py benchmark_models.py --no-pull  (skip models not installed)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

# ── UTF-8 stdout for Windows terminals ───────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from shorts_gen import script as script_mod
from shorts_gen import quality as quality_mod
from shorts_gen.news import _extract_top_paragraphs  # noqa: SLF001

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CACHE_DIR      = PROJECT_ROOT / "output" / "_articles_cache"
DATASET_FILE   = PROJECT_ROOT / "benchmark_dataset.json"
RESULTS_FILE   = PROJECT_ROOT / "benchmark_results.json"

# These are the cached articles from real --auto runs (fair, representative)
ARTICLE_FIXTURES: list[dict] = [
    {
        "title": "From Materials Simulation to Experimental Astronomy, New NVIDIA AI Software Unlocks Scientific Discoveries",
        "url":   "https://blogs.nvidia.com/blog/ai-for-science-software-cuda/",
        "cache": "a4f16ee0c4efb1365da47f38b2dc4095.txt",
    },
    {
        "title": "How ChatGPT adoption has expanded",
        "url":   "https://openai.com/index/how-chatgpt-adoption-has-expanded",
        "cache": "068c9bc43551fee5154c8fa53e2673ad.txt",
    },
    {
        "title": "Core dump epidemiology: fixing an 18-year-old bug",
        "url":   "https://openai.com/index/core-dump-epidemiology-data-infrastructure-bug",
        "cache": "f1f7d4f8001b4288ac8c42bb6a97b813.txt",
    },
]

DEFAULT_MODELS  = ["qwen2.5", "qwen3:8b", "llama3.1:8b", "gemma3:12b"]
DEFAULT_ATTEMPTS = 5   # attempts per (model, story)
MAX_REGEN_ROUNDS = 5   # mirrors quality_mod.SCRIPT_MAX_REGEN_ROUNDS

GATE_MIN_STRUCTURE = quality_mod.SCRIPT_MIN_STRUCTURE_SCORE   # = 8


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AttemptResult:
    """One round of 5-candidate script generation for a (model, story, attempt#)."""
    model:              str
    story_title:        str
    attempt_num:        int      # 1-based index within the current (model, story) run
    structure_score:    int
    hook_score:         int
    word_count:         int
    estimated_seconds:  float
    fact_safe:          bool
    invented_items:     list[str]
    has_curiosity_gap:  bool
    has_payoff:         bool
    has_why_it_matters: bool
    has_cta:            bool
    passes_gate:        bool
    generation_time_s:  float
    script_text:        str = ""
    error:              str = ""


@dataclass
class StoryRun:
    """All 5 sequential attempts for one (model, story) pair.

    This mirrors what _generate_script_with_retry() does in production:
    try up to MAX_REGEN_ROUNDS rounds, stop at first pass.
    """
    model:       str
    story_title: str
    attempts:    list[AttemptResult] = field(default_factory=list)

    @property
    def first_pass_attempt(self) -> int | None:
        """1-based index of the first passing attempt, or None."""
        for a in self.attempts:
            if a.passes_gate:
                return a.attempt_num
        return None

    @property
    def passed(self) -> bool:
        return self.first_pass_attempt is not None

    @property
    def first_pass_success(self) -> bool:
        """True if attempt 1 passed (no regeneration needed)."""
        return bool(self.attempts) and self.attempts[0].passes_gate

    @property
    def regen_rounds_used(self) -> int:
        """Regeneration rounds required before script passes (0 if attempt 1 passed)."""
        fp = self.first_pass_attempt
        return (fp - 1) if fp is not None else (len(self.attempts) - 1)

    def best_attempt(self) -> AttemptResult | None:
        """The passing attempt, or the highest-scoring attempt if none pass."""
        passing = [a for a in self.attempts if a.passes_gate]
        if passing:
            return passing[0]
        valid = [a for a in self.attempts if not a.error]
        return max(valid, key=lambda a: a.structure_score) if valid else None


@dataclass
class ModelSummary:
    model:      str
    story_runs: list[StoryRun] = field(default_factory=list)

    # ── Convenience helpers ───────────────────────────────────────────────────

    @property
    def all_attempts(self) -> list[AttemptResult]:
        return [a for sr in self.story_runs for a in sr.attempts]

    @property
    def valid_attempts(self) -> list[AttemptResult]:
        return [a for a in self.all_attempts if not a.error]

    # ── Aggregated metrics ────────────────────────────────────────────────────

    @property
    def avg_structure_score(self) -> float:
        v = self.valid_attempts
        return mean(a.structure_score for a in v) if v else 0.0

    @property
    def avg_hook_score(self) -> float:
        v = self.valid_attempts
        return mean(a.hook_score for a in v) if v else 0.0

    @property
    def avg_word_count(self) -> float:
        v = self.valid_attempts
        return mean(a.word_count for a in v) if v else 0.0

    @property
    def avg_time(self) -> float:
        v = self.valid_attempts
        return mean(a.generation_time_s for a in v) if v else 0.0

    @property
    def pass_rate(self) -> float:
        """% of story-runs that eventually produced a passing script."""
        if not self.story_runs:
            return 0.0
        return sum(1 for sr in self.story_runs if sr.passed) / len(self.story_runs) * 100

    @property
    def first_pass_success_rate(self) -> float:
        """% of story-runs where attempt #1 passed immediately."""
        if not self.story_runs:
            return 0.0
        return sum(1 for sr in self.story_runs if sr.first_pass_success) / len(self.story_runs) * 100

    @property
    def avg_regen_rounds(self) -> float:
        """Mean number of rounds until first pass (or MAX_REGEN_ROUNDS if never)."""
        if not self.story_runs:
            return 0.0
        return mean(sr.regen_rounds_used for sr in self.story_runs)

    @property
    def fact_safe_rate(self) -> float:
        v = self.valid_attempts
        if not v:
            return 0.0
        return sum(1 for a in v if a.fact_safe) / len(v) * 100

    @property
    def hallucination_rate(self) -> float:
        return 100.0 - self.fact_safe_rate

    @property
    def word_count_compliance(self) -> float:
        v = self.valid_attempts
        if not v:
            return 0.0
        return sum(1 for a in v if 90 <= a.word_count <= 120) / len(v) * 100

    @property
    def curiosity_gap_rate(self) -> float:
        v = self.valid_attempts
        if not v:
            return 0.0
        return sum(1 for a in v if a.has_curiosity_gap) / len(v) * 100

    @property
    def total_attempts(self) -> int:
        return len(self.all_attempts)

    @property
    def error_rate(self) -> float:
        t = self.total_attempts
        if not t:
            return 0.0
        return sum(1 for a in self.all_attempts if a.error) / t * 100


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 — Frozen dataset preparation
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic_notes(title: str, article_text: str) -> str:
    """Build grounding notes without any LLM (fully deterministic)."""
    top_paragraphs = _extract_top_paragraphs(article_text)
    excerpt = article_text[:600].strip()
    return (
        f"ARTICLE EXCERPT (verbatim — use product/tool names exactly as written):\n{excerpt}\n\n"
        f"TOP ARTICLE PARAGRAPHS:\n{top_paragraphs}"
    )


def prepare_dataset(use_llm_notes: bool = True) -> list[dict]:
    """Load cached articles and build frozen notes.  Save to benchmark_dataset.json.

    Notes are generated ONCE using a heuristic (no LLM) so they are identical
    for all models.  This ensures the ONLY variable is the script generation LLM.

    If benchmark_dataset.json already exists it is reused without regeneration.
    """
    if DATASET_FILE.exists():
        print(f"[bench] Reusing frozen dataset: {DATASET_FILE}")
        return json.loads(DATASET_FILE.read_text(encoding="utf-8"))

    print("[bench] Preparing frozen benchmark dataset…")
    dataset: list[dict] = []

    for fixture in ARTICLE_FIXTURES:
        path = CACHE_DIR / fixture["cache"]
        if not path.exists():
            print(f"  [bench] ⚠  Cache file missing: {fixture['cache']} — skipping story.")
            continue
        article_text = path.read_text(encoding="utf-8", errors="replace")
        notes = _heuristic_notes(fixture["title"], article_text)
        entry = {
            "title":        fixture["title"],
            "url":          fixture["url"],
            "article_text": article_text,
            "article_notes": notes,
            "chars":        len(article_text),
            "notes_words":  len(notes.split()),
        }
        dataset.append(entry)
        print(
            f"  ✓  {fixture['title'][:65]}\n"
            f"     {len(article_text):,} chars  |  {len(notes.split())} note words"
        )

    if not dataset:
        # Fallback: any cache files present
        for p in sorted(CACHE_DIR.glob("*.txt"))[:3]:
            text = p.read_text(encoding="utf-8", errors="replace")
            title = text[:80].split("|")[0].strip() or p.stem
            notes = _heuristic_notes(title, text)
            dataset.append({
                "title":        title,
                "url":          f"https://example.com/{p.stem}",
                "article_text": text,
                "article_notes": notes,
                "chars":        len(text),
                "notes_words":  len(notes.split()),
            })
            print(f"  ✓  (fallback) {title[:65]}")

    DATASET_FILE.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[bench] Dataset frozen → {DATASET_FILE}\n")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Per-model benchmarking
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_attempt(
    *,
    model:         str,
    provider:      script_mod.OllamaProvider,
    story:         dict,
    attempt_num:   int,
    verbose:       bool = False,
) -> AttemptResult:
    """One round of 5-candidate script generation + full pipeline validation.

    Uses the FROZEN article_text and article_notes from the dataset — identical
    inputs for every model.  No LLM is called for grounding during this step.
    """
    title        = story["title"]
    article_text = story["article_text"]
    article_notes = story["article_notes"]   # pre-frozen, heuristic

    t0 = time.time()
    try:
        # Script generation — the only LLM-variable step
        raw_script = script_mod.generate_script(
            title,
            40,                  # 40s target (same as production default)
            provider="ollama",
            model=model,
            article_notes=article_notes,
            verbose=verbose,
        )
        gen_time = time.time() - t0

        # Validation — uses same provider (for structural review)
        validation = quality_mod.validate_script(
            raw_script, title,
            provider=provider,
            model=model,
            article_text=article_text,
        )
        # Fact-safety check — uses full article text (no title-only false positives)
        fact_safe, invented = quality_mod.fact_safety_check(
            raw_script, title,
            provider=provider,
            model=model,
            article_text=article_text,
        )

        passes = (
            validation.passes
            and validation.structure_score >= GATE_MIN_STRUCTURE
            and fact_safe
        )

        return AttemptResult(
            model=model,
            story_title=title,
            attempt_num=attempt_num,
            structure_score=validation.structure_score,
            hook_score=validation.hook_score,
            word_count=validation.word_count,
            estimated_seconds=validation.estimated_seconds,
            fact_safe=fact_safe,
            invented_items=invented,
            has_curiosity_gap=validation.has_curiosity_gap,
            has_payoff=validation.has_payoff,
            has_why_it_matters=validation.has_why_it_matters,
            has_cta=validation.has_cta,
            passes_gate=passes,
            generation_time_s=gen_time,
            script_text=raw_script,
        )

    except Exception as exc:
        return AttemptResult(
            model=model, story_title=title, attempt_num=attempt_num,
            structure_score=0, hook_score=0, word_count=0,
            estimated_seconds=0, fact_safe=False, invented_items=[],
            has_curiosity_gap=False, has_payoff=False,
            has_why_it_matters=False, has_cta=False,
            passes_gate=False, generation_time_s=time.time() - t0,
            error=str(exc),
        )


def benchmark_model(
    model:    str,
    dataset:  list[dict],
    attempts: int,
    verbose:  bool = False,
) -> ModelSummary:
    """Run the full benchmark for one model.

    For each story, runs up to `attempts` sequential rounds.
    Stops early for that story as soon as a round passes all gates
    (mirrors production _generate_script_with_retry behaviour).
    """
    summary = ModelSummary(model=model)
    total_story_runs = len(dataset) * attempts   # upper bound
    run_num = 0

    for story in dataset:
        story_run = StoryRun(model=model, story_title=story["title"])
        story_short = story["title"][:55]

        print(f"\n  Story: {story_short}…")

        for attempt_num in range(1, attempts + 1):
            run_num += 1
            pct = run_num / total_story_runs * 100
            print(f"    Attempt {attempt_num}/{attempts} ", end="", flush=True)

            result = _run_one_attempt(
                model=model,
                provider=script_mod.OllamaProvider(),
                story=story,
                attempt_num=attempt_num,
                verbose=verbose,
            )
            story_run.attempts.append(result)

            if result.error:
                print(f"❌ ERROR: {result.error[:60]}")
                continue

            pi = "✅ PASS" if result.passes_gate else "❌ FAIL"
            fi = "safe" if result.fact_safe else f"UNSAFE [{', '.join(result.invented_items[:1])}]"
            gi = "✓gap" if result.has_curiosity_gap else "✗gap"
            print(
                f"{pi}  str={result.structure_score}/10  hook={result.hook_score}/10  "
                f"wds={result.word_count}  fact={fi}  {gi}  {result.generation_time_s:.0f}s"
            )

            # Mirror production behaviour: stop this story once gate passes
            if result.passes_gate:
                remaining = attempts - attempt_num
                if remaining > 0:
                    print(f"    → Gate passed on attempt {attempt_num}. "
                          f"({remaining} remaining attempt(s) skipped — gate already met)")
                break

        if story_run.first_pass_attempt is None:
            print(f"    → All {len(story_run.attempts)} attempt(s) failed for this story.")

        summary.story_runs.append(story_run)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Ollama availability helpers
# ─────────────────────────────────────────────────────────────────────────────

def check_model_available(model: str) -> bool:
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        installed = [m.get("name", "") for m in data.get("models", [])]
        base = model.split(":")[0]
        tag  = model.split(":")[1] if ":" in model else "latest"
        full = f"{base}:{tag}"
        return any(m == full or m == model or m.startswith(f"{base}:") for m in installed)
    except Exception:
        return False


def pull_model(model: str) -> bool:
    import subprocess
    print(f"  [bench] Pulling {model} (may take several minutes)…")
    try:
        result = subprocess.run(["ollama", "pull", model], timeout=1800)
        return result.returncode == 0
    except Exception as e:
        print(f"  [bench] Pull failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Reporting
# ─────────────────────────────────────────────────────────────────────────────

W = 120   # table width

def _hdr(title: str) -> str:
    return f"\n{'═'*W}\n{title.center(W)}\n{'═'*W}"


def print_per_story_detail(summary: ModelSummary) -> None:
    """Print per-story attempt detail for one model."""
    for sr in summary.story_runs:
        story_short = sr.story_title[:50]
        status = f"✅ passed on attempt {sr.first_pass_attempt}" if sr.passed else "❌ never passed"
        print(f"\n  [{story_short}]  →  {status}  ({sr.regen_rounds_used} round(s) used)")
        print(f"  {'Att':>3} {'Str':>4} {'Hook':>4} {'Wds':>4} {'Fact':>10} {'CurGap':>6} {'Pass':>6} {'Time':>6}")
        print(f"  {'─'*3} {'─'*4} {'─'*4} {'─'*4} {'─'*10} {'─'*6} {'─'*6} {'─'*6}")
        for a in sr.attempts:
            if a.error:
                print(f"  #{a.attempt_num:>2}  ERROR: {a.error[:70]}")
                continue
            fi  = "safe     " if a.fact_safe else f"UNSAFE   "
            inv = f"[{a.invented_items[0][:20]}]" if a.invented_items else ""
            gi  = "✓" if a.has_curiosity_gap else "✗"
            pi  = "✅" if a.passes_gate else "❌"
            print(
                f"  #{a.attempt_num:>2} "
                f"{a.structure_score:>4} {a.hook_score:>4} {a.word_count:>4} "
                f"  {fi}{inv[:14]:<14}  {gi:>4}   {pi}  {a.generation_time_s:>5.0f}s"
            )


def print_comparison_table(summaries: list[ModelSummary]) -> None:
    print(_hdr("BENCHMARK COMPARISON — FINAL SUMMARY TABLE"))

    # Row format (exactly 10 columns matching user's request)
    col = (
        f"  {'Model':<18} | "
        f"{'Avg Script Score':<16} | "
        f"{'Pass Rate':<9} | "
        f"{'First-Pass Success':<18} | "
        f"{'Avg Regen Rounds':<16} | "
        f"{'Fact Safety':<11} | "
        f"{'Hallucination Rate':<18} | "
        f"{'Hook Score':<10} | "
        f"{'Word Count Compliance':<21} | "
        f"{'Avg Generation Time':<19}"
    )
    sep = (
        f"  {'─'*18}─┼─"
        f"{'─'*16}─┼─"
        f"{'─'*9}─┼─"
        f"{'─'*18}─┼─"
        f"{'─'*16}─┼─"
        f"{'─'*11}─┼─"
        f"{'─'*18}─┼─"
        f"{'─'*10}─┼─"
        f"{'─'*21}─┼─"
        f"{'─'*19}"
    )
    print(f"\n{col}\n{sep}")

    for s in summaries:
        if not s.valid_attempts:
            print(f"  {s.model:<18} | {'— ALL RUNS FAILED —':^135}")
            continue
        print(
            f"  {s.model:<18} | "
            f"{f'{s.avg_structure_score:.1f}/10':>16} | "
            f"{f'{s.pass_rate:.0f}%':>9} | "
            f"{f'{s.first_pass_success_rate:.0f}%':>18} | "
            f"{f'{s.avg_regen_rounds:.2f}':>16} | "
            f"{f'{s.fact_safe_rate:.0f}%':>11} | "
            f"{f'{s.hallucination_rate:.0f}%':>18} | "
            f"{f'{s.avg_hook_score:.1f}/10':>10} | "
            f"{f'{s.word_count_compliance:.0f}%':>21} | "
            f"{f'{s.avg_time:.0f}s':>19}"
        )

    print(f"\n{'─'*180}")
    print(
        f"  Gate: structure ≥ {GATE_MIN_STRUCTURE}/10  AND  fact_safe = True  "
        f"AND  word count 90–120  AND  all 5 structural sections present"
    )
    print(
        f"  Avg Regen Rounds: mean regeneration rounds required before script passes (lower = better; 0 = always first-pass success, {MAX_REGEN_ROUNDS - 1} = never passed)\n"
        f"  Word Count Compliance: % of scripts in 90–120 word target range\n"
        f"  First-Pass Success: % of stories where attempt #1 immediately passed (no regeneration needed)"
    )
    print(f"{'─'*180}\n")


def make_recommendation(summaries: list[ModelSummary]) -> str:
    """Rank and recommend.

    Priority weights (per user specification):
      1. Fact Safety       30%   (publishability is paramount)
      2. Pass Rate         25%   (overall quality gate pass)
      3. Structure Score   20%   (script quality)
      4. Hook Score        10%   (first-impression quality)
      5. Word Compliance   10%   (broadcast-ready length)
      6. Speed             5%    (secondary)
    """
    scored: list[tuple[float, ModelSummary]] = []
    max_time = max((s.avg_time for s in summaries if s.valid_attempts), default=1)

    for s in summaries:
        if not s.valid_attempts:
            continue
        speed_score = 1.0 - (s.avg_time / max_time) if max_time > 0 else 0
        composite = (
            0.30 * s.fact_safe_rate / 100
            + 0.25 * s.pass_rate / 100
            + 0.20 * s.avg_structure_score / 10
            + 0.10 * s.avg_hook_score / 10
            + 0.10 * s.word_count_compliance / 100
            + 0.05 * speed_score
        )
        scored.append((composite, s))

    scored.sort(reverse=True, key=lambda x: x[0])

    lines = [
        _hdr("RECOMMENDATION"),
        "",
        "  Priority weights: Fact Safety 30%  |  Pass Rate 25%  |  Structure 20%",
        "                    Hook 10%  |  Word Compliance 10%  |  Speed 5%",
        "",
    ]

    if not scored:
        lines.append("  ❌ No models produced valid results.")
        return "\n".join(lines)

    best_score, best = scored[0]

    lines += [
        f"  ┌─ RECOMMENDED MODEL: {best.model} ─────────────────────────────",
        f"  │  Composite score   : {best_score:.3f}",
        f"  │  Pass rate         : {best.pass_rate:.0f}%",
        f"  │  First-pass rate   : {best.first_pass_success_rate:.0f}%",
        f"  │  Avg regen rounds  : {best.avg_regen_rounds:.2f}",
        f"  │  Fact-safe rate    : {best.fact_safe_rate:.0f}%",
        f"  │  Hallucination rate: {best.hallucination_rate:.0f}%",
        f"  │  Avg structure     : {best.avg_structure_score:.1f}/10",
        f"  │  Hook score        : {best.avg_hook_score:.1f}/10",
        f"  │  Word compliance   : {best.word_count_compliance:.0f}%",
        f"  │  Avg gen time      : {best.avg_time:.0f}s/attempt",
        f"  └────────────────────────────────────────────────────",
        "",
        "  Full ranking:",
    ]

    for rank, (sc, s) in enumerate(scored, 1):
        marker = " ←── BEST" if rank == 1 else ""
        lines.append(
            f"    {rank}. {s.model:<24}"
            f"  composite={sc:.3f}"
            f"  pass={s.pass_rate:.0f}%"
            f"  1st={s.first_pass_success_rate:.0f}%"
            f"  regen={s.avg_regen_rounds:.1f}"
            f"  fact={s.fact_safe_rate:.0f}%"
            f"  time={s.avg_time:.0f}s"
            f"{marker}"
        )

    lines += [
        "",
        f"  → Action: Set --model {best.model} as the default in your --auto runs." if best.pass_rate > 0
        else (
            "  → Action: No local model passed consistently.\n"
            "    Consider switching to a cloud LLM provider (OpenAI / Gemini API)\n"
            "    or pulling a larger quantization (e.g., q8_0) of the best-performing model."
        ),
        "",
    ]
    return "\n".join(lines)


def save_results(
    summaries:  list[ModelSummary],
    dataset:    list[dict],
    models:     list[str],
    attempts:   int,
) -> Path:
    raw: list[dict] = []
    for s in summaries:
        for sr in s.story_runs:
            for a in sr.attempts:
                raw.append({
                    "model":            a.model,
                    "story":            a.story_title,
                    "attempt_num":      a.attempt_num,
                    "structure_score":  a.structure_score,
                    "hook_score":       a.hook_score,
                    "word_count":       a.word_count,
                    "fact_safe":        a.fact_safe,
                    "invented_items":   a.invented_items,
                    "has_curiosity_gap": a.has_curiosity_gap,
                    "passes_gate":      a.passes_gate,
                    "generation_time_s": round(a.generation_time_s, 1),
                    "error":            a.error,
                })

    output = {
        "benchmark_time":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models":            models,
        "max_attempts_per_story": attempts,
        "stories":           [{"title": d["title"], "url": d["url"]} for d in dataset],
        "gate": {
            "min_structure_score": GATE_MIN_STRUCTURE,
            "word_range":          [90, 120],
            "fact_safe_required":  True,
            "all_sections_required": True,
        },
        "raw_attempts": raw,
        "summaries": [
            {
                "model":                      s.model,
                "total_attempts":             s.total_attempts,
                "valid_attempts":             len(s.valid_attempts),
                "pass_rate_pct":              round(s.pass_rate, 1),
                "first_pass_success_rate_pct": round(s.first_pass_success_rate, 1),
                "avg_regen_rounds":           round(s.avg_regen_rounds, 2),
                "fact_safe_rate_pct":         round(s.fact_safe_rate, 1),
                "hallucination_rate_pct":     round(s.hallucination_rate, 1),
                "avg_structure_score":        round(s.avg_structure_score, 2),
                "avg_hook_score":             round(s.avg_hook_score, 2),
                "avg_word_count":             round(s.avg_word_count, 1),
                "word_count_compliance_pct":  round(s.word_count_compliance, 1),
                "curiosity_gap_rate_pct":     round(s.curiosity_gap_rate, 1),
                "avg_time_s":                 round(s.avg_time, 1),
                "error_rate_pct":             round(s.error_rate, 1),
            }
            for s in summaries
        ],
    }

    RESULTS_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return RESULTS_FILE


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rigorous controlled LLM benchmark for yt-gen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models",   nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--attempts", type=int,  default=DEFAULT_ATTEMPTS,
                        help="Max attempts per model per story before giving up.")
    parser.add_argument("--no-pull",  action="store_true",
                        help="Skip models not yet installed instead of pulling them.")
    parser.add_argument("--verbose",  action="store_true",
                        help="Show verbose generation output per attempt.")
    parser.add_argument("--reset-dataset", action="store_true",
                        help="Delete and re-prepare the frozen dataset.")
    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────────────
    print(_hdr("YT-GEN RIGOROUS MODEL BENCHMARK"))
    print(f"  Models        : {', '.join(args.models)}")
    print(f"  Max attempts  : {args.attempts} per model per story (early exit on pass)")
    print(f"  Stories       : {len(ARTICLE_FIXTURES)} frozen articles")
    print(f"  Gate          : structure ≥ {GATE_MIN_STRUCTURE}/10  +  fact_safe  +  words 90–120")
    print(f"  Hardware      : CPU-only (16 GB RAM, Intel UHD 620 integrated GPU)")
    print(f"  Variable      : LLM model ONLY — all other inputs are frozen")
    print(f"{'═'*W}\n")

    # ── Phase 0: Prepare frozen dataset ──────────────────────────────────────
    if args.reset_dataset and DATASET_FILE.exists():
        DATASET_FILE.unlink()
        print("[bench] Frozen dataset reset.\n")

    dataset = prepare_dataset()
    if not dataset:
        print("❌ No articles available. Run `py -m shorts_gen.main --auto` to populate cache.")
        return

    print(f"\n[bench] Frozen dataset: {len(dataset)} stories")
    for i, d in enumerate(dataset, 1):
        print(f"  {i}. {d['title'][:70]}")
        print(f"     {d['chars']:,} chars  |  {d['notes_words']} note words  |  {d['url']}")
    print()

    # ── Phase 1: Per-model benchmarking ──────────────────────────────────────
    all_summaries: list[ModelSummary] = []
    benchmark_start = time.time()

    for model in args.models:
        print(f"\n{'━'*W}")
        print(f"  MODEL: {model}")
        print(f"{'━'*W}")

        # Check / pull
        available = check_model_available(model)
        if not available:
            if args.no_pull:
                print(f"  ⚠  {model} not installed and --no-pull set. Skipping.")
                s = ModelSummary(model=model)
                s.story_runs.append(StoryRun(model=model, story_title="N/A"))
                s.story_runs[0].attempts.append(AttemptResult(
                    model=model, story_title="N/A", attempt_num=1,
                    structure_score=0, hook_score=0, word_count=0,
                    estimated_seconds=0, fact_safe=False, invented_items=[],
                    has_curiosity_gap=False, has_payoff=False,
                    has_why_it_matters=False, has_cta=False,
                    passes_gate=False, generation_time_s=0,
                    error="Not installed (--no-pull)",
                ))
                all_summaries.append(s)
                continue
            ok = pull_model(model)
            if not ok:
                print(f"  ❌ Could not pull {model}. Skipping.")
                continue

        model_start = time.time()
        summary = benchmark_model(
            model=model,
            dataset=dataset,
            attempts=args.attempts,
            verbose=args.verbose,
        )
        model_elapsed = time.time() - model_start

        # Per-model detail
        print_per_story_detail(summary)

        # Per-model summary line
        print(
            f"\n  Model summary: pass={summary.pass_rate:.0f}%  "
            f"1st-pass={summary.first_pass_success_rate:.0f}%  "
            f"regen={summary.avg_regen_rounds:.2f}  "
            f"fact={summary.fact_safe_rate:.0f}%  "
            f"struct={summary.avg_structure_score:.1f}/10  "
            f"wds={summary.avg_word_count:.0f}  "
            f"elapsed={model_elapsed:.0f}s"
        )

        all_summaries.append(summary)

    total_elapsed = time.time() - benchmark_start

    # ── Phase 2: Reporting ────────────────────────────────────────────────────
    print_comparison_table(all_summaries)
    print(make_recommendation(all_summaries))

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = save_results(all_summaries, dataset, args.models, args.attempts)
    print(f"[bench] Full results saved → {out}")
    print(f"[bench] Total benchmark time: {total_elapsed/60:.1f} min\n")


if __name__ == "__main__":
    main()
