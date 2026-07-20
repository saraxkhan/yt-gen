"""CLI entry point: topic → finished YouTube Short.

Two visual modes:
  - image (default, recommended): generate topic-specific AI images from the
    script, apply Ken Burns zoom/pan + crossfades.
  - video: original background-footage mode (category-routed clips).

Output layout (created automatically):
  output/videos/      YYYY-MM-DD_HH-MM-SS_<slug>.mp4              ← final video
  output/videos/      YYYY-MM-DD_HH-MM-SS_<slug>_quality_report.json ← QA report
  output/audio/       YYYY-MM-DD_HH-MM-SS_<slug>.mp3              ← voiceover
  output/subtitles/   YYYY-MM-DD_HH-MM-SS_<slug>.ass              ← karaoke subs
  output/images/      YYYY-MM-DD_HH-MM-SS_<slug>/                 ← generated images
  output/short.mp4                                                 ← latest (copy)
  output/history.json                                              ← generation log
  output/_work/                                                    ← temp scratch
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure Unicode output (emoji, box-drawing chars) works on Windows terminals.
# This is a no-op when the stream is already UTF-8 (Linux/macOS) or when
# the stream is not a TTY (file redirect, pytest, etc.).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


from . import script as script_mod
from . import tts, subtitles, video, categorize
from . import image_prompts as img_prompts
from . import images as images_mod
from . import news as news_mod
from . import quality as quality_mod
from .output_paths import make_output_paths, write_latest_copy, print_summary



# ---------------------------------------------------------------------------
# Generation summary helper
# ---------------------------------------------------------------------------

def _print_generation_summary(
    *,
    article_extracted: bool,
    article_chars: int,
    article_notes_words: int,
    script_score: int,
    script_words: int,
    script_seconds: float,
    fact_safe: bool,
    visual_assets: str = "pending",
    total_time: float | None = None,
) -> None:
    """Print a concise generation summary block to stdout."""
    width = 56
    bar = "━" * width
    print(f"\n{bar}")
    print("  GENERATION SUMMARY".center(width))
    print(bar)

    # Article extraction
    if article_extracted:
        print(f"  Article extracted   │ ✓  ({article_chars:,} chars)")
        print(f"  Summary length      │ {article_notes_words} words")
    else:
        print(f"  Article extracted   │ ✗  (manual topic — no grounding)")

    # Script
    print(f"  Chosen script score │ {script_score}/10")
    print(f"  Final script length │ {script_words} words (~{script_seconds:.0f}s)")
    print(f"  Fact safety         │ {'✅ safe' if fact_safe else '⚠  check warnings below'}")

    # Visual + timing (filled in later if provided)
    if visual_assets != "pending":
        print(f"  Visual assets       │ {visual_assets}")
    if total_time is not None:
        print(f"  Total time          │ {total_time:.0f}s")

    print(bar + "\n")


# Minimum structure score for a script to be accepted for publishing.
# Must match quality_mod.SCRIPT_MIN_STRUCTURE_SCORE.
_SCRIPT_PASS_MIN_STRUCTURE: int = 8


def _script_passes_gate(
    validation: quality_mod.ScriptValidation,
    fact_safe: bool,
) -> bool:
    """Return True only when ALL mandatory publication gates pass.

    Gates (ALL must pass):
      ✓ validation.passes     — word count, duration, all 5 structural sections
      ✓ structure_score ≥ 8   — prevents low-quality scripts from shipping
      ✓ fact_safe             — no invented names / stats detected
    """
    return (
        validation.passes
        and validation.structure_score >= _SCRIPT_PASS_MIN_STRUCTURE
        and fact_safe
    )


def _attempt_script_round(
    topic: str,
    duration: int,
    provider: str,
    model: str | None,
    script_file: str | None,
    prov_obj,
    article_notes: str = "",
    article_text: str = "",
    round_num: int = 1,
    max_rounds: int = 5,
) -> tuple[str, quality_mod.ScriptValidation, bool, list[str]]:
    """Run a single script generation round and return (text, validation, fact_safe, invented).

    Does NOT retry on failure — that is the caller's responsibility.
    Raises on unrecoverable generation errors.
    """
    raw = script_mod.generate_script(
        topic,
        duration,
        provider=provider,
        model=model,
        script_file=script_file,
        article_notes=article_notes,
    )

    validation = quality_mod.validate_script(
        raw, topic,
        provider=prov_obj,
        model=model,
        article_text=article_text,
    )
    fact_safe, invented = quality_mod.fact_safety_check(
        raw, topic,
        provider=prov_obj,
        model=model,
        article_text=article_text,
    )
    return raw, validation, fact_safe, invented


def _generate_script_with_retry(
    topic: str,
    duration: int,
    provider: str,
    model: str | None,
    script_file: str | None,
    prov_obj,
    article_notes: str = "",
    article_text: str = "",
    max_rounds: int = 5,
) -> tuple[str, quality_mod.ScriptValidation, bool, list[str]] | None:
    """Try up to *max_rounds* independent script generation rounds.

    Each round generates NUM_CANDIDATES=5 independent candidates, scores them,
    and selects the best.  The result is then tested against ALL mandatory gates:
      ✓ word count 90–120
      ✓ duration 35–45s
      ✓ all 5 structural sections present
      ✓ structure_score ≥ 8
      ✓ fact_safe (no invented names/stats)

    If the gate passes: return (script_text, validation, fact_safe, invented).
    If ALL rounds fail: return None (caller must skip this story).

    Side-effects: prints progress to stdout.
    """
    best: tuple[str, quality_mod.ScriptValidation, bool, list[str]] | None = None
    best_score: int = -1

    for rnd in range(1, max_rounds + 1):
        print(f"  [script] Round {rnd}/{max_rounds}: generating 5 candidates…")
        try:
            result = _attempt_script_round(
                topic, duration, provider, model, script_file, prov_obj,
                article_notes=article_notes,
                article_text=article_text,
                round_num=rnd,
                max_rounds=max_rounds,
            )
        except Exception as exc:
            print(f"  [script] Round {rnd} generation error: {exc}")
            continue

        raw, validation, fact_safe, invented = result
        passes = _script_passes_gate(validation, fact_safe)

        # Gate diagnostic
        status = "✅ PASS" if passes else "❌ FAIL"
        invented_flag = f" [INVENTED: {', '.join(invented[:2])}]" if invented else ""
        print(
            f"  [script] Round {rnd} {status}: "
            f"structure={validation.structure_score}/10  "
            f"words={validation.word_count}  "
            f"fact={'safe' if fact_safe else 'UNSAFE'}"
            f"{invented_flag}"
        )
        if validation.issues:
            print(f"           issues: {'; '.join(validation.issues[:3])}")

        # Track best (highest structure score) regardless of gate
        if validation.structure_score > best_score:
            best = result
            best_score = validation.structure_score

        if passes:
            return result  # hard gate passed — done

        if rnd < max_rounds:
            print(f"  [script] Gate failed — regenerating (attempt {rnd + 1}/{max_rounds})…")

    print(f"  [script] All {max_rounds} rounds failed the quality gate for this story.")
    return None  # caller must skip this story


# ---------------------------------------------------------------------------
# Image mode helper (returns image_count + prompts_sanitized for report)
# ---------------------------------------------------------------------------

def _run_image_mode(
    args,
    script_text: str,
    paths,
    final_video_path: Path,
    provider: str,
    prov_obj,
) -> tuple[int, int]:
    """Run the image-mode pipeline and return (image_count, prompts_sanitized)."""
    import json as _json

    # 2a) Scene plan — drives durations, transitions, and cinematic prompts
    print("\n[2/5] Building cinematic scene plan…")
    plan = img_prompts.generate_scene_plan(
        args.topic,
        script_text,
        provider=provider,
        model=args.model,
        script_file=args.script_file,
        count=args.num_images,
        verbose=True,
    )
    print(
        f"  scene plan: {len(plan.scenes)} scenes, "
        f"visual quality={plan.quality_score}/10"
    )
    for s in plan.scenes:
        print(
            f"    Scene {s.index}: [{s.purpose}] "
            f"{s.duration:.1f}s  {s.emotion}  {s.camera_style}"
        )

    # Save scene plan JSON for debugging / quality reports
    plan_path = paths.work / "scene_plan.json"
    plan_path.write_text(
        _json.dumps(plan.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 2b) Cinematic prompts from scene plan
    prompts = img_prompts.scene_plan_to_prompts(
        plan,
        args.topic,
        provider=provider,
        model=args.model,
        script_file=args.script_file,
        verbose=True,
    )

    # 2c) Visual validation — rewrite any non-cinematic prompts
    prompts, sanitized_count = quality_mod.sanitize_visual_prompts(
        prompts,
        provider=prov_obj,
        model=args.model,
        verbose=True,
    )

    (paths.work / "image_prompts.txt").write_text("\n\n".join(prompts), encoding="utf-8")
    for i, p in enumerate(prompts):
        print(f"  {i + 1}. {p[:90]}{'…' if len(p) > 90 else ''}")

    vo_path, ass_path = _make_voice_and_subs(args, script_text, paths)

    # Extract scene metadata for video engine
    scene_durations   = [s.duration for s in plan.scenes]
    scene_transitions = [s.transition for s in plan.scenes[1:]]  # n-1 transitions

    # 5) Generate images + scene-aware Ken Burns composite
    print(
        f"\n[5/5] Generating {len(prompts)} images via "
        f"'{args.image_provider}' + cinematic compositing…"
    )
    image_paths = images_mod.generate_images(
        prompts,
        paths.images_dir,
        provider=args.image_provider,
        images_dir=args.images_dir,
        model=args.image_model,
    )
    video.compose_from_images(
        image_paths,
        vo_path,
        ass_path,
        final_video_path,
        scene_durations=scene_durations,
        scene_transitions=scene_transitions,
    )

    write_latest_copy(final_video_path, paths.latest)
    print_summary(paths, visual_mode="image")

    return len(image_paths), sanitized_count


# ---------------------------------------------------------------------------
# Video mode helper
# ---------------------------------------------------------------------------

def _run_video_mode(
    args,
    script_text: str,
    paths,
    final_video_path: Path,
) -> tuple[int, int]:
    """Run the video-mode pipeline and return (image_count=0, sanitized=0)."""
    print("\n[2/5] Selecting background category…")
    backgrounds_root = Path(args.backgrounds)
    cfg = categorize.CategoryConfig.load(args.categories_config)

    if args.category:
        category = args.category
        print(f"  category (forced): {category}")
    else:
        category, ranked = categorize.select_category(
            args.topic, script_text, backgrounds_root, cfg
        )
        top = [f"{n}={s:.1f}({h})" for n, s, h in ranked[:5] if h > 0] or [
            "<no keyword hits>"
        ]
        print(f"  scores: {', '.join(top)}")
        print(f"  category (auto):   {category}")

    vo_path, ass_path = _make_voice_and_subs(args, script_text, paths)

    print("\n[5/5] Compositing with FFmpeg…")
    bg = video.pick_background(backgrounds_root / category)
    print(f"  background: {category}/{bg.name}")
    video.compose(bg, vo_path, ass_path, final_video_path)

    write_latest_copy(final_video_path, paths.latest)
    print_summary(paths, visual_mode="video")

    return 0, 0


# ---------------------------------------------------------------------------
# Voice + subtitles (shared)
# ---------------------------------------------------------------------------

def _make_voice_and_subs(args, script_text: str, paths):
    """Synthesise voiceover and build karaoke subtitles.

    Assets are written directly to their permanent timestamped locations
    (``output/audio/`` and ``output/subtitles/``), not to ``_work/``.
    """
    print("\n[3/5] Synthesizing voiceover (edge-tts)…")
    vo_path = tts.synthesize(script_text, paths.audio, voice=args.voice)

    print("[4/5] Transcribing for word timings (faster-whisper)…")
    words = subtitles.transcribe_words(vo_path, model_size=args.whisper_model)
    ass_path = subtitles.build_ass(words, paths.subtitles)
    return vo_path, ass_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a YouTube Short from a topic.\n\n"
            "Examples:\n"
            "  py -m shorts_gen.main \"OpenAI launches GPT-5\"  # manual topic\n"
            "  py -m shorts_gen.main --auto                    # auto-pick from today's AI news"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="Topic / idea for the Short. Omit when using --auto.",
    )

    # --- Auto news mode ---
    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Fetch today's top AI news, apply quality gate, audience-score "
            "the top candidates, then generate a Short for the best story."
        ),
    )
    parser.add_argument(
        "--news-top-n",
        type=int,
        default=10,
        metavar="N",
        help="How many stories to show in the news leaderboard (default: 10).",
    )

    # --- Output ---
    parser.add_argument(
        "--out-root",
        default="output",
        help="Root output directory (default: output/).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Exact output MP4 path (overrides --out-root). Legacy flag.",
    )

    # --- Visual mode ---
    parser.add_argument(
        "--visual-mode",
        choices=("image", "video"),
        default="image",
        help="image (default, AI images + Ken Burns) | video (background footage)",
    )
    parser.add_argument("--backgrounds", default="backgrounds")
    parser.add_argument("--category", default=None)
    parser.add_argument("--categories-config", default=None)

    # --- Image mode ---
    parser.add_argument(
        "--image-provider",
        choices=images_mod.IMAGE_PROVIDERS,
        default="file",
        help=(
            "Image backend (default: file):\n"
            "  file         – images from --images-dir or backgrounds/\n"
            "  placeholder  – colour-gradient PNGs, no files needed\n"
            "  screenshot   – headless browser (needs playwright)"
        ),
    )
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--image-model", default=None)
    parser.add_argument(
        "--num-images",
        type=int,
        default=img_prompts.MAX_IMAGES,
        help="Number of images per Short (4–6)",
    )

    # --- Shared ---
    parser.add_argument("--voice", default=tts.DEFAULT_VOICE)
    parser.add_argument(
        "--provider", "--llm-provider",
        dest="llm_provider",
        default="ollama",
        choices=script_mod.PROVIDERS,
        help="Script generator: ollama (default) | openai | file",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name. Ollama default: qwen2.5",
    )
    parser.add_argument("--duration", type=int, default=40)
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument(
        "--script-file",
        default=None,
        help="Path to a .txt script file. Implies --llm-provider file.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate: need exactly one of topic or --auto
    # ------------------------------------------------------------------
    if args.auto and args.topic:
        parser.error("Provide either a manual topic OR --auto, not both.")
    if not args.auto and not args.topic:
        parser.error("A topic is required unless you use --auto.")

    provider = args.llm_provider
    if args.script_file and provider == "ollama":
        provider = "file"

    # Resolve the LLM provider object once — used by quality checks throughout.
    # FileProvider has no chat(); quality functions handle that gracefully.
    try:
        prov_obj = script_mod.get_provider(provider, script_file=args.script_file)
    except Exception:
        prov_obj = None

    # Record pipeline start time for the generation report
    run_start = time.time()

    # State collected for the production report
    selected_story = None
    selected_audience: quality_mod.AudienceScores | None = None
    article_notes = ""
    article_chars = 0
    article_extracted = False
    article_notes_words = 0

    # Article cache directory lives under the output root
    cache_dir = Path(args.out_root) / "_articles_cache"

    # ------------------------------------------------------------------
    # --auto: fetch → quality gate → audience score → rank candidates
    # ------------------------------------------------------------------
    if args.auto:
        print("\n[auto] Fetching today's AI news…")
        stories = news_mod.fetch_and_score(
            provider=prov_obj,
            model=args.model,
            verbose=True,
        )
        news_mod.print_news_board(stories, top_n=args.news_top_n)

        # Story quality gate
        gated = [s for s in stories if quality_mod.story_passes_gate(s)[0]]
        if not gated:
            print("\nNo high-quality AI news found today.")
            print(
                "Requirements: score ≥ "
                f"{quality_mod.GATE_MIN_SCORE}, "
                f"published within {int(quality_mod.GATE_MAX_AGE_HOURS)}h, "
                f"category in {sorted(quality_mod.GATE_VALID_CATEGORIES)}."
            )
            print(
                "Tip: Try again later, or use a manual topic:\n"
                "  py -m shorts_gen.main \"your topic\""
            )
            return

        print(f"\n[auto] {len(gated)} stor{'y' if len(gated)==1 else 'ies'} passed the quality gate.")

        # Audience-score the top 3 candidates
        candidates = gated[:3]
        print(f"[auto] Audience-scoring {len(candidates)} candidate(s)…")
        audience_map: dict[str, quality_mod.AudienceScores] = {}
        for cand in candidates:
            aud = quality_mod.score_audience_appeal(
                cand, provider=prov_obj, model=args.model
            )
            audience_map[cand.title] = aud
            print(
                f"  {cand.title[:52]}… "
                f"viral={cand.score}  audience={aud.overall}/100"
            )

        # Rank by combined score
        ranked_candidates = sorted(
            candidates,
            key=lambda s: quality_mod.combined_story_score(
                s, audience_map.get(s.title)
            ),
            reverse=True,
        )

        # ------------------------------------------------------------------
        # Story → grounding → script gate loop
        # Try each ranked story in order.
        # For each story: extract article → summarise → attempt up to 5 script
        # rounds.  If all 5 rounds fail, skip to the next story.
        # If all stories are exhausted: exit cleanly.
        # ------------------------------------------------------------------
        print("\n[auto] Starting story → script pipeline…")

        winning_story   = None
        winning_script  = None
        winning_val     = None
        winning_safe    = None
        winning_inv: list[str] = []
        winning_aud     = None

        for rank_idx, cand in enumerate(ranked_candidates):
            combined = quality_mod.combined_story_score(cand, audience_map.get(cand.title))
            print(
                f"\n[auto] Story {rank_idx + 1}/{len(ranked_candidates)}: "
                f"{cand.title[:65]} (combined={combined:.0f}/100)"
            )

            # Step A: Ground the story (article extraction + summarisation)
            print("  [ground] Extracting and summarising article…")
            news_mod.ground_story(
                cand,
                provider=prov_obj,
                model=args.model,
                cache_dir=cache_dir,
                verbose=True,
            )
            if not cand.article_extracted:
                print("  [ground] Article extraction failed — skipping to next story.")
                continue

            print(
                f"  [ground] ✓ {cand.article_chars:,} chars extracted, "
                f"{len(cand.article_notes.split())} note words"
            )

            # Step B: Attempt script generation with mandatory quality gate
            print(
                f"  [script] Generating script (≤{quality_mod.SCRIPT_MAX_REGEN_ROUNDS} rounds, "
                f"gate: structure ≥ {quality_mod.SCRIPT_MIN_STRUCTURE_SCORE}/10 + fact-safe + word count 90-120)…"
            )
            result = _generate_script_with_retry(
                cand.title,
                args.duration,
                provider,
                args.model,
                args.script_file,
                prov_obj,
                article_notes=cand.article_notes,
                article_text=cand.article_text,
                max_rounds=quality_mod.SCRIPT_MAX_REGEN_ROUNDS,
            )

            if result is None:
                print(
                    f"  [auto] Story {rank_idx + 1} could not produce a publishable script. "
                    f"Skipping to next story."
                )
                continue

            # Gate passed — lock in this story
            winning_story  = cand
            winning_aud    = audience_map.get(cand.title)
            winning_script, winning_val, winning_safe, winning_inv = result
            break

        # If no story produced a passing script, exit cleanly
        if winning_story is None:
            print(
                "\n❌ No publishable AI story found today."
                "\n   All candidate stories failed the script quality gate after "
                f"{quality_mod.SCRIPT_MAX_REGEN_ROUNDS} rounds each."
                "\n   Try again later, or use a manual topic: py -m shorts_gen.main \"topic\""
            )
            return

        # Unpack winning results into the variables the rest of main() uses
        selected_story    = winning_story
        selected_audience = winning_aud
        args.topic        = selected_story.title
        article_notes     = selected_story.article_notes
        article_chars     = selected_story.article_chars
        article_extracted = selected_story.article_extracted
        article_notes_words = len(article_notes.split()) if article_notes else 0

        combined = quality_mod.combined_story_score(selected_story, selected_audience)
        print(f"\n[auto] ✅ Publishable script found!")
        print(f"       Topic    : {args.topic}")
        print(f"       Category : {selected_story.category}")
        print(f"       Viral    : {selected_story.score}/100")
        print(
            f"       Audience : "
            f"{selected_audience.overall if selected_audience else 'N/A'}/100"
        )
        print(f"       Combined : {combined:.0f}/100")
        print(f"       Source   : {selected_story.source}")
        print(f"       URL      : {selected_story.url}")
        print(f"       Article  : {article_chars:,} chars, {article_notes_words} note words\n")

    # ------------------------------------------------------------------
    # Build output paths
    # ------------------------------------------------------------------
    paths = make_output_paths(args.topic, out_root=args.out_root)

    # Legacy --out override
    final_video_path = Path(args.out) if args.out else paths.video
    if args.out:
        final_video_path.parent.mkdir(parents=True, exist_ok=True)

    work = paths.work

    # ------------------------------------------------------------------
    # [1/5] Script generation
    # --auto: script already generated above and gate passed.
    # Manual topic: generate here with mandatory gate.
    # ------------------------------------------------------------------
    if args.auto:
        # Script already generated and gate-passed in the story loop above.
        script_text = winning_script
        validation  = winning_val
        fact_safe   = winning_safe
        invented    = winning_inv
        print("[1/5] Script ✅ (gate-passed during story selection)")
    else:
        # Manual topic: generate with retry loop
        print(f"[1/5] Generating script via provider '{provider}'…")
        result = _generate_script_with_retry(
            args.topic,
            args.duration,
            provider,
            args.model,
            args.script_file,
            prov_obj,
            article_notes=article_notes,
            article_text="",   # no article for manual topics
            max_rounds=quality_mod.SCRIPT_MAX_REGEN_ROUNDS,
        )
        if result is None:
            print(
                "\n❌ Script quality gate failed after all attempts."
                "\n   The topic may not be suitable for a 90-120 word Short."
                "\n   Try a more specific or newsworthy topic."
            )
            return
        script_text, validation, fact_safe, invented = result


    # Save script and article debug files
    (work / "script.txt").write_text(script_text, encoding="utf-8")

    if article_extracted and selected_story is not None:
        try:
            (work / "article_text.txt").write_text(selected_story.article_text, encoding="utf-8")
            (work / "article_notes.txt").write_text(selected_story.article_notes, encoding="utf-8")
        except Exception:
            pass

    print(script_text)

    # Generation summary (mid-run snapshot before visuals)
    _print_generation_summary(
        article_extracted=article_extracted,
        article_chars=article_chars,
        article_notes_words=article_notes_words,
        script_score=validation.structure_score,
        script_words=validation.word_count,
        script_seconds=validation.estimated_seconds,
        fact_safe=fact_safe,
    )

    # ------------------------------------------------------------------
    # [2–5] Visual pipeline — only reached when script gate has PASSED
    # ------------------------------------------------------------------
    if args.visual_mode == "image":
        image_count, sanitized_count = _run_image_mode(
            args, script_text, paths, final_video_path, provider, prov_obj
        )
    else:
        image_count, sanitized_count = _run_video_mode(
            args, script_text, paths, final_video_path
        )

    # ------------------------------------------------------------------
    # Production report + generation history
    # ------------------------------------------------------------------
    report = quality_mod.build_production_report(
        topic=args.topic,
        start_time=run_start,
        story=selected_story,
        audience=selected_audience,
        validation=validation,
        fact_safe=fact_safe,
        invented_items=invented,
        image_count=image_count,
        prompts_sanitized=sanitized_count,
    )

    quality_mod.print_quality_report(report)

    # Save article debug files to permanent video folder too
    if article_extracted and selected_story is not None:
        try:
            article_dest = paths.video.parent / f"{paths.stem}_article_text.txt"
            notes_dest   = paths.video.parent / f"{paths.stem}_article_notes.txt"
            article_dest.write_text(selected_story.article_text, encoding="utf-8")
            notes_dest.write_text(selected_story.article_notes, encoding="utf-8")
            print(f"Article → {article_dest}")
            print(f"Notes   → {notes_dest}")
        except Exception:
            pass

    quality_mod.write_production_report(report, paths.quality_report)
    print(f"Report  → {paths.quality_report}")

    history_entry = quality_mod.build_history_entry(report, final_video_path)
    quality_mod.append_history(history_entry, paths.history)
    print(f"History → {paths.history}")

    # Final summary with visual assets and total time
    visual_summary = (
        f"{image_count} image(s)"
        + (f" ({sanitized_count} sanitized)" if sanitized_count else "")
        if image_count > 0
        else "video background"
    )
    _print_generation_summary(
        article_extracted=article_extracted,
        article_chars=article_chars,
        article_notes_words=article_notes_words,
        script_score=validation.structure_score,
        script_words=validation.word_count,
        script_seconds=validation.estimated_seconds,
        fact_safe=fact_safe,
        visual_assets=visual_summary,
        total_time=time.time() - run_start,
    )


if __name__ == "__main__":
    main()

