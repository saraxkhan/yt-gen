"""dashboard/server.py — Flask backend for the AI Shorts Studio dashboard.

Run from the project root:
    python -m dashboard.server
    # or
    python dashboard/server.py

All generation happens in background threads so SSE progress streams work.
The existing shorts_gen pipeline is imported directly — no logic is duplicated.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── make sure the project root is on sys.path ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import (
    Flask, Response, jsonify, render_template,
    request, send_file, send_from_directory,
)

# ── pipeline imports ───────────────────────────────────────────────────────
from shorts_gen import script as script_mod
from shorts_gen import tts, subtitles, video as video_mod
from shorts_gen import image_prompts as img_prompts_mod
from shorts_gen import images as images_mod
from shorts_gen import news as news_mod
from shorts_gen import quality as quality_mod
from shorts_gen.output_paths import make_output_paths, write_latest_copy

OUTPUT_ROOT = ROOT / "output"

# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024   # 64 MB upload limit

# ── In-memory job registry ─────────────────────────────────────────────────
# jobs[job_id] = { "status": "running"|"done"|"error",
#                  "log": [...], "result": {...}, "q": queue.Queue }
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _new_job_id() -> str:
    return f"job_{int(time.time() * 1000)}"


def _job_log(job_id: str, msg: str, kind: str = "info") -> None:
    """Append a log line and push it to the SSE queue."""
    entry = {"ts": time.time(), "msg": msg, "kind": kind}
    with _jobs_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id]["log"].append(entry)
        q: queue.Queue = _jobs[job_id]["q"]
    q.put(entry)


def _finish_job(job_id: str, status: str, result: dict | None = None) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id]["status"] = status
        _jobs[job_id]["result"] = result or {}
        _jobs[job_id]["q"].put({"done": True, "status": status, "result": result or {}})


def _read_history() -> list[dict]:
    p = OUTPUT_ROOT / "history.json"
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _read_quality_report(video_stem: str) -> dict | None:
    p = OUTPUT_ROOT / "videos" / f"{video_stem}_quality_report.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _list_videos() -> list[dict]:
    """Return sorted list of generated video metadata."""
    vdir = OUTPUT_ROOT / "videos"
    if not vdir.is_dir():
        return []
    results = []
    for mp4 in sorted(vdir.glob("*.mp4"), reverse=True):
        stem = mp4.stem
        report = _read_quality_report(stem)
        results.append({
            "stem":    stem,
            "file":    mp4.name,
            "size_mb": round(mp4.stat().st_size / 1024 / 1024, 2),
            "report":  report,
        })
    return results


# ──────────────────────────────────────────────────────────────────────────
# Static / page routes
# ──────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# Serve output files (videos, audio, images) to the browser
@app.route("/output/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(str(OUTPUT_ROOT), filename)


# ──────────────────────────────────────────────────────────────────────────
# API — dashboard data
# ──────────────────────────────────────────────────────────────────────────

@app.route("/api/history")
def api_history():
    history = _read_history()
    # Optional filters from query params
    q_search  = (request.args.get("q") or "").lower()
    q_min_score = int(request.args.get("min_score") or 0)
    q_date    = request.args.get("date") or ""   # YYYY-MM-DD

    if q_search:
        history = [h for h in history if q_search in (h.get("topic") or "").lower()
                   or q_search in (h.get("source") or "").lower()]
    if q_min_score:
        history = [h for h in history if (h.get("script_score") or 0) >= q_min_score
                   or (h.get("viral_score") or 0) >= q_min_score]
    if q_date:
        history = [h for h in history if (h.get("date") or "").startswith(q_date)]

    return jsonify({"history": history})


@app.route("/api/videos")
def api_videos():
    return jsonify({"videos": _list_videos()})


@app.route("/api/quality_report/<stem>")
def api_quality_report(stem: str):
    report = _read_quality_report(stem)
    if report is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(report)


@app.route("/api/stats")
def api_stats():
    history = _read_history()
    vdir = OUTPUT_ROOT / "videos"
    n_videos = len(list(vdir.glob("*.mp4"))) if vdir.is_dir() else 0
    scores = [h.get("script_score") or 0 for h in history if h.get("script_score")]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    passed = sum(1 for h in history if h.get("final_pass"))
    return jsonify({
        "total_videos": n_videos,
        "total_runs":   len(history),
        "avg_script_score": avg_score,
        "total_passed": passed,
        "output_dir":   str(OUTPUT_ROOT),
    })


@app.route("/api/open_output_folder", methods=["POST"])
def api_open_output_folder():
    try:
        if sys.platform == "win32":
            os.startfile(str(OUTPUT_ROOT))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(OUTPUT_ROOT)])
        else:
            subprocess.Popen(["xdg-open", str(OUTPUT_ROOT)])
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)})


# ──────────────────────────────────────────────────────────────────────────
# API — News
# ──────────────────────────────────────────────────────────────────────────

@app.route("/api/news/fetch", methods=["POST"])
def api_news_fetch():
    """Kick off a background news fetch and return a job_id."""
    body = request.get_json(silent=True) or {}
    model = body.get("model") or None

    job_id = _new_job_id()
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "log": [], "result": {}, "q": queue.Queue()}

    def _run():
        try:
            _job_log(job_id, "Fetching AI news from all sources…")
            prov = None
            try:
                prov = script_mod.get_provider("ollama")
            except Exception:
                pass

            stories = news_mod.fetch_and_score(provider=prov, model=model, verbose=False)
            _job_log(job_id, f"Fetched {len(stories)} unique stories")

            # Apply quality gate
            gated = []
            for s in stories:
                passes, reasons = quality_mod.story_passes_gate(s)
                gated.append({
                    "title":      s.title,
                    "source":     s.source,
                    "url":        s.url,
                    "score":      s.score,
                    "age_hours":  round(s.age_hours, 1),
                    "category":   s.category,
                    "summary":    s.summary or "",
                    "passes_gate": passes,
                    "gate_reasons": reasons,
                })

            gated.sort(key=lambda x: x["score"], reverse=True)
            _job_log(job_id, f"{sum(1 for g in gated if g['passes_gate'])} stories passed quality gate")
            _finish_job(job_id, "done", {"stories": gated})
        except Exception as exc:
            _job_log(job_id, f"Error: {exc}", "error")
            _finish_job(job_id, "error", {"error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})


# ──────────────────────────────────────────────────────────────────────────
# API — Script generation / preview
# ──────────────────────────────────────────────────────────────────────────

@app.route("/api/generate/script", methods=["POST"])
def api_generate_script():
    body = request.get_json(silent=True) or {}
    topic    = (body.get("topic") or "").strip()
    model    = body.get("model") or None
    duration = int(body.get("duration") or 40)
    provider = body.get("provider") or "ollama"

    if not topic:
        return jsonify({"error": "topic required"}), 400

    job_id = _new_job_id()
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "log": [], "result": {}, "q": queue.Queue()}

    def _run():
        try:
            _job_log(job_id, f"Generating script for: {topic}")
            prov_obj = None
            try:
                prov_obj = script_mod.get_provider(provider)
            except Exception:
                pass

            text = script_mod.generate_script(
                topic, duration, provider=provider, model=model
            )
            _job_log(job_id, "Script generated — running validation…")

            validation = quality_mod.validate_script(text, topic, provider=prov_obj, model=model)
            fact_safe, invented = quality_mod.fact_safety_check(text, topic, provider=prov_obj, model=model)

            _job_log(job_id,
                f"Validation: {'✅ PASS' if validation.passes else '❌ FAIL'} "
                f"(score={validation.structure_score}/10, words={validation.word_count})"
            )
            _finish_job(job_id, "done", {
                "script":   text,
                "validation": {
                    "passes":           validation.passes,
                    "word_count":       validation.word_count,
                    "estimated_seconds": validation.estimated_seconds,
                    "hook_score":       validation.hook_score,
                    "structure_score":  validation.structure_score,
                    "has_hook":         validation.has_hook,
                    "has_curiosity_gap":validation.has_curiosity_gap,
                    "has_payoff":       validation.has_payoff,
                    "has_why_it_matters":validation.has_why_it_matters,
                    "has_cta":          validation.has_cta,
                    "fact_safe":        fact_safe,
                    "invented_items":   invented,
                    "issues":           validation.issues,
                },
            })
        except Exception as exc:
            _job_log(job_id, f"Error: {exc}", "error")
            _job_log(job_id, traceback.format_exc(), "error")
            _finish_job(job_id, "error", {"error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})


# ──────────────────────────────────────────────────────────────────────────
# API — Scene plan / image prompts preview
# ──────────────────────────────────────────────────────────────────────────

@app.route("/api/generate/scene_plan", methods=["POST"])
def api_generate_scene_plan():
    body = request.get_json(silent=True) or {}
    topic  = (body.get("topic") or "").strip()
    script = (body.get("script") or "").strip()
    model  = body.get("model") or None
    provider = body.get("provider") or "ollama"

    if not topic or not script:
        return jsonify({"error": "topic and script required"}), 400

    job_id = _new_job_id()
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "log": [], "result": {}, "q": queue.Queue()}

    def _run():
        try:
            _job_log(job_id, "Building cinematic scene plan…")
            plan = img_prompts_mod.generate_scene_plan(
                topic, script, provider=provider, model=model, verbose=False
            )
            _job_log(job_id, f"Scene plan: {len(plan.scenes)} scenes, quality={plan.quality_score}/10")

            _job_log(job_id, "Generating cinematic image prompts…")
            prompts = img_prompts_mod.scene_plan_to_prompts(
                plan, topic, provider=provider, model=model, verbose=False
            )
            _job_log(job_id, f"Generated {len(prompts)} prompts")

            _finish_job(job_id, "done", {
                "plan":    plan.to_dict(),
                "prompts": prompts,
            })
        except Exception as exc:
            _job_log(job_id, f"Error: {exc}", "error")
            _finish_job(job_id, "error", {"error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})


# ──────────────────────────────────────────────────────────────────────────
# API — Full pipeline generation
# ──────────────────────────────────────────────────────────────────────────

@app.route("/api/generate/full", methods=["POST"])
def api_generate_full():
    body = request.get_json(silent=True) or {}
    topic    = (body.get("topic") or "").strip()
    model    = body.get("model") or None
    provider = body.get("provider") or "ollama"
    duration = int(body.get("duration") or 40)
    voice    = body.get("voice") or tts.DEFAULT_VOICE
    image_provider = body.get("image_provider") or "placeholder"
    whisper_model  = body.get("whisper_model") or "small"
    auto_news      = bool(body.get("auto_news", False))

    if not topic and not auto_news:
        return jsonify({"error": "topic required unless auto_news=true"}), 400

    job_id = _new_job_id()
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "log": [], "result": {}, "q": queue.Queue()}

    def _run():
        run_start = time.time()
        selected_story = None
        selected_audience = None

        try:
            prov_obj = None
            try:
                prov_obj = script_mod.get_provider(provider)
            except Exception:
                pass

            # ── Auto news ──────────────────────────────────────────────
            nonlocal topic
            if auto_news:
                _job_log(job_id, "📰 Fetching today's AI news…")
                stories = news_mod.fetch_and_score(provider=prov_obj, model=model, verbose=False)
                _job_log(job_id, f"Found {len(stories)} stories")

                gated = [s for s in stories if quality_mod.story_passes_gate(s)[0]]
                if not gated:
                    _finish_job(job_id, "error", {"error": "No high-quality AI news found today."})
                    return

                _job_log(job_id, f"{len(gated)} stories passed quality gate")
                candidates = gated[:3]

                _job_log(job_id, "Scoring audience appeal…")
                aud_map = {}
                for cand in candidates:
                    aud = quality_mod.score_audience_appeal(cand, provider=prov_obj, model=model)
                    aud_map[cand.title] = aud

                selected_story = max(
                    candidates,
                    key=lambda s: quality_mod.combined_story_score(s, aud_map.get(s.title)),
                )
                selected_audience = aud_map.get(selected_story.title)
                topic = selected_story.title
                _job_log(job_id, f"✅ Selected: {topic}")
                _job_log(job_id, f"   Score: {selected_story.score}/100  Category: {selected_story.category}")

            # ── Script ────────────────────────────────────────────────
            _job_log(job_id, "✍️  Generating script…")
            paths = make_output_paths(topic, out_root=OUTPUT_ROOT)

            script_text = script_mod.generate_script(
                topic, duration, provider=provider, model=model
            )
            _job_log(job_id, f"Script generated ({len(script_text.split())} words)")

            validation = quality_mod.validate_script(script_text, topic, provider=prov_obj, model=model)
            fact_safe, invented = quality_mod.fact_safety_check(script_text, topic, provider=prov_obj, model=model)
            _job_log(job_id,
                f"Validation: {'✅ PASS' if validation.passes else '⚠️ WARN'} "
                f"score={validation.structure_score}/10 words={validation.word_count}"
            )

            if not validation.passes or not fact_safe:
                _job_log(job_id, "Regenerating script (validation failed)…", "warn")
                script_text2 = script_mod.generate_script(topic, duration, provider=provider, model=model)
                val2 = quality_mod.validate_script(script_text2, topic, provider=prov_obj, model=model)
                fs2, inv2 = quality_mod.fact_safety_check(script_text2, topic, provider=prov_obj, model=model)
                if val2.structure_score >= validation.structure_score:
                    script_text, validation, fact_safe, invented = script_text2, val2, fs2, inv2

            (paths.work / "script.txt").write_text(script_text, encoding="utf-8")

            # ── Scene plan + image prompts ─────────────────────────────
            _job_log(job_id, "🎬 Building cinematic scene plan…")
            plan = img_prompts_mod.generate_scene_plan(
                topic, script_text, provider=provider, model=model, verbose=False
            )
            _job_log(job_id, f"Scene plan: {len(plan.scenes)} scenes, quality={plan.quality_score}/10")

            prompts = img_prompts_mod.scene_plan_to_prompts(
                plan, topic, provider=provider, model=model, verbose=False
            )
            prompts, sanitized = quality_mod.sanitize_visual_prompts(
                prompts, provider=prov_obj, model=model, verbose=False
            )
            (paths.work / "image_prompts.txt").write_text("\n\n".join(prompts), encoding="utf-8")
            _job_log(job_id, f"Generated {len(prompts)} image prompts ({sanitized} sanitized)")

            # ── Voice ─────────────────────────────────────────────────
            _job_log(job_id, "🔊 Synthesizing voiceover…")
            vo_path = tts.synthesize(script_text, paths.audio, voice=voice)
            _job_log(job_id, f"Voiceover: {paths.audio.name}")

            # ── Subtitles ─────────────────────────────────────────────
            _job_log(job_id, "📝 Transcribing for word timings…")
            words = subtitles.transcribe_words(vo_path, model_size=whisper_model)
            ass_path = subtitles.build_ass(words, paths.subtitles)
            _job_log(job_id, f"Subtitles: {paths.subtitles.name}")

            # ── Images ────────────────────────────────────────────────
            _job_log(job_id, f"🖼️  Generating {len(prompts)} images via '{image_provider}'…")
            image_paths = images_mod.generate_images(
                prompts, paths.images_dir, provider=image_provider
            )
            _job_log(job_id, f"Generated {len(image_paths)} images")

            # ── Video render ──────────────────────────────────────────
            _job_log(job_id, "🎞️  Compositing video…")
            scene_durations   = [s.duration for s in plan.scenes]
            scene_transitions = [s.transition for s in plan.scenes[1:]]
            video_mod.compose_from_images(
                image_paths, vo_path, ass_path, paths.video,
                scene_durations=scene_durations,
                scene_transitions=scene_transitions,
            )
            write_latest_copy(paths.video, paths.latest)
            _job_log(job_id, f"✅ Video rendered: {paths.video.name}")

            # ── Quality report + history ──────────────────────────────
            report = quality_mod.build_production_report(
                topic=topic,
                start_time=run_start,
                story=selected_story,
                audience=selected_audience,
                validation=validation,
                fact_safe=fact_safe,
                invented_items=invented,
                image_count=len(image_paths),
                prompts_sanitized=sanitized,
            )
            quality_mod.write_production_report(report, paths.quality_report)
            entry = quality_mod.build_history_entry(report, paths.video)
            quality_mod.append_history(entry, paths.history)

            _finish_job(job_id, "done", {
                "video_stem":   paths.stem,
                "video_file":   f"output/videos/{paths.video.name}",
                "script":       script_text,
                "prompts":      prompts,
                "scene_plan":   plan.to_dict(),
                "report":       quality_mod._report_to_dict(report),
                "generation_time": round(time.time() - run_start, 1),
            })

        except Exception as exc:
            _job_log(job_id, f"❌ Error: {exc}", "error")
            _job_log(job_id, traceback.format_exc(), "error")
            _finish_job(job_id, "error", {"error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})


# ──────────────────────────────────────────────────────────────────────────
# API — Partial regeneration (individual stages)
# ──────────────────────────────────────────────────────────────────────────

@app.route("/api/regen/<stage>", methods=["POST"])
def api_regen(stage: str):
    """Regenerate a single pipeline stage for an existing stem."""
    body = request.get_json(silent=True) or {}
    stem     = (body.get("stem") or "").strip()
    topic    = (body.get("topic") or "").strip()
    model    = body.get("model") or None
    provider = body.get("provider") or "ollama"

    if not stem or not topic:
        return jsonify({"error": "stem and topic required"}), 400
    if stage not in ("script", "scene_plan", "voice", "subtitles", "video"):
        return jsonify({"error": f"unknown stage '{stage}'"}), 400

    job_id = _new_job_id()
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "log": [], "result": {}, "q": queue.Queue()}

    def _run():
        try:
            vdir    = OUTPUT_ROOT / "videos"
            adir    = OUTPUT_ROOT / "audio"
            sdir    = OUTPUT_ROOT / "subtitles"
            idir    = OUTPUT_ROOT / "images" / stem
            work    = OUTPUT_ROOT / "_work"
            script_path = work / "script.txt"
            prompts_path = work / "image_prompts.txt"

            prov_obj = None
            try:
                prov_obj = script_mod.get_provider(provider)
            except Exception:
                pass

            if stage == "script":
                _job_log(job_id, f"Regenerating script for: {topic}")
                text = script_mod.generate_script(topic, 40, provider=provider, model=model)
                script_path.write_text(text, encoding="utf-8")
                val = quality_mod.validate_script(text, topic, provider=prov_obj, model=model)
                fs, inv = quality_mod.fact_safety_check(text, topic, provider=prov_obj, model=model)
                _job_log(job_id, f"Done — score={val.structure_score}/10 words={val.word_count}")
                _finish_job(job_id, "done", {
                    "script": text,
                    "validation": {
                        "structure_score": val.structure_score,
                        "word_count":      val.word_count,
                        "passes":          val.passes,
                        "fact_safe":       fs,
                        "issues":          val.issues,
                    },
                })

            elif stage == "scene_plan":
                script_text = script_path.read_text(encoding="utf-8") if script_path.is_file() else topic
                _job_log(job_id, "Regenerating scene plan + image prompts…")
                plan = img_prompts_mod.generate_scene_plan(
                    topic, script_text, provider=provider, model=model, verbose=False
                )
                prompts = img_prompts_mod.scene_plan_to_prompts(
                    plan, topic, provider=provider, model=model, verbose=False
                )
                prompts, _ = quality_mod.sanitize_visual_prompts(
                    prompts, provider=prov_obj, model=model, verbose=False
                )
                prompts_path.write_text("\n\n".join(prompts), encoding="utf-8")
                _job_log(job_id, f"Done — {len(plan.scenes)} scenes, {len(prompts)} prompts")
                _finish_job(job_id, "done", {"plan": plan.to_dict(), "prompts": prompts})

            elif stage == "voice":
                script_text = script_path.read_text(encoding="utf-8") if script_path.is_file() else topic
                audio_path  = adir / f"{stem}.mp3"
                _job_log(job_id, "Regenerating voiceover…")
                tts.synthesize(script_text, audio_path, voice=body.get("voice") or tts.DEFAULT_VOICE)
                _job_log(job_id, f"Done — {audio_path.name}")
                _finish_job(job_id, "done", {"audio_file": f"output/audio/{audio_path.name}"})

            elif stage == "subtitles":
                audio_path = adir / f"{stem}.mp3"
                sub_path   = sdir / f"{stem}.ass"
                if not audio_path.is_file():
                    raise FileNotFoundError("Audio file not found — regenerate voice first")
                _job_log(job_id, "Regenerating subtitles…")
                words = subtitles.transcribe_words(audio_path, model_size=body.get("whisper_model") or "small")
                subtitles.build_ass(words, sub_path)
                _job_log(job_id, f"Done — {sub_path.name}")
                _finish_job(job_id, "done", {"subtitles_file": f"output/subtitles/{sub_path.name}"})

            elif stage == "video":
                audio_path = adir / f"{stem}.mp3"
                sub_path   = sdir / f"{stem}.ass"
                video_path = vdir / f"{stem}.mp4"
                imgs = sorted(idir.glob("*.png")) + sorted(idir.glob("*.jpg")) if idir.is_dir() else []
                if not imgs:
                    imgs = sorted((OUTPUT_ROOT / "images" / stem).glob("*.*"))
                if not audio_path.is_file() or not sub_path.is_file():
                    raise FileNotFoundError("Audio or subtitles missing — regenerate those first")
                _job_log(job_id, f"Recompositing video from {len(imgs)} images…")
                video_mod.compose_from_images(imgs, audio_path, sub_path, video_path)
                write_latest_copy(video_path, OUTPUT_ROOT / "short.mp4")
                _job_log(job_id, f"Done — {video_path.name}")
                _finish_job(job_id, "done", {"video_file": f"output/videos/{video_path.name}"})

        except Exception as exc:
            _job_log(job_id, f"Error: {exc}", "error")
            _finish_job(job_id, "error", {"error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})


# ──────────────────────────────────────────────────────────────────────────
# SSE — real-time job progress stream
# ──────────────────────────────────────────────────────────────────────────

@app.route("/api/jobs/<job_id>/stream")
def api_job_stream(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404

    def _generate():
        q: queue.Queue = job["q"]
        # Replay existing log entries first
        with _jobs_lock:
            backlog = list(job["log"])
            already_done = job["status"] != "running"
        for entry in backlog:
            yield f"data: {json.dumps(entry)}\n\n"
        if already_done:
            done_payload = {"done": True, "status": job["status"], "result": job.get("result", {})}
            yield f"data: {json.dumps(done_payload)}\n\n"
            return
        # Stream live updates
        while True:
            try:
                msg = q.get(timeout=30)
            except queue.Empty:
                yield "data: {\"ping\":true}\n\n"
                continue
            yield f"data: {json.dumps(msg)}\n\n"
            if msg.get("done"):
                break

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/jobs/<job_id>")
def api_job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "log":    job["log"],
        "result": job.get("result", {}),
    })


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser
    port = int(os.environ.get("DASHBOARD_PORT", 7842))
    print(f"\n  AI Shorts Studio  ->  http://localhost:{port}\n")
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
