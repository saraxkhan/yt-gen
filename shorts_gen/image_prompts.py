"""Scene-plan-driven cinematic image prompts — shorts_gen/image_prompts.py

Instead of generating flat image descriptions, this module:
  1. Asks the LLM to create a structured SCENE PLAN for the narration,
     dividing it into narrative beats with purpose, emotion, camera style,
     duration, and transition metadata.
  2. Converts each scene into a richly-specified cinematic image prompt
     that includes subject, environment, lighting, composition, camera
     angle, cinematic style, and emotion.
  3. Scores the resulting scene plan for visual quality (variety, motion,
     retention, scene flow, cinematic quality); regenerates if < 8/10.
  4. Falls back gracefully to a heuristic when no LLM is available.

Public API consumed by main.py / video.py:
    generate_scene_plan(...)   → ScenePlan
    scene_plan_to_prompts(...) → list[str]
    generate_image_prompts(...)→ list[str]  (legacy compat, still works)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from . import script as script_mod

MIN_IMAGES = 4
MAX_IMAGES = 7   # allow up to 7 for richer scene pacing

# Cohesive style suffix appended to every generated prompt.
STYLE_SUFFIX = (
    "cinematic, ultra-detailed, dramatic lighting, high contrast, "
    "9:16 vertical composition, photorealistic, 8k, no text, no watermark, "
    "no logos, no UI elements"
)

# Narrative beat purposes that map to script structure
SCENE_PURPOSES = [
    "Hook",
    "Curiosity Gap",
    "Reveal",
    "Main Explanation",
    "Why It Matters",
    "Payoff",
    "CTA",
]

# Duration targets per purpose (seconds)
SCENE_DURATION_MAP = {
    "Hook":             (3, 5),
    "Curiosity Gap":    (3, 5),
    "Reveal":           (4, 6),
    "Main Explanation": (5, 8),
    "Why It Matters":   (4, 7),
    "Payoff":           (5, 8),
    "CTA":              (2, 3),
}

# ---------------------------------------------------------------------------
# Scene Plan dataclass
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    """A single narrative scene in the shot plan."""
    index: int
    purpose: str            # e.g. "Hook", "Reveal", "CTA"
    duration: float         # seconds
    visual_description: str # what to show
    emotion: str            # e.g. "Curiosity", "Surprise", "Excitement"
    camera_style: str       # e.g. "extreme close-up", "wide establishing"
    transition: str         # e.g. "crossfade", "zoom", "flash"
    prompt: str = ""        # filled by scene_plan_to_prompts()


@dataclass
class ScenePlan:
    """Complete shot plan for a single Short."""
    scenes: List[Scene] = field(default_factory=list)
    quality_score: int = 0          # 0-10 overall visual quality
    visual_variety: int = 0         # 0-10
    motion_quality: int = 0         # 0-10
    retention: int = 0              # 0-10
    scene_flow: int = 0             # 0-10
    cinematic_quality: int = 0      # 0-10
    quality_issues: List[str] = field(default_factory=list)

    @property
    def passes_quality(self) -> bool:
        return self.quality_score >= 8

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.scenes)

    def to_dict(self) -> dict:
        return {
            "scenes": [
                {
                    "index": s.index,
                    "purpose": s.purpose,
                    "duration": s.duration,
                    "visual_description": s.visual_description,
                    "emotion": s.emotion,
                    "camera_style": s.camera_style,
                    "transition": s.transition,
                    "prompt": s.prompt,
                }
                for s in self.scenes
            ],
            "quality": {
                "score": self.quality_score,
                "visual_variety": self.visual_variety,
                "motion_quality": self.motion_quality,
                "retention": self.retention,
                "scene_flow": self.scene_flow,
                "cinematic_quality": self.cinematic_quality,
                "issues": self.quality_issues,
            },
        }


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_SCENE_PLAN_SYSTEM = """\
You are a cinematic director and visual storyteller for YouTube Shorts AI news videos.
Your job is to design a compelling shot plan that keeps viewers engaged for 35-45 seconds.

Return valid JSON only — no prose, no markdown fences.
"""

_SCENE_PLAN_PROMPT = """\
Create a cinematic scene plan for this AI news YouTube Short.

Topic: {topic}

Narration script:
\"\"\"
{script}
\"\"\"

Design {n} scenes (between 4 and 7) that visually tell this story beat by beat.
Each scene should serve a clear narrative purpose and keep the viewer hooked.

Return ONLY a JSON object:
{{
  "scenes": [
    {{
      "index": 1,
      "purpose": "<one of: Hook | Curiosity Gap | Reveal | Main Explanation | Why It Matters | Payoff | CTA>",
      "duration": <float, seconds — Hook: 3-5s, Reveal: 4-6s, Explanation: 5-8s, CTA: 2-3s>,
      "visual_description": "<rich visual description: subject, environment, what is happening>",
      "emotion": "<one word: Curiosity | Surprise | Excitement | Tension | Awe | Hope | Urgency>",
      "camera_style": "<e.g.: extreme close-up on eyes, wide establishing shot, Dutch angle, low-angle hero shot, over-the-shoulder, aerial drone perspective>",
      "transition": "<one of: crossfade | zoom_in | zoom_out | flash | blur | slide_left | slide_right>"
    }}
  ]
}}

Rules:
- Each scene must have a DISTINCT emotion and camera style — never repeat the same camera angle twice
- Duration must total 35-45 seconds across all scenes
- Hook scene must be dramatic and scroll-stopping
- CTA scene should feel energetic and urgent
- Visual descriptions must be specific and cinematic (no generic descriptions)
- NO text, logos, UI elements, or screenshots in any visual description
"""

_QUALITY_SCORE_SYSTEM = """\
You are a visual quality assessor for YouTube Shorts productions.
Score scene plans honestly and harshly. Return valid JSON only.
"""

_QUALITY_SCORE_PROMPT = """\
Score this YouTube Shorts scene plan for visual quality.

Scene plan:
{scene_plan_json}

Return ONLY a JSON object:
{{
  "visual_variety":    <integer 0-10, are scenes visually distinct from each other?>,
  "motion_quality":    <integer 0-10, does camera movement feel cinematic and varied?>,
  "retention":         <integer 0-10, will viewers stay engaged for 35-45 seconds?>,
  "scene_flow":        <integer 0-10, does the story build naturally from scene to scene?>,
  "cinematic_quality": <integer 0-10, does it feel like a produced video or a slideshow?>,
  "overall":           <integer 0-10, holistic quality — must be 8+ for professional output>,
  "issues":            <array of up to 4 specific problems, empty if overall >= 8>
}}

Be harsh. A scene plan with repeated emotions, identical camera styles, or generic
descriptions should score 5-6. Only genuinely cinematic plans score 8+.
"""

_CINEMATIC_PROMPT_SYSTEM = """\
You are a senior AI image-generation prompt engineer specialising in cinematic
vertical video for YouTube Shorts. Convert scene descriptions into richly-detailed
image generation prompts that produce stunning, professional-looking visuals.
Return valid JSON only — no prose, no markdown fences.
"""

_CINEMATIC_PROMPT_TEMPLATE = """\
Convert these scene descriptions into richly-detailed image generation prompts.

Topic: {topic}
Overall emotion arc: {emotion_arc}

Scenes to convert:
{scenes_json}

For EACH scene, create one image-generation prompt containing ALL of:
  - Subject: the main visual subject (person, object, environment)
  - Environment: physical setting, atmosphere, time of day
  - Lighting: type and direction (golden hour, neon-lit, dramatic chiaroscuro, etc.)
  - Composition: how the frame is structured (rule of thirds, leading lines, etc.)
  - Camera angle: exactly as specified in the scene plan
  - Cinematic style: (e.g. "shot on ARRI Alexa", "anamorphic lens flare", "noir aesthetic")
  - Emotion: convey the specified emotion through visual elements

Rules — every prompt MUST follow these:
  - NO text, words, letters, captions, subtitles of any kind
  - NO logos, brand marks, watermarks, or UI elements
  - NO screenshots, app interfaces, or website mockups
  - NO repeated subjects or environments across prompts
  - EACH prompt must be visually distinct from all others
  - End every prompt with the style suffix exactly as: {style_suffix}

Return ONLY a JSON object:
{{
  "prompts": ["<prompt 1>", "<prompt 2>", ...]
}}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict | None:
    raw = re.sub(r"```[^\n]*\n?|```", "", raw).strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        return json.loads(raw[s : e + 1])
    except json.JSONDecodeError:
        return None


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Heuristic fallback (no LLM)
# ---------------------------------------------------------------------------

_HEURISTIC_TRANSITIONS = [
    "crossfade", "zoom_in", "flash", "blur", "zoom_out",
    "slide_left", "slide_right",
]

_HEURISTIC_CAMERAS = [
    "dramatic extreme close-up, shallow depth of field",
    "wide cinematic establishing shot, anamorphic lens",
    "low-angle hero shot looking up",
    "Dutch angle medium shot creating tension",
    "over-the-shoulder perspective, bokeh background",
    "aerial top-down perspective looking straight down",
    "eye-level tight framing, foreground elements blurred",
]

_HEURISTIC_EMOTIONS = [
    "Curiosity", "Surprise", "Excitement", "Awe", "Tension", "Hope", "Urgency"
]

_PURPOSE_SEQUENCE = [
    ("Hook", (3, 5)),
    ("Curiosity Gap", (3, 5)),
    ("Main Explanation", (6, 8)),
    ("Why It Matters", (5, 7)),
    ("Payoff", (5, 7)),
    ("CTA", (2, 3)),
]


def _heuristic_scene_plan(topic: str, script_text: str, n: int) -> ScenePlan:
    """Build a basic scene plan without LLM — guaranteed to produce n scenes."""
    sentences = _split_sentences(script_text) or [topic]
    n = max(MIN_IMAGES, min(MAX_IMAGES, n))

    # Distribute sentences across scenes
    size = max(1, len(sentences) / n)
    chunks: List[str] = []
    for i in range(n):
        start = int(round(i * size))
        end = int(round((i + 1) * size))
        chunks.append(" ".join(sentences[start:end]).strip() or topic)

    # Pick purpose sequence
    purposes = _PURPOSE_SEQUENCE[:n]
    if len(purposes) < n:
        purposes += [("Main Explanation", (5, 7))] * (n - len(purposes))

    scenes: List[Scene] = []
    for i, (chunk, (purpose, dur_range)) in enumerate(zip(chunks, purposes)):
        duration = (dur_range[0] + dur_range[1]) / 2
        scenes.append(Scene(
            index=i + 1,
            purpose=purpose,
            duration=duration,
            visual_description=f"{chunk}. Topic: {topic}.",
            emotion=_HEURISTIC_EMOTIONS[i % len(_HEURISTIC_EMOTIONS)],
            camera_style=_HEURISTIC_CAMERAS[i % len(_HEURISTIC_CAMERAS)],
            transition=_HEURISTIC_TRANSITIONS[i % len(_HEURISTIC_TRANSITIONS)],
        ))

    plan = ScenePlan(scenes=scenes, quality_score=6)
    return plan


def _heuristic_prompt_for_scene(scene: Scene, topic: str) -> str:
    """Convert a single scene into a cinematic image prompt (no LLM)."""
    return (
        f"{scene.camera_style.capitalize()} of a scene depicting: "
        f"{scene.visual_description} "
        f"Emotion: {scene.emotion.lower()} atmosphere. "
        f"Topic context: {topic}. "
        f"{STYLE_SUFFIX}"
    )


# ---------------------------------------------------------------------------
# Scene plan generation (LLM-powered)
# ---------------------------------------------------------------------------

def _parse_scenes_from_json(data: dict) -> List[Scene]:
    """Parse LLM scene plan JSON into Scene objects."""
    raw_scenes = data.get("scenes", [])
    scenes: List[Scene] = []
    for raw in raw_scenes:
        if not isinstance(raw, dict):
            continue
        try:
            scenes.append(Scene(
                index=int(raw.get("index", len(scenes) + 1)),
                purpose=str(raw.get("purpose", "Main Explanation")),
                duration=float(raw.get("duration", 5.0)),
                visual_description=str(raw.get("visual_description", "")),
                emotion=str(raw.get("emotion", "Curiosity")),
                camera_style=str(raw.get("camera_style", "medium shot")),
                transition=str(raw.get("transition", "crossfade")),
            ))
        except (TypeError, ValueError):
            continue
    return scenes


def _score_scene_plan(
    plan: ScenePlan,
    chat,
    model: str | None,
) -> ScenePlan:
    """Ask the LLM to score the scene plan and annotate the plan in-place."""
    scenes_json = json.dumps(
        [
            {
                "purpose": s.purpose,
                "duration": s.duration,
                "visual_description": s.visual_description,
                "emotion": s.emotion,
                "camera_style": s.camera_style,
                "transition": s.transition,
            }
            for s in plan.scenes
        ],
        indent=2,
    )
    try:
        raw = chat(
            _QUALITY_SCORE_SYSTEM,
            _QUALITY_SCORE_PROMPT.format(scene_plan_json=scenes_json),
            model,
        )
        data = _extract_json(raw)
        if not data:
            plan.quality_score = 6
            return plan

        def _i(key: str, default: int = 5) -> int:
            return max(0, min(10, int(data.get(key, default) or default)))

        plan.visual_variety    = _i("visual_variety")
        plan.motion_quality    = _i("motion_quality")
        plan.retention         = _i("retention")
        plan.scene_flow        = _i("scene_flow")
        plan.cinematic_quality = _i("cinematic_quality")
        plan.quality_score     = _i("overall")
        plan.quality_issues    = [str(x) for x in (data.get("issues") or []) if x][:4]
    except Exception:
        plan.quality_score = 6

    return plan


def generate_scene_plan(
    topic: str,
    script_text: str,
    *,
    provider: str = "ollama",
    model: str | None = None,
    script_file: str | None = None,
    count: int = MAX_IMAGES,
    verbose: bool = True,
) -> ScenePlan:
    """Generate a cinematic scene plan for the narration.

    Attempts LLM generation with quality scoring (regenerates once if < 8/10).
    Falls back to heuristic if no LLM is available.
    """
    count = max(MIN_IMAGES, min(MAX_IMAGES, count))

    try:
        prov = script_mod.get_provider(provider, script_file=script_file)
    except Exception:
        prov = None

    chat = getattr(prov, "chat", None) if prov is not None else None
    if not callable(chat):
        if verbose:
            print("  using heuristic scene plan (no LLM provider)")
        return _heuristic_scene_plan(topic, script_text, count)

    best_plan: ScenePlan | None = None

    for attempt in range(2):          # attempt 0 = first try; attempt 1 = regeneration
        prompt = _SCENE_PLAN_PROMPT.format(
            topic=topic, script=script_text, n=count
        )
        try:
            raw = chat(_SCENE_PLAN_SYSTEM, prompt, model)
            data = _extract_json(raw)
            if not data:
                continue
            scenes = _parse_scenes_from_json(data)
        except Exception as exc:
            if verbose:
                print(f"  scene plan LLM call failed ({exc}); using heuristic")
            return _heuristic_scene_plan(topic, script_text, count)

        if not scenes:
            continue

        # Ensure scene count is in range
        if len(scenes) < MIN_IMAGES:
            scenes += _heuristic_scene_plan(topic, script_text, MIN_IMAGES - len(scenes)).scenes
        scenes = scenes[:MAX_IMAGES]
        for i, s in enumerate(scenes):
            s.index = i + 1

        plan = ScenePlan(scenes=scenes)
        plan = _score_scene_plan(plan, chat, model)

        if verbose:
            print(
                f"  scene plan (attempt {attempt + 1}): "
                f"score={plan.quality_score}/10 "
                f"(variety={plan.visual_variety} motion={plan.motion_quality} "
                f"retention={plan.retention} flow={plan.scene_flow} "
                f"cinematic={plan.cinematic_quality})"
            )
            if plan.quality_issues:
                for issue in plan.quality_issues:
                    print(f"    ⚠  {issue}")

        if best_plan is None or plan.quality_score > best_plan.quality_score:
            best_plan = plan

        if plan.passes_quality:
            break
        elif attempt == 0 and verbose:
            print(
                f"  quality score {plan.quality_score}/10 < 8 "
                f"→ regenerating scene plan…"
            )

    return best_plan if best_plan is not None else _heuristic_scene_plan(topic, script_text, count)


# ---------------------------------------------------------------------------
# Cinematic prompt generation from scene plan
# ---------------------------------------------------------------------------

def scene_plan_to_prompts(
    plan: ScenePlan,
    topic: str,
    *,
    provider: str = "ollama",
    model: str | None = None,
    script_file: str | None = None,
    verbose: bool = True,
) -> List[str]:
    """Convert a ScenePlan into richly-specified image generation prompts."""
    try:
        prov = script_mod.get_provider(provider, script_file=script_file)
    except Exception:
        prov = None

    chat = getattr(prov, "chat", None) if prov is not None else None

    # Compute emotion arc for context
    emotion_arc = " → ".join(s.emotion for s in plan.scenes)

    if callable(chat):
        scenes_json = json.dumps(
            [
                {
                    "index": s.index,
                    "purpose": s.purpose,
                    "visual_description": s.visual_description,
                    "emotion": s.emotion,
                    "camera_style": s.camera_style,
                }
                for s in plan.scenes
            ],
            indent=2,
        )
        prompt = _CINEMATIC_PROMPT_TEMPLATE.format(
            topic=topic,
            emotion_arc=emotion_arc,
            scenes_json=scenes_json,
            style_suffix=STYLE_SUFFIX,
        )
        try:
            raw = chat(_CINEMATIC_PROMPT_SYSTEM, prompt, model)
            data = _extract_json(raw)
            if data:
                prompts = data.get("prompts", [])
                if isinstance(prompts, list) and len(prompts) >= MIN_IMAGES:
                    # Attach prompts back to scenes
                    for i, scene in enumerate(plan.scenes):
                        if i < len(prompts):
                            scene.prompt = str(prompts[i])
                    # Ensure style suffix is present
                    result = []
                    for p in prompts[: len(plan.scenes)]:
                        p = str(p).strip()
                        if STYLE_SUFFIX.split(",")[0].lower() not in p.lower():
                            p = f"{p}. {STYLE_SUFFIX}"
                        result.append(p)
                    if verbose:
                        print(
                            f"  generated {len(result)} cinematic prompts "
                            f"(emotion arc: {emotion_arc})"
                        )
                    return result
        except Exception as exc:
            if verbose:
                print(f"  cinematic prompt generation failed ({exc}); using heuristic")

    # Heuristic fallback
    prompts = [_heuristic_prompt_for_scene(s, topic) for s in plan.scenes]
    for i, (scene, p) in enumerate(zip(plan.scenes, prompts)):
        scene.prompt = p
    if verbose:
        print(f"  using heuristic cinematic prompts ({len(prompts)} scenes)")
    return prompts


# ---------------------------------------------------------------------------
# Legacy public API (used by existing main.py call)
# ---------------------------------------------------------------------------

def generate_image_prompts(
    topic: str,
    script_text: str,
    *,
    provider: str = "ollama",
    model: str | None = None,
    script_file: str | None = None,
    count: int = MAX_IMAGES,
    verbose: bool = True,
) -> List[str]:
    """Generate cinematic image prompts via scene plan pipeline.

    This is the legacy entry point preserved for backward compatibility.
    Internally it now runs the full scene-plan → cinematic-prompt pipeline.
    """
    plan = generate_scene_plan(
        topic,
        script_text,
        provider=provider,
        model=model,
        script_file=script_file,
        count=count,
        verbose=verbose,
    )
    return scene_plan_to_prompts(
        plan,
        topic,
        provider=provider,
        model=model,
        script_file=script_file,
        verbose=verbose,
    )
