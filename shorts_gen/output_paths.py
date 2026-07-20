"""Timestamped output-path management for yt-gen.

Responsibilities
----------------
* Build a unique filename stem from the current wall-clock time and a
  sanitised topic slug:  ``YYYY-MM-DD_HH-MM-SS_topic-slug``
* Create the four permanent asset folders on first use:
    output/videos      – final MP4 files
    output/audio       – voiceover MP3 files
    output/subtitles   – ASS subtitle files
    output/images      – per-run image sub-folders
* Expose an ``OutputPaths`` dataclass that carries every resolved path
  used by the pipeline so ``main.py`` never has to build a path string
  by hand after the first call.
* Write the convenience copy ``output/short.mp4`` so the "latest run"
  is always a fixed, predictable location.

Nothing in this module imports from any other shorts_gen module, so it
can be imported first without circular-dependency risk.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

# Characters that are illegal in Windows filenames (beyond the usual / \ :)
_WIN_ILLEGAL = re.compile(r'[\\/:*?"<>|]')
# Any run of whitespace -> single hyphen
_SPACE_RUN = re.compile(r'\s+')
# Anything that is not a letter, digit, hyphen, or underscore
_NON_SLUG = re.compile(r'[^A-Za-z0-9\-_]')
# Collapse consecutive hyphens that appear after the previous two subs
_MULTI_HYPHEN = re.compile(r'-{2,}')

_SLUG_MAX = 50
_FALLBACK_SLUG = "short"


def _make_slug(topic: str, max_len: int = _SLUG_MAX) -> str:
    """Return a Windows-safe, URL-safe lowercase slug from *topic*.

    Steps applied in order:
    1. Strip Windows-illegal characters (\\\\  /  :  *  ?  "  <  >  |).
    2. Replace whitespace runs with a single hyphen.
    3. Remove every character that is not alphanumeric, hyphen, or underscore.
    4. Collapse consecutive hyphens into one.
    5. Strip leading/trailing hyphens and convert to lowercase.
    6. Truncate to *max_len* characters; strip any trailing hyphen produced
       by the truncation.
    7. Fall back to ``"short"`` if the result is empty.
    """
    s = _WIN_ILLEGAL.sub("", topic)
    s = _SPACE_RUN.sub("-", s.strip())
    s = _NON_SLUG.sub("", s)
    s = _MULTI_HYPHEN.sub("-", s)
    s = s.strip("-").lower()
    if not s:
        return _FALLBACK_SLUG
    s = s[:max_len].rstrip("-")
    return s or _FALLBACK_SLUG


def make_stem(topic: str, *, ts: datetime | None = None) -> str:
    """Return ``YYYY-MM-DD_HH-MM-SS_<slug>`` for *topic*.

    Pass *ts* explicitly in tests to get a deterministic result; leave it
    ``None`` in production so it defaults to ``datetime.now()``.
    """
    stamp = (ts or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{stamp}_{_make_slug(topic)}"


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

def _ensure_dirs(root: Path) -> dict[str, Path]:
    """Create and return the four permanent output sub-directories."""
    dirs = {
        "videos":    root / "videos",
        "audio":     root / "audio",
        "subtitles": root / "subtitles",
        "images":    root / "images",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# OutputPaths dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutputPaths:
    """All filesystem paths for a single pipeline run.

    Attributes
    ----------
    stem            Unique filename stem, e.g. ``2026-06-15_09-42-18_openai-agent``
    video           Final MP4: ``output/videos/<stem>.mp4``
    audio           Voiceover MP3: ``output/audio/<stem>.mp3``
    subtitles       Subtitle ASS: ``output/subtitles/<stem>.ass``
    images_dir      Per-run image folder: ``output/images/<stem>/``
    work            Temp scratch folder: ``output/_work/``
    latest          Convenience copy: ``output/short.mp4``
    quality_report  Per-run quality report: ``output/videos/<stem>_quality_report.json``
    history         Global generation history: ``output/history.json``
    """

    stem:           str
    video:          Path
    audio:          Path
    subtitles:      Path
    images_dir:     Path
    work:           Path
    latest:         Path
    quality_report: Path
    history:        Path


def make_output_paths(topic: str, out_root: Path | str = "output") -> OutputPaths:
    """Build and create all output paths for *topic*.

    Parameters
    ----------
    topic:
        The raw topic string passed by the user (e.g. ``"OpenAI agent launch"``).
    out_root:
        The top-level output directory (default: ``output/`` relative to CWD).
        Created automatically if it does not exist.

    Returns
    -------
    OutputPaths
        A frozen dataclass holding every resolved path this run will use.
    """
    root = Path(out_root)
    dirs = _ensure_dirs(root)

    # Temp scratch dir for script.txt, image_prompts.txt, and any other
    # intermediate files that do not need permanent storage.
    work = root / "_work"
    work.mkdir(parents=True, exist_ok=True)

    stem = make_stem(topic)

    return OutputPaths(
        stem=stem,
        video=dirs["videos"]        / f"{stem}.mp4",
        audio=dirs["audio"]         / f"{stem}.mp3",
        subtitles=dirs["subtitles"] / f"{stem}.ass",
        images_dir=dirs["images"]   / stem,
        work=work,
        latest=root / "short.mp4",
        quality_report=dirs["videos"] / f"{stem}_quality_report.json",
        history=root / "history.json",
    )


# ---------------------------------------------------------------------------
# Post-render helpers
# ---------------------------------------------------------------------------

def write_latest_copy(video_path: Path, latest_path: Path) -> None:
    """Overwrite ``output/short.mp4`` with *video_path* as a convenience copy.

    Uses ``shutil.copy2`` so file metadata is preserved.  Silently skips if
    *video_path* does not exist (e.g. a dry-run or test that skipped FFmpeg).
    """
    if not video_path.is_file():
        return
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, latest_path)


def print_summary(paths: OutputPaths, *, visual_mode: str = "image") -> None:
    """Print the final asset paths in a consistent, readable format."""
    print(f"\nDone    -> {paths.video}")
    print(f"Audio   -> {paths.audio}")
    print(f"Subs    -> {paths.subtitles}")
    if visual_mode == "image":
        print(f"Images  -> {paths.images_dir}/")
    print(f"Latest  -> {paths.latest}  (convenience copy)")
