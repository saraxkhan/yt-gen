"""Pluggable image sourcing for the image visual mode.

Backends — all free, no external API calls required:
  1. file         - reuse existing image files from backgrounds/ or --images-dir
                    (default; works offline with zero dependencies)
  2. placeholder  - generate solid-colour gradient PNGs using stdlib only
                    (ultimate offline fallback, no files needed)
  3. screenshot   - capture a web page screenshot via a headless browser
                    (optional; requires playwright: `pip install playwright`)

Former paid/rate-limited providers (pollinations, huggingface, openai) have
been removed because:
  - pollinations returns HTTP 402 Payment Required on current endpoints
  - huggingface and openai require API keys and incur costs

The pipeline tries providers in this waterfall order when the default fails:
  file -> (images found in backgrounds/) -> placeholder

Each backend returns a list of saved image paths that feed the Ken Burns
compositor in video.py.

Caching: images are saved with a prompt-hash filename so the same prompt
is never re-fetched/re-generated twice.
"""
from __future__ import annotations

import hashlib
import os
import re
import struct
import zlib
from pathlib import Path
from typing import List, Protocol

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Target vertical render size for 9:16 Shorts (used by placeholder generator)
IMG_WIDTH = 1080
IMG_HEIGHT = 1920

IMAGE_PROVIDERS = ("file", "placeholder", "screenshot")


class ImageProvider(Protocol):
    def generate(self, prompts: List[str], out_dir: Path) -> List[Path]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt_hash(prompt: str) -> str:
    """Short hash used for cache filenames."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _find_cached(out_dir: Path, prompt: str) -> Path | None:
    """Return an existing cached image for *prompt*, or None."""
    for ext in IMAGE_EXTS:
        p = out_dir / f"img_{_prompt_hash(prompt)}{ext}"
        if p.exists():
            return p
    return None


def _find_images_in_tree(root: Path) -> list[Path]:
    """Recursively find all image files under *root*."""
    found: list[Path] = []
    if not root.is_dir():
        return found
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            found.append(p)
    return sorted(found)


def _semantic_score(prompt: str, image_path: Path) -> int:
    """Score how well *image_path* matches *prompt* using filename/folder keyword overlap.

    Uses the image filename and its parent folder names as a searchable string.
    Returns an integer score (0 = no overlap, higher = better match).
    """
    # Build a searchable string from path components
    parts = [image_path.stem]  # filename without extension
    for parent in image_path.parts[:-1]:  # folder names
        parts.append(parent)
    searchable = " ".join(parts).lower()
    searchable = re.sub(r"[_\-]", " ", searchable)  # split snake/kebab

    # Tokenise the prompt into meaningful words (3+ chars, no stop words)
    _STOP = {"the", "and", "for", "with", "from", "this", "that", "are",
             "was", "will", "its", "into", "via", "has", "have", "been"}
    tokens = [
        w for w in re.findall(r"[a-z]{3,}", prompt.lower())
        if w not in _STOP
    ]
    if not tokens:
        return 0

    score = sum(1 for t in tokens if t in searchable)
    return score


# ---------------------------------------------------------------------------
# Placeholder generator — stdlib only, no dependencies
# ---------------------------------------------------------------------------

def _make_placeholder_png(r: int, g: int, b: int, w: int = IMG_WIDTH, h: int = IMG_HEIGHT) -> bytes:
    """Create a minimal valid PNG with a top-to-bottom dark gradient."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    rows = bytearray()
    for y in range(h):
        rows.append(0)  # filter type None
        t = y / max(h - 1, 1)
        pr = int(r * 0.3 + r * 0.7 * t)
        pg = int(g * 0.3 + g * 0.7 * t)
        pb = int(b * 0.3 + b * 0.7 * t)
        for _ in range(w):
            rows += bytes([pr, pg, pb])
    compressed = zlib.compress(bytes(rows), 6)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


_PALETTE = [
    (15, 20, 40),   # deep blue
    (20, 10, 35),   # deep purple
    (10, 30, 25),   # dark teal
    (30, 15, 10),   # dark red
    (10, 25, 35),   # midnight blue
    (25, 20, 10),   # dark amber
]


def _make_placeholder_images(n: int, out_dir: Path) -> list[Path]:
    """Write *n* placeholder PNG files to *out_dir* and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n):
        r, g, b = _PALETTE[i % len(_PALETTE)]
        p = out_dir / f"placeholder_{i:02d}.png"
        if not p.exists():
            p.write_bytes(_make_placeholder_png(r, g, b))
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Provider 1: file  (default)
# ---------------------------------------------------------------------------

class FileImageProvider:
    """Sources images from a local folder — zero network calls, fully offline.

    Resolution order:
      1. --images-dir if supplied explicitly
      2. backgrounds/ tree next to the project root (auto-discovered)
      3. Falls back to PlaceholderImageProvider if nothing is found
    """

    def __init__(self, images_dir: str | os.PathLike | None = None, *, project_root: Path | None = None):
        self._explicit_dir = Path(images_dir) if images_dir else None
        self._project_root = project_root or Path(__file__).resolve().parent.parent

    def _candidates(self) -> list[Path]:
        if self._explicit_dir:
            if not self._explicit_dir.is_dir():
                raise FileNotFoundError(f"Images folder not found: {self._explicit_dir}")
            found = _find_images_in_tree(self._explicit_dir)
            if not found:
                raise FileNotFoundError(f"No image files in {self._explicit_dir}")
            return found
        # Auto-discover from backgrounds/
        bg_root = self._project_root / "backgrounds"
        found = _find_images_in_tree(bg_root)
        return found  # may be empty — caller falls back to placeholder

    def generate(self, prompts: List[str], out_dir: Path) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            candidates = self._candidates()
        except FileNotFoundError:
            candidates = []

        if not candidates:
            print("  No local images found in backgrounds/ — using placeholders.")
            return PlaceholderImageProvider().generate(prompts, out_dir)

        n = len(prompts)
        paths: List[Path] = []
        print(f"  Matching {n} scene(s) to library ({len(candidates)} images available)…")

        for i, prompt in enumerate(prompts):
            # Score all candidates against this prompt
            scored = sorted(
                candidates,
                key=lambda p: _semantic_score(prompt, p),
                reverse=True,
            )
            best = scored[0]
            match_score = _semantic_score(prompt, best)
            if match_score > 0:
                print(f"    scene {i+1}: semantic match '{best.parent.name}/{best.name}' (score={match_score})")
            else:
                # No keyword match — cycle through candidates so visuals vary
                best = candidates[i % len(candidates)]
                print(f"    scene {i+1}: no keyword match, using '{best.parent.name}/{best.name}'")
            paths.append(best)

        return paths


# ---------------------------------------------------------------------------
# Provider 2: placeholder  (guaranteed offline fallback)
# ---------------------------------------------------------------------------

class PlaceholderImageProvider:
    """Generates solid-colour gradient PNGs using stdlib only.

    Never fails. No network, no dependencies beyond Python itself.
    """

    def generate(self, prompts: List[str], out_dir: Path) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        n = len(prompts)
        print(f"  Generating {n} placeholder image(s)...")
        return _make_placeholder_images(n, out_dir)


# ---------------------------------------------------------------------------
# Provider 3: screenshot  (optional, requires playwright)
# ---------------------------------------------------------------------------

class ScreenshotImageProvider:
    """Captures full-page screenshots via a headless Chromium browser.

    Requires:  pip install playwright && playwright install chromium

    Each prompt is URL-encoded and passed to a simple search-results page
    (DuckDuckGo HTML endpoint — no JS required) so that the screenshot at
    least relates to the topic rather than being a blank page.
    """

    _SEARCH_URL = "https://html.duckduckgo.com/html/?q={query}"

    def __init__(self, width: int = IMG_WIDTH, height: int = IMG_HEIGHT):
        self.width = width
        self.height = height

    def _take_screenshot(self, prompt: str, out_path: Path) -> None:
        try:
            from playwright.sync_api import sync_playwright  # lazy
        except ImportError as exc:
            raise RuntimeError(
                "playwright is not installed. Run: pip install playwright && playwright install chromium\n"
                "Or use --image-provider file (default, no dependencies)."
            ) from exc
        import urllib.parse
        url = self._SEARCH_URL.format(query=urllib.parse.quote_plus(prompt[:200]))
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": self.width, "height": self.height})
            try:
                page.goto(url, timeout=20_000, wait_until="domcontentloaded")
                page.screenshot(path=str(out_path), full_page=False)
            finally:
                browser.close()

    def generate(self, prompts: List[str], out_dir: Path) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        n = len(prompts)
        for i, prompt in enumerate(prompts):
            print(f"  Generating image {i + 1}/{n} (screenshot)...")
            cached = _find_cached(out_dir, prompt)
            if cached:
                print(f"    -> reusing cached {cached.name}")
                paths.append(cached)
                continue
            out_path = out_dir / f"img_{_prompt_hash(prompt)}.png"
            try:
                self._take_screenshot(prompt, out_path)
                paths.append(out_path)
            except Exception as e:
                print(f"    screenshot failed ({e}); using placeholder")
                ph = _make_placeholder_images(1, out_dir)
                # rename so it doesn't collide with other placeholders
                dest = out_dir / f"img_{_prompt_hash(prompt)}_ph.png"
                if ph:
                    ph[0].rename(dest)
                    paths.append(dest)
        return paths


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_image_provider(
    name: str,
    *,
    images_dir: str | None = None,
    model: str | None = None,  # kept for API compatibility, unused
) -> ImageProvider:
    name = (name or "file").lower()
    if name == "file":
        return FileImageProvider(images_dir)
    if name == "placeholder":
        return PlaceholderImageProvider()
    if name == "screenshot":
        return ScreenshotImageProvider()
    raise ValueError(
        f"Unknown image provider '{name}'. Choose one of: {', '.join(IMAGE_PROVIDERS)}"
    )


def generate_images(
    prompts: List[str],
    out_dir: Path,
    *,
    provider: str = "file",
    images_dir: str | None = None,
    model: str | None = None,
) -> List[Path]:
    """Source images for the given prompts using the chosen backend.

    Waterfall on failure:
      chosen provider -> file (backgrounds/) -> placeholder
    """
    prov = get_image_provider(provider, images_dir=images_dir, model=model)
    try:
        paths = prov.generate(prompts, out_dir)
        if paths:
            return paths
    except Exception as e:
        print(f"\n  Image provider '{provider}' failed: {e}")

    # Waterfall: try file provider (backgrounds/) before giving up
    if not isinstance(prov, FileImageProvider):
        print("  Trying file provider (backgrounds/)...")
        try:
            file_paths = FileImageProvider().generate(prompts, out_dir)
            if file_paths:
                return file_paths
        except Exception as e2:
            print(f"  File provider also failed: {e2}")

    # Final fallback: placeholders always work
    print("  Falling back to placeholder images...")
    return PlaceholderImageProvider().generate(prompts, out_dir)
