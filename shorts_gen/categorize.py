"""Topic/script -> background category routing.

Categories and keywords live in `categories.json` so non-coders can tweak them.
Matching is case-insensitive whole-word; per-category `weight` lets you favour
specific niches when they overlap (e.g. "Python" hits coding, not tech).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "categories.json"


@dataclass
class CategoryConfig:
    fallback: List[str]
    categories: Dict[str, Dict]  # name -> {weight, keywords}

    @classmethod
    def load(cls, path: Path | str | None = None) -> "CategoryConfig":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        data = json.loads(p.read_text(encoding="utf-8"))
        cats = data.get("categories", {})
        # Drop comment keys and normalize.
        cats = {k: v for k, v in cats.items() if not k.startswith("_")}
        return cls(fallback=data.get("fallback", ["general"]), categories=cats)


def _compile_patterns(keywords: List[str]) -> List[re.Pattern]:
    pats = []
    for kw in keywords:
        kw = kw.strip().lower()
        if not kw:
            continue
        # \b doesn't play nice with punctuation/dots, so we anchor on
        # non-word boundaries manually.
        esc = re.escape(kw)
        pats.append(re.compile(rf"(?<![\w]){esc}(?![\w])", re.IGNORECASE))
    return pats


def score_categories(text: str, cfg: CategoryConfig) -> List[Tuple[str, float, int]]:
    """Return [(category, score, hit_count)] sorted high-to-low."""
    haystack = text.lower()
    results: List[Tuple[str, float, int]] = []
    for name, spec in cfg.categories.items():
        weight = float(spec.get("weight", 1.0))
        hits = 0
        for pat in _compile_patterns(spec.get("keywords", [])):
            hits += len(pat.findall(haystack))
        results.append((name, hits * weight, hits))
    # Preserve declaration order for ties.
    results.sort(key=lambda r: r[1], reverse=True)
    return results


def select_category(
    topic: str,
    script: str,
    backgrounds_root: Path,
    cfg: CategoryConfig | None = None,
) -> Tuple[str, List[Tuple[str, float, int]]]:
    """Pick the best category folder that actually exists and has clips."""
    cfg = cfg or CategoryConfig.load()
    # Topic is weighted heavier than script (mentioned twice).
    blob = f"{topic}\n{topic}\n{script}"
    ranked = score_categories(blob, cfg)

    def _is_usable(name: str) -> bool:
        d = backgrounds_root / name
        if not d.is_dir():
            return False
        return any(
            p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"} for p in d.iterdir()
        )

    # Highest-scoring category with hits AND usable folder.
    for name, score, hits in ranked:
        if hits > 0 and _is_usable(name):
            return name, ranked

    # Otherwise walk the fallback list.
    for name in cfg.fallback:
        if _is_usable(name):
            return name, ranked

    # Last resort: any usable category folder.
    for name, _, _ in ranked:
        if _is_usable(name):
            return name, ranked

    raise FileNotFoundError(
        f"No usable background folders under {backgrounds_root}. "
        f"Add some .mp4 files into one of: {', '.join(cfg.categories)}."
    )
