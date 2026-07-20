"""Word-level karaoke subtitles (faster-whisper → ASS file).

Caption placement improvements:
- Captions sit in the lower-third safe zone (bottom 25% of frame)
  using ASS Alignment=2 (bottom-centre) + generous MarginV.
- This guarantees captions never cover the main subject, which almost
  always appears in the upper-centre of a 9:16 vertical AI news image.
- Pop-scale animation is preserved for TikTok-style visual energy.
- Bold, high-contrast white text with black outline for readability on
  any background colour.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Caption safe-zone constants (1080 × 1920 canvas)
# ---------------------------------------------------------------------------

# MarginV in ASS pixels: how far from the BOTTOM of the frame.
# 1920 * 0.12 ≈ 230 → captions appear in the bottom 12–25% of the frame.
CAPTION_MARGIN_V: int = 230

# MarginL / MarginR push the text away from the vertical edges.
CAPTION_MARGIN_H: int = 80


@dataclass
class Word:
    start: float
    end: float
    text: str


def transcribe_words(audio_path: Path, model_size: str = "small") -> List[Word]:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(audio_path), word_timestamps=True, vad_filter=True
    )
    words: List[Word] = []
    for seg in segments:
        for w in seg.words or []:
            txt = (w.word or "").strip()
            if not txt:
                continue
            words.append(Word(start=float(w.start), end=float(w.end), text=txt))
    return words


def _fmt_ts(t: float) -> str:
    if t < 0:
        t = 0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# ASS header — lower-third safe-zone placement
# ---------------------------------------------------------------------------
#
# Key ASS alignment values:
#   1=bottom-left, 2=bottom-centre, 3=bottom-right
#   4=middle-left, 5=middle-centre, 6=middle-right
#   7=top-left,    8=top-centre,    9=top-right
#
# We use Alignment=2 (bottom-centre) and set MarginV so captions land
# comfortably in the lower-third of the 1080×1920 canvas, well away from
# the centre-frame AI imagery.
#
# The "Pop" style parameters:
#   FontSize 105  – large enough to be readable on mobile but not overwhelming
#   Bold 1        – always bold for contrast
#   Outline 7     – thick black outline for readability on any background
#   Shadow 3      – soft drop shadow for depth
#   PrimaryColour &H00FFFFFF – white text
#   OutlineColour &H00000000 – black outline
#   BackColour    &H80000000 – semi-transparent background (50% black)
#   BorderStyle 1 – outline + shadow (not opaque box)

ASS_HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Pop,Impact,105,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,7,3,2,{CAPTION_MARGIN_H},{CAPTION_MARGIN_H},{CAPTION_MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass(
    words: List[Word],
    out_path: Path,
    words_per_caption: int = 2,
) -> Path:
    """Write an ASS subtitle file with word-by-word karaoke captions.

    Captions are placed in the lower-third safe zone (Alignment=2, MarginV),
    ensuring they never cover the central subject of the 9:16 imagery.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Group words into chunks of `words_per_caption`
    chunks: List[List[Word]] = [
        words[i : i + words_per_caption]
        for i in range(0, len(words), words_per_caption)
    ]

    lines = [ASS_HEADER]
    for chunk in chunks:
        if not chunk:
            continue
        start = chunk[0].start
        end   = chunk[-1].end
        # Ensure minimum display time so captions don't flicker
        end   = max(end, start + 0.35)

        text = " ".join(w.text for w in chunk).upper().replace("\n", " ")
        # ASS escape: braces are control characters in ASS
        text = text.replace("{", "(").replace("}", ")")

        # Pop-scale animation: scale up on entry, settle back
        # \an2  = alignment override (bottom-centre, redundant but explicit)
        # \fad  = soft fade in/out so text doesn't hard-cut
        # \t    = tween for the pop scale effect
        effect = r"{\an2\fad(60,60)\t(0,100,\fscx112\fscy112)\t(100,200,\fscx100\fscy100)}"
        lines.append(
            f"Dialogue: 0,{_fmt_ts(start)},{_fmt_ts(end)},"
            f"Pop,,0,0,0,,{effect}{text}"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
