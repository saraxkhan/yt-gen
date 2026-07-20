"""Offline voiceover synthesis via edge-tts."""
from __future__ import annotations

import asyncio
from pathlib import Path

import edge_tts

DEFAULT_VOICE = "en-US-GuyNeural"


async def _synthesize(text: str, voice: str, out_path: Path) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def synthesize(text: str, out_path: Path, voice: str = DEFAULT_VOICE) -> Path:
    """Write an MP3 voiceover of `text` to `out_path`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # asyncio.run() is the canonical way since Python 3.7.  On Python 3.12+
    # the legacy get_event_loop() path is gone, so we always use run().
    # We guard against the case where a caller has already started a loop
    # (e.g. Jupyter), falling back to nest_asyncio or a new thread if needed.
    try:
        asyncio.run(_synthesize(text, voice, out_path))
    except RuntimeError as _e:
        if "This event loop is already running" not in str(_e):
            raise
        # Fallback for environments with an already-running event loop.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(asyncio.run, _synthesize(text, voice, out_path))
            fut.result()
    return out_path
