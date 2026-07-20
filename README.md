#Automated yt

Give it a topic, get back an upload-ready 1080x1920 MP4 — voiceover,
karaoke captions, and visuals sourced from your local `backgrounds/` library.
Fully offline by default. No API keys, no paid services required.

**Anti-hallucination**: scripts are strictly grounded in the topic you provide.
The LLM is forbidden from inventing product names, company names, or facts.
A self-review loop detects and rewrites hallucinated content automatically.

Two visual modes:

- **`--visual-mode image`** (default, recommended) — generate 4-6
  topic-specific AI images from the script, apply Ken Burns zoom/pan and
  crossfade transitions that track the narration.
- **`--visual-mode video`** — original mode: route to a category folder of
  background footage and crop/loop it. Unchanged fallback.

Pipeline:

1. **Script** — pluggable LLM (Ollama by default, fully offline).
2. **Visuals**
   - *image mode:* derive image prompts from the script, generate real AI
     images via Pollinations (default, free) or HuggingFace / OpenAI,
     Ken Burns + crossfade them. Images are cached by prompt hash.
   - *video mode:* analyze topic+script, pick the best background category
     folder using a keyword config you can edit without touching code.
3. **Voiceover** — `edge-tts` (free Microsoft voices, no key).
4. **Subtitles** — word-level karaoke captions via `faster-whisper`.
5. **Video** — FFmpeg composes to 1080x1920 and burns subs.

## Visual modes

```bash
# Image mode (default): AI images + Ken Burns. Needs an image backend below.
python -m shorts_gen.main "ChatGPT just got a huge upgrade"

# Background-footage mode (fallback)
python -m shorts_gen.main "ChatGPT just got a huge upgrade" --visual-mode video
```

### Image backends (`--image-provider`)

All providers are **free and require no API keys**. Paid/rate-limited
providers (pollinations, huggingface, openai) have been removed.

| Provider | Needs files | Needs network | Notes |
|---|---|---|---|
| `file` | ✅ `backgrounds/` or `--images-dir` | ❌ | **Default.** Reuses your local images |
| `placeholder` | ❌ | ❌ | Solid-colour gradient PNGs, stdlib only |
| `screenshot` | ❌ | ✅ | Headless browser (needs `playwright`) |

**Waterfall fallback**: if the chosen provider fails, the pipeline
automatically tries `file` (backgrounds/) then `placeholder` — the video
is always generated, never crashes on missing images.

```bash
# Default: use images from backgrounds/  (drop images into those folders)
python -m shorts_gen.main "OpenAI launches a new AI agent"

# Use your own image folder
python -m shorts_gen.main "OpenAI launches a new AI agent" \
  --image-provider file --images-dir my_images/

# Pure offline, zero files needed
python -m shorts_gen.main "OpenAI launches a new AI agent" \
  --image-provider placeholder

# Headless screenshot (requires: pip install playwright && playwright install chromium)
python -m shorts_gen.main "OpenAI launches a new AI agent" \
  --image-provider screenshot
```


## Install

```bash
pip install -r requirements.txt
# FFmpeg must be installed:
#   macOS:  brew install ffmpeg
#   Ubuntu: sudo apt install ffmpeg
```

### Ollama (default LLM)

```bash
# https://ollama.com/download
ollama pull llama3.1   # or qwen2.5, gemma2, mistral, ...
```

## Background footage

Drop clips into the matching category folder:

```
backgrounds/
├── ai/          AI, ChatGPT, Claude, Gemini, agents, LLMs
├── coding/      programming languages, IDEs, frameworks, dev tools
├── business/    startups, funding, acquisitions, markets
├── tech/        gadgets, phones, hardware, product launches
├── abstract/    philosophical / abstract topics
└── general/     catch-all fallback
```

Any aspect ratio works — FFmpeg crops and scales to 1080x1920. If the clip is
shorter than the voiceover it loops automatically.

### Editing categories — no code changes needed

All routing lives in **`categories.json`** at the project root:

```jsonc
{
  "fallback": ["abstract", "general"],
  "categories": {
    "ai":     { "weight": 1.0, "keywords": ["ai", "chatgpt", "llm", ...] },
    "coding": { "weight": 1.0, "keywords": ["python", "javascript", ...] },
    ...
  }
}
```

- Add a category: create a new key + a matching `backgrounds/<name>/` folder.
- Add keywords: just edit the array. Whole-word, case-insensitive.
- Bias a category: bump its `weight` (each hit = `weight` points).
- `fallback` lists the folders to try when no keywords match.

## Run

```bash
# Default: Ollama + auto-pick category
python -m shorts_gen.main "ChatGPT just got a huge upgrade"
# -> [2/5] category (auto): ai

# Force a category
python -m shorts_gen.main "any topic" --category coding

# Custom config location
python -m shorts_gen.main "..." --categories-config my-categories.json

# OpenAI-compatible API (optional)
export OPENAI_API_KEY=sk-...
python -m shorts_gen.main "Black hole basics" --llm-provider openai --model gpt-4o-mini

# Bring your own script
python -m shorts_gen.main "label" --script-file my_script.txt
```

### All flags

```
--visual-mode {image,video}          default: image (recommended)
--llm-provider {ollama,openai,file}   default: ollama
--model NAME                          provider-specific model id
--script-file PATH                    text file (implies --llm-provider file)
--image-provider {sd,openai,file}     default: sd (image mode only)
--images-dir PATH                     folder of images for --image-provider file
--image-model NAME                    image model id (provider-specific)
--num-images 6                        images per Short (4-6, image mode)
--category NAME                       force background category (video mode)
--categories-config PATH              custom categories.json (video mode)
--backgrounds backgrounds             root folder of category subfolders (video mode)
--voice en-US-GuyNeural               any edge-tts voice
--duration 45                         target spoken length in seconds
--whisper-model small                 tiny | base | small | medium
--out output/short.mp4
```

## Env vars

- `OLLAMA_HOST` (default `http://localhost:11434`), `OLLAMA_MODEL`
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_IMAGE_MODEL`
- `SD_HOST` (default `http://localhost:7860`), `SD_STEPS`, `SD_CFG`, `SD_SAMPLER`

## Output

`output/short.mp4` — 1080x1920 H.264 + AAC, ready to upload.
