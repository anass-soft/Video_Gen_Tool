# AI Short Video Generator — Setup & Usage

A Tkinter desktop app that turns a **title** into a **short narrated video** using:

* **Gemini** for script + scene prompts
* **Imagen (Gemini API)** for images (with **Hugging Face** fallback)
* **ElevenLabs** for voice-over
* **MoviePy / FFmpeg** for video assembly
* **Captions**: burn-in (on the frames) + `.srt` file
* **Proportional timing** so captions line up better with speech

---

## 1) Requirements

* **Python** 3.9–3.12
* **FFmpeg** on your system `PATH`
* API keys/tokens:

  * **Google AI Studio** API key (for Gemini text)
  * **ElevenLabs** API key + **Voice ID**
  * **(Optional)** **Hugging Face** fine-grained token with **Inference Providers** permission (image fallback)

---

## 2) Install

### A) Get the code

Put the single file in a folder:

```
video_tool.py
```

### B) Create a virtual env (recommended)

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### C) Install Python dependencies

```bash
python -m pip install google-generativeai moviepy pillow requests
# Optional (for Arabic burn-in shaping):
python -m pip install arabic-reshaper python-bidi
```

### D) Install FFmpeg

* **Windows**: Download from ffmpeg.org or `choco install ffmpeg` (Chocolatey). Ensure `ffmpeg.exe` is on PATH.
* **macOS**: `brew install ffmpeg`
* **Linux**: `sudo apt-get install ffmpeg` (or your distro’s package manager)

Verify:

```bash
ffmpeg -version
```

---

## 3) Run

```bash
python video_tool.py
```

Fill the GUI fields:

* **Google (Gemini) API Key** – from [ai.google.dev](https://ai.google.dev) (AI Studio key)
* **Hugging Face Token (fallback)** – from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

  * Must be **Fine-grained** and enable **Make calls to Inference Providers** ✅
* **ElevenLabs API Key** – from [elevenlabs.io](https://elevenlabs.io)
* **ElevenLabs Voice ID** – pick a voice in your ElevenLabs dashboard and copy the ID
* **Video Title** – your topic/title
* **# Images** – frames count (each frame = one caption chunk)
* **Output Video Path** – where to save `.mp4`
* **Burn-in captions** – draw captions on images
* **Save SRT subtitle file** – create `.srt` next to the video

Click **Generate Video**.

---

## 4) How it works (pipeline)

1. **Script** → Gemini (text) writes 4–6 short paragraphs
2. **Image prompts** → Gemini converts the script to N scene prompts
3. **Images**

   * Tries **Imagen** (requires Google billing)
   * If blocked, falls back to **Hugging Face Inference API** (e.g., `black-forest-labs/FLUX.1-schnell`)
4. **TTS** → ElevenLabs generates narration `.mp3`
5. **Captions**

   * Script split into N chunks
   * **Durations proportional to word counts** (align better with speech)
   * Optional **burn-in** onto frames + **SRT** export
6. **Video** → MoviePy stitches frames using the same per-chunk durations and overlays the narration

---

## 5) Configuration tips

* **Portrait videos (Shorts/Reels)**
  In `video_tool.py`, set:

  ```python
  VIDEO_WIDTH, VIDEO_HEIGHT = 1080, 1920
  ```
* **Arabic captions**
  Install `arabic-reshaper` and `python-bidi` (app will auto-use them).
* **Hugging Face token via env var**
  Optionally set once:

  * Windows (PowerShell):

    ```powershell
    [System.Environment]::SetEnvironmentVariable("HF_TOKEN","hf_xxx","User")
    ```
  * macOS/Linux:

    ```bash
    echo 'export HF_TOKEN=hf_xxx' >> ~/.bashrc   # or ~/.zshrc
    source ~/.bashrc
    ```

  The app will read `HF_TOKEN` if the GUI field is empty.

---

## 6) Troubleshooting

### “ModuleNotFoundError: No module named 'google'”

Install Gemini SDK into the **same** Python env:

```bash
python -m pip install google-generativeai
```

### Imagen error: “only accessible to billed users”

* Enable **billing** on your Google AI Studio project **or**
* Use the **Hugging Face fallback**: create a fine-grained token and enable
  **Make calls to Inference Providers** under token permissions.

### Hugging Face error about “Providers” / 403

Create a **Fine-grained** token with:

* **Repositories → Read**
* **Inference → Use the Inference API**
* **Inference Providers → Allow calling Inference Providers** ✅

### Pillow text measurement error

This project includes a compatibility patch (uses `textbbox`/`textsize` safely).
If you still see drawing issues, upgrade Pillow:

```bash
python -m pip install --upgrade "Pillow>=10.0"
```

### FFmpeg not found / MoviePy error

Ensure `ffmpeg` is installed and on your system `PATH`.
Reopen the terminal after installation.

### Captions out of sync

Captions use **word proportional timing**. If you want *perfect* alignment:

* Use ElevenLabs timestamps (if your plan exposes them), or
* Add a forced alignment step (e.g., **aeneas**). Ask and I can wire it in.

### Multiple Python installs

Use `python -m pip ...` to guarantee you install into the same interpreter you run.

---

## 7) Security & keys

* Treat API keys/tokens like passwords.
* Don’t commit them to source control.
* Prefer environment variables where possible.

---

## 8) Optional enhancements

* **Background music** mix (ducked under narration)
* **Per-word karaoke captions**
* **Batch render** multiple titles
* **Asset caching** to skip regeneration
* **Theme/style presets** for different genres

> If you want any of these, say which one and I’ll provide the patch.

---

## 9) License

You can use this project in your own work. If you need a formal license, MIT is a good default.
