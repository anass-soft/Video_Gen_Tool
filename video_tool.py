# -*- coding: utf-8 -*-
"""
AI Short Video Generator — Gemini (text) + Imagen (images, with HF fallback) + ElevenLabs + MoviePy
Captions (burn-in + SRT) with **proportional timing** aligned to speech length.

Requires: google-generativeai, moviepy, pillow, requests
Optional (Arabic burn-in): arabic-reshaper, python-bidi
"""

import os
import io
import json
import base64
import threading
import traceback
from typing import List, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont
from tkinter import (
    Tk, Label, Entry, Button, Text, END, StringVar, IntVar, BooleanVar,
    filedialog, messagebox, Checkbutton
)
from tkinter.ttk import Combobox

# MoviePy
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# Gemini text SDK
import google.generativeai as genai

# Optional RTL shaping (only used if installed)
try:
    import arabic_reshaper  # type: ignore
    from bidi.algorithm import get_display  # type: ignore
except Exception:
    arabic_reshaper = None
    get_display = None


# =============================
# ---------- CONFIG -----------
# =============================

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FPS = 30
FADE_SEC = 0.25
MIN_SEG_DUR = max(1.0, FADE_SEC * 2 + 0.2)   # ensure segments are long enough for fades

# Gemini text model for script & prompt generation
GEMINI_TEXT_MODEL = "gemini-1.5-flash"

# Imagen (Gemini API) REST (PRIMARY, requires billing)
IMAGEN_MODEL_PRIMARY = "imagen-4.0-generate-001"
IMAGEN_MODEL_FALLBACK = "imagen-3.0-generate-002"
IMAGEN_URL_PRIMARY = f"https://generativelanguage.googleapis.com/v1beta/models/{IMAGEN_MODEL_PRIMARY}:predict"
IMAGEN_URL_FALLBACK = f"https://generativelanguage.googleapis.com/v1beta/models/{IMAGEN_MODEL_FALLBACK}:predict"

# Hugging Face Inference API (SECONDARY, free token)
HF_DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"

# ElevenLabs REST
ELEVEN_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

SCRIPT_SYSTEM_INSTRUCTIONS = (
    "You are a concise, cinematic scriptwriter for short educational/entertainment videos. "
    "Return a clean narration (no markdown) in 5–10 short sentences, grouped into 4–6 short lines/paragraphs. "
    "Avoid scene numbering, emojis, and hashtags. Keep it PG-13."
)

IMAGE_PROMPT_INSTRUCTIONS = (
    "Given this short video script, produce N vivid scene prompts (one per line), each self-contained and "
    "photorealistic, suitable for a text-to-image model. Avoid text in images and watermarks, "
    "assume a cohesive photographic style, and 16:9 framing. Output exactly N lines, no titles or extra commentary."
)


# =============================
# --------- HELPERS -----------
# =============================

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def log_to_widget(widget: Text, msg: str):
    widget.insert(END, msg + "\n")
    widget.see(END)
    widget.update_idletasks()

def safe_filename(s: str) -> str:
    keep = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in keep else "_" for c in s)[:120].strip().replace(" ", "_")

def save_base64_png(b64: str, path: str):
    img_bytes = base64.b64decode(b64)
    with Image.open(io.BytesIO(img_bytes)) as im:
        im = im.convert("RGB")
        ensure_dir(os.path.dirname(path))
        im.save(path, format="PNG", optimize=True)

def save_bytes_png(raw: bytes, path: str):
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        ensure_dir(os.path.dirname(path))
        im.save(path, format="PNG", optimize=True)

def split_text_into_chunks(text: str, n: int) -> List[str]:
    """Split narration into N chunks by sentence boundaries (simple heuristic)."""
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if not sentences:
        return [text.strip()] + [""] * (n - 1)
    if len(sentences) <= n:
        while len(sentences) < n:
            sentences.append("")
        return sentences[:n]
    avg = max(1, len(sentences) // n)
    chunks, i = [], 0
    while i < len(sentences):
        chunks.append(" ".join(sentences[i:i+avg]).strip())
        i += avg
    while len(chunks) > n:
        chunks[-2] = (chunks[-2] + " " + chunks[-1]).strip()
        chunks.pop()
    while len(chunks) < n:
        chunks.append("")
    return chunks[:n]

def shape_if_rtl(text: str) -> str:
    """Apply Arabic shaping/BiDi if libraries are available."""
    if not text:
        return text
    has_arabic = any('\u0600' <= ch <= '\u06FF' for ch in text)
    if has_arabic and arabic_reshaper and get_display:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Try common fonts; fallback to default."""
    for name in ["Arial.ttf", "arial.ttf", "DejaVuSans.ttf", "NotoSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()

# ---- Pillow-version-safe text measurement & wrapping ----

def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    """Width/height measurement that works across Pillow versions."""
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)  # no stroke args here
        return right - left, bottom - top
    except Exception:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            return int(len(text) * font.size * 0.6), int(font.size * 1.2)

def text_wrap_lines(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Word-wrap text into lines that fit max_width (px)."""
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    words = text.split()
    if not words:
        return [""]
    lines, line = [], ""
    for w in words:
        test = (line + " " + w).strip()
        w_px, _ = measure_text(draw, test, font)
        if w_px <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines or [""]

def render_caption_on_image(
    img_path: str, caption: str, out_path: str,
    video_w: int = VIDEO_WIDTH, video_h: int = VIDEO_HEIGHT
) -> str:
    """
    Create a new (video_w x video_h) canvas, paste image center, and draw
    caption at the bottom with a translucent background.
    """
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        src_w, src_h = im.size
        scale = min(video_w / src_w, video_h / src_h)
        new_w, new_h = int(src_w * scale), int(src_h * scale)
        im_resized = im.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (video_w, video_h), (0, 0, 0))
        canvas.paste(im_resized, ((video_w - new_w) // 2, (video_h - new_h) // 2))

    overlay = Image.new("RGBA", (video_w, video_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    cap = shape_if_rtl(ccaption := caption)
    font_size = max(24, int(video_h * 0.045))
    font = load_font(font_size)

    margin_x = int(video_w * 0.06)
    margin_y = int(video_h * 0.05)
    max_text_width = video_w - 2 * margin_x

    lines = text_wrap_lines(cap, font, max_text_width)
    line_h = int(font.size * 1.3)
    text_h = len(lines) * line_h
    y0 = video_h - margin_y - text_h
    pad = 12

    # Background rectangle (semi-transparent)
    rect = (margin_x - pad, y0 - pad, video_w - margin_x + pad, video_h - margin_y + pad)
    draw.rectangle(rect, fill=(0, 0, 0, 150))

    # Draw each line, centered horizontally
    y = y0
    for line in lines:
        w_px, _ = measure_text(draw, line, font)
        x = (video_w - w_px) // 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0, 200))  # stroke supported widely
        y += line_h

    final = Image.alpha_composite(canvas.convert("RGBA"), overlay)
    ensure_dir(os.path.dirname(out_path))
    final.convert("RGB").save(out_path, "PNG", optimize=True)
    return out_path

def caption_images(image_paths: List[str], captions: List[str], out_dir: str,
                   video_w: int = VIDEO_WIDTH, video_h: int = VIDEO_HEIGHT) -> List[str]:
    ensure_dir(out_dir)
    out_paths = []
    for i, (img, cap) in enumerate(zip(image_paths, captions), start=1):
        out_img = os.path.join(out_dir, f"cap_{i:02d}.png")
        render_caption_on_image(img, cap, out_img, video_w, video_h)
        out_paths.append(out_img)
    return out_paths

# ---- Proportional timing helpers ----

def allocate_proportional_durations(chunks: List[str], total: float, min_sec: float = MIN_SEG_DUR) -> List[float]:
    """
    Allocate per-chunk durations proportional to word counts, respecting a minimum per segment.
    """
    import math
    words = [max(1, len(ch.strip().split())) for ch in chunks]
    s = sum(words)
    if s == 0:
        return [total / len(chunks)] * len(chunks)

    # initial proportional allocation
    durs = [total * w / s for w in words]

    # enforce minimums
    # 1) cap too-small segments to min_sec
    durs = [max(min_sec, d) for d in durs]

    # 2) rescale to match total (preserve relative proportions among those above min)
    sum_d = sum(durs)
    if sum_d != total:
        # If sum_d > total, shrink all but keep >= min_sec
        # If sum_d < total, distribute extra proportional to weights
        if sum_d > 0:
            scale = total / sum_d
            durs = [max(min_sec, d * scale) for d in durs]
        else:
            durs = [total / len(chunks)] * len(chunks)

    # final normalization for tiny rounding errors
    diff = total - sum(durs)
    if abs(diff) > 1e-6:
        # add/subtract from the longest segment
        idx = max(range(len(durs)), key=lambda i: durs[i])
        durs[idx] += diff
    return durs

def format_timestamp(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt_with_durations(captions: List[str], durations: List[float], out_path: str):
    t = 0.0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (text, dur) in enumerate(zip(captions, durations), start=1):
            start, end = t, t + max(0.01, dur)
            f.write(f"{i}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text}\n\n")
            t = end


# =============================
# --------- GEMINI ------------
# =============================

def gemini_generate_script(api_key: str, title: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_TEXT_MODEL)
    prompt = f"{SCRIPT_SYSTEM_INSTRUCTIONS}\n\nTitle: {title}\nWrite the short video script now."
    resp = model.generate_content(prompt)
    text = getattr(resp, "text", None)
    if not text and hasattr(resp, "candidates") and resp.candidates:
        parts = getattr(resp.candidates[0], "content", None)
        if parts and getattr(parts, "parts", None):
            text = parts.parts[0].text
    if not text:
        raise RuntimeError("Gemini returned an empty script.")
    return text.strip()

def gemini_generate_image_prompts(api_key: str, script_text: str, n_images: int) -> List[str]:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_TEXT_MODEL)
    prompt = f"{IMAGE_PROMPT_INSTRUCTIONS}\n\nN = {n_images}\nScript:\n{script_text}"
    resp = model.generate_content(prompt)
    txt = getattr(resp, "text", None)
    if not txt and hasattr(resp, "candidates") and resp.candidates:
        parts = getattr(resp.candidates[0], "content", None)
        if parts and getattr(parts, "parts", None):
            txt = parts.parts[0].text
    if not txt:
        raise RuntimeError("Gemini returned empty image prompts.")
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    while len(lines) < n_images:
        lines.append("Cinematic natural-light wide shot, cohesive color grading, 16:9.")
    return lines[:n_images]


# =============================
# -------- IMAGEN REST --------
# =============================

def _imagen_predict(url: str, api_key: str, prompt: str, aspect_ratio: str = "16:9") -> dict:
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1, "aspectRatio": aspect_ratio}
    }
    r = requests.post(url, headers=headers, json=payload, timeout=240)
    r.raise_for_status()
    return r.json()

def _extract_imagen_base64(data: dict) -> str:
    b64 = None
    preds = data.get("predictions")
    if preds and isinstance(preds, list):
        b64 = preds[0].get("bytesBase64Encoded") or preds[0].get("imageBytes")
    if not b64 and "generatedImages" in data:
        gi0 = (data.get("generatedImages") or [{}])[0]
        b64 = gi0.get("bytesBase64Encoded") or (gi0.get("image") or {}).get("imageBytes")
    if not b64:
        raise RuntimeError(f"Unexpected Imagen response shape: {json.dumps(data)[:600]}")
    return b64

def gemini_images_generate_imagen(api_key: str, prompt: str, aspect_ratio: str = "16:9") -> str:
    try:
        data = _imagen_predict(IMAGEN_URL_PRIMARY, api_key, prompt, aspect_ratio)
        return _extract_imagen_base64(data)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code in (400, 403, 404):
            data2 = _imagen_predict(IMAGEN_URL_FALLBACK, api_key, prompt, aspect_ratio)
            return _extract_imagen_base64(data2)
        raise


# =============================
# ---- HUGGING FACE FALLBACK ---
# =============================

def hf_generate_bytes(hf_token: str, prompt: str, model: str = HF_DEFAULT_MODEL) -> bytes:
    """
    Use Hugging Face Inference API (text-to-image).
    Returns raw PNG/JPEG bytes. Requires fine-grained token with Providers permission.
    """
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Accept": "image/png",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": f"{prompt} -- 16:9 aspect, 1920x1080, high detail, cinematic photo",
        "options": {"wait_for_model": True}
    }
    r = requests.post(url, headers=headers, json=payload, timeout=300)
    ctype = r.headers.get("content-type", "")
    if r.status_code != 200 or "application/json" in ctype.lower():
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(f"Hugging Face error: {detail}")
    return r.content

def generate_images_with_fallback(gemini_key: str, hf_token: str, prompts: List[str], out_dir: str) -> List[str]:
    paths = []
    use_hf_only = False
    for i, p in enumerate(prompts, start=1):
        img_path = os.path.join(out_dir, f"image_{i:02d}.png")
        if not use_hf_only:
            try:
                b64 = gemini_images_generate_imagen(gemini_key, p, aspect_ratio="16:9")
                save_base64_png(b64, img_path)
                paths.append(img_path)
                continue
            except requests.HTTPError as e:
                msg = ""
                try:
                    msg = e.response.json().get("error", {}).get("message", "")
                except Exception:
                    pass
                if "billed users" in msg.lower() or e.response.status_code in (400, 403):
                    use_hf_only = True
                else:
                    raise
            except Exception:
                use_hf_only = True

        if not hf_token:
            raise RuntimeError("Imagen is blocked (billing). Add a Hugging Face token (Providers enabled) to use the fallback.")
        raw = hf_generate_bytes(hf_token, p, model=HF_DEFAULT_MODEL)
        save_bytes_png(raw, img_path)
        paths.append(img_path)
    return paths


# =============================
# ---------- TTS --------------
# =============================

def elevenlabs_tts(eleven_key: str, voice_id: str, text: str, out_path: str) -> str:
    url = ELEVEN_TTS_URL.format(voice_id=voice_id)
    headers = {
        "xi-api-key": eleven_key,
        "accept": "audio/mpeg",
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    with requests.post(url, headers=headers, data=json.dumps(payload), stream=True, timeout=240) as r:
        r.raise_for_status()
        ensure_dir(os.path.dirname(out_path))
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return out_path


# =============================
# --------- VIDEO -------------
# =============================

def build_video_from_assets(
    image_paths: List[str],
    audio_path: str,
    out_path: str,
    per_durations: List[float],
    video_w: int = VIDEO_WIDTH,
    video_h: int = VIDEO_HEIGHT,
    fps: int = FPS
) -> Tuple[float, str]:
    """
    Slideshow where each image uses the *provided* per-image duration.
    """
    if not image_paths:
        raise ValueError("No images to compile.")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if len(per_durations) != len(image_paths):
        raise ValueError("per_durations length must match image count.")

    audio_clip = AudioFileClip(audio_path)
    total_dur = max(2.0, audio_clip.duration)

    clips = []
    for img, d in zip(image_paths, per_durations):
        dur = max(MIN_SEG_DUR, float(d))
        clip = (
            ImageClip(img)
            .resize(height=video_h)
            .on_color(size=(video_w, video_h), color=(0, 0, 0), pos="center")
            .set_duration(dur)
            .fadein(FADE_SEC)
            .fadeout(FADE_SEC)
        )
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose").set_audio(audio_clip)
    ensure_dir(os.path.dirname(out_path) or ".")
    video.write_videofile(
        out_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="medium",
        bitrate="6000k",
        verbose=False,
        logger=None
    )
    video.close()
    audio_clip.close()
    return total_dur, out_path


# =============================
# ----------- GUI -------------
# =============================

class App:
    def __init__(self, root: Tk):
        self.root = root
        root.title("AI Short Video Generator — Gemini (Imagen + HF) with Proportional Captions")
        root.geometry("940x840")

        # Inputs
        Label(root, text="Google (Gemini) API Key:").place(x=20, y=20)
        self.gemini_key = StringVar()
        Entry(root, width=60, textvariable=self.gemini_key, show="*").place(x=280, y=20)

        Label(root, text="Hugging Face Token (fallback):").place(x=20, y=60)
        self.hf_token = StringVar()
        Entry(root, width=60, textvariable=self.hf_token, show="*").place(x=280, y=60)

        Label(root, text="ElevenLabs API Key:").place(x=20, y=100)
        self.eleven_key = StringVar()
        Entry(root, width=60, textvariable=self.eleven_key, show="*").place(x=280, y=100)

        Label(root, text="ElevenLabs Voice ID:").place(x=20, y=140)
        self.voice_id = StringVar()
        Entry(root, width=60, textvariable=self.voice_id).place(x=280, y=140)

        Label(root, text="Video Title:").place(x=20, y=180)
        self.title_var = StringVar()
        Entry(root, width=60, textvariable=self.title_var).place(x=280, y=180)

        Label(root, text="# Images:").place(x=20, y=220)
        self.n_images = IntVar(value=5)
        Combobox(root, state="readonly", width=10,
                 values=[3, 4, 5, 6, 7, 8], textvariable=self.n_images).place(x=280, y=220)

        Label(root, text="Output Video Path:").place(x=20, y=260)
        self.out_path = StringVar(value=os.path.join("output", "ai_short_video_gemini.mp4"))
        Entry(root, width=47, textvariable=self.out_path).place(x=280, y=260)
        Button(root, text="Browse…", command=self.browse_output).place(x=730, y=256)

        # Caption options
        self.burn_captions = BooleanVar(value=True)
        self.save_srt = BooleanVar(value=True)
        Checkbutton(root, text="Burn-in captions", variable=self.burn_captions).place(x=20, y=300)
        Checkbutton(root, text="Save SRT subtitle file", variable=self.save_srt).place(x=170, y=300)

        self.btn = Button(root, text="Generate Video", width=20, command=self.on_generate_clicked)
        self.btn.place(x=20, y=340)

        self.cancel_flag = False
        self.cancel_btn = Button(root, text="Cancel", width=12, command=self.on_cancel, state="disabled")
        self.cancel_btn.place(x=200, y=340)

        Label(root, text="Progress / Log:").place(x=20, y=390)
        self.log = Text(root, width=114, height=20)
        self.log.place(x=20, y=420)

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")],
            initialfile="ai_short_video_gemini.mp4"
        )
        if path:
            self.out_path.set(path)

    def set_working(self, working: bool):
        self.btn.config(state="disabled" if working else "normal")
        self.cancel_btn.config(state="normal" if working else "disabled")

    def on_cancel(self):
        self.cancel_flag = True
        self.log_msg("Cancel requested… (will stop after current step)")
        self.cancel_btn.config(state="disabled")

    def log_msg(self, msg: str):
        log_to_widget(self.log, msg)

    def _check_cancel(self):
        if self.cancel_flag:
            raise RuntimeError("Operation canceled by user.")

    def on_generate_clicked(self):
        if not self.gemini_key.get().strip():
            messagebox.showerror("Missing Gemini Key", "Please enter your Google (Gemini) API key.")
            return
        if not self.eleven_key.get().strip():
            messagebox.showerror("Missing ElevenLabs Key", "Please enter your ElevenLabs API key.")
            return
        if not self.voice_id.get().strip():
            messagebox.showerror("Missing Voice ID", "Please enter your ElevenLabs Voice ID.")
            return
        if not self.title_var.get().strip():
            messagebox.showerror("Missing Title", "Please enter a video title.")
            return

        self.cancel_flag = False
        self.set_working(True)
        self.log.delete("1.0", END)
        self.log_msg("Starting…")

        threading.Thread(target=self.run_pipeline, daemon=True).start()

    # ---------------------------
    # -------- PIPELINE ---------
    # ---------------------------
    def run_pipeline(self):
        try:
            title = self.title_var.get().strip()
            out_video = self.out_path.get().strip()
            out_dir = os.path.join("output", safe_filename(title) or "project")
            assets_dir = os.path.join(out_dir, "assets")
            ensure_dir(assets_dir)
            self.log_msg(f"Assets folder: {assets_dir}")

            # 1) Script
            self._check_cancel()
            self.log_msg("Generating script via Gemini…")
            script_text = gemini_generate_script(self.gemini_key.get().strip(), title)
            self.log_msg("Script generated:\n" + script_text + "\n")
            with open(os.path.join(out_dir, "script.txt"), "w", encoding="utf-8") as f:
                f.write(script_text)

            # 2) Image prompts
            self._check_cancel()
            n = int(self.n_images.get())
            self.log_msg(f"Creating {n} image prompts via Gemini…")
            prompts = gemini_generate_image_prompts(self.gemini_key.get().strip(), script_text, n)
            for i, p in enumerate(prompts, start=1):
                self.log_msg(f"[Prompt {i}] {p}")

            # 3) Images (Imagen -> HF fallback)
            self._check_cancel()
            self.log_msg("Generating images (Imagen -> HF fallback if needed)…")
            image_paths = generate_images_with_fallback(
                self.gemini_key.get().strip(),
                os.environ.get("HF_TOKEN", "") or self.hf_token.get().strip(),
                prompts,
                assets_dir
            )
            self.log_msg(f"Generated {len(image_paths)} images.")

            # 4) TTS
            self._check_cancel()
            self.log_msg("Synthesizing narration with ElevenLabs…")
            tts_path = os.path.join(assets_dir, "narration.mp3")
            elevenlabs_tts(self.eleven_key.get().strip(), self.voice_id.get().strip(), script_text, tts_path)
            self.log_msg(f"Narration saved -> {tts_path}")

            # 5) Captions + proportional durations
            chunks = split_text_into_chunks(script_text, len(image_paths))

            with AudioFileClip(tts_path) as aclip:
                total_dur = max(2.0, aclip.duration)

            durations = allocate_proportional_durations(chunks, total_dur, MIN_SEG_DUR)

            if self.save_srt.get():
                srt_path = os.path.join(out_dir, f"{safe_filename(title)}.srt")
                write_srt_with_durations(chunks, durations, srt_path)
                self.log_msg(f"SRT saved -> {srt_path}")

            use_paths = image_paths
            if self.burn_captions.get():
                self.log_msg("Rendering burn-in captions onto images…")
                cap_dir = os.path.join(assets_dir, "captioned")
                use_paths = caption_images(image_paths, chunks, cap_dir, VIDEO_WIDTH, VIDEO_HEIGHT)
                self.log_msg(f"Captioned frames -> {cap_dir}")

            # 6) Video (use proportional per-image durations)
            self._check_cancel()
            self.log_msg("Compiling video with MoviePy…")
            ensure_dir(os.path.dirname(out_video) or ".")
            total_dur_final, out_path = build_video_from_assets(use_paths, tts_path, out_video, durations)
            self.log_msg(f"Done! Video length: {total_dur_final:.1f}s")
            self.log_msg(f"Final video -> {out_path}")
            messagebox.showinfo("Success", f"Video created!\n\n{out_path}")

        except requests.HTTPError as e:
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text if hasattr(e, "response") and e.response is not None else str(e)
            self.log_msg("HTTP Error:\n" + (json.dumps(detail, ensure_ascii=False, indent=2) if isinstance(detail, dict) else str(detail)))
            messagebox.showerror("HTTP Error", f"{e}\n\n{detail}")
        except Exception as ex:
            err = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
            self.log_msg("Error:\n" + err)
            messagebox.showerror("Error", str(ex))
        finally:
            self.set_working(False)


# =============================
# --------- MAIN --------------
# =============================

def main():
    root = Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
