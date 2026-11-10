"""YouTube snapshots and crowd inference, without a persistent decoder or worker."""

from __future__ import annotations

import csv
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image


def youtube_url(value: str) -> str:
    """Accept video links only and strip playlist/tracking parameters."""
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise ValueError("Vul een geldige YouTube-videolink in.")
    if host == "youtu.be":
        video_id = parsed.path.strip("/")
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/live/", "/embed/")):
            video_id = parsed.path.split("/")[2]
        else:
            video_id = ""
    else:
        video_id = ""
    if len(video_id) != 11 or not all(c.isascii() and (c.isalnum() or c in "-_") for c in video_id):
        raise ValueError("Gebruik een YouTube-link naar één video of livestream.")
    return f"https://www.youtube.com/watch?v={video_id}"


def resolve_stream(value: str) -> dict:
    """Resolve a currently live YouTube video to an FFmpeg-readable video URL."""
    url = youtube_url(value)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg ontbreekt. Installeer ffmpeg en herstart de WebUI.")
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("Installeer de webafhankelijkheden uit crowd_counting/requirements-web.txt.") from exc
    options = {
        "format": "best[height<=720][vcodec!=none]/bestvideo[height<=720]/best[vcodec!=none]/bestvideo",
        "noplaylist": True, "quiet": True, "no_warnings": True,
        "socket_timeout": 10, "retries": 1, "extractor_retries": 1,
        "js_runtimes": {"deno": {}, "node": {}},
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError("YouTube-stream niet bereikbaar. Controleer de link en werk zo nodig yt-dlp bij. "
                           "Deze versie ondersteunt openbare streams zonder login.") from exc
    if not info or info.get("live_status") != "is_live":
        raise ValueError("Deze video is momenteel niet live. Kies een lopende YouTube-livestream.")
    if not info.get("url"):
        raise RuntimeError("YouTube heeft geen bruikbare videostream teruggegeven.")
    return {"url": info["url"], "title": info.get("title", "YouTube-livestream")}


def capture_frame(stream_url: str) -> bytes:
    """Open at the live edge each time, avoiding accumulated video buffering."""
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-rw_timeout", "10000000", "-i", stream_url,
        "-frames:v", "1", "-an", "-vf", "scale=1024:-2",
        "-f", "image2pipe", "-c:v", "mjpeg", "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=25, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg ontbreekt. Installeer ffmpeg en herstart de WebUI.") from exc
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Geen beeld ontvangen binnen de toegestane tijd. "
                           "De stream is onderbroken of niet bereikbaar. Start de meting opnieuw.") from exc
    if not result.stdout:
        raise RuntimeError("De livestream leverde geen beeld op. Mogelijk is de uitzending afgelopen.")
    with Image.open(io.BytesIO(result.stdout)) as image:
        image.verify()
    return result.stdout


def heatmap_overlay(original: Image.Image, density: np.ndarray) -> bytes:
    """Render relative intensity over the analyzed frame, including empty maps."""
    from matplotlib import colormaps

    values = np.maximum(density, 0)
    peak = float(values.max()) if values.size else 0
    normalized = values / peak if peak > 0 else np.zeros_like(values)
    colors = colormaps["jet"](normalized)
    colors[..., 3] = normalized * 0.65
    overlay = Image.fromarray((colors * 255).astype(np.uint8)).resize(original.size, Image.Resampling.BILINEAR)
    composed = Image.alpha_composite(original.convert("RGBA"), overlay).convert("RGB")
    buffer = io.BytesIO()
    composed.save(buffer, format="JPEG")
    return buffer.getvalue()


def analyze_frame(frame: bytes, loaded, *, max_edge=1024, threshold=0.5) -> dict:
    """Reuse the photo pipeline; discard temporary exports after this measurement."""
    from infer import InferenceOptions, predict_image
    from scipy.ndimage import gaussian_filter

    with tempfile.TemporaryDirectory(prefix="crowd-live-") as directory:
        root = Path(directory)
        image_path = root / "frame.jpg"
        image_path.write_bytes(frame)
        result = predict_image(InferenceOptions(
            checkpoint=str(loaded.checkpoint_path), image=str(image_path), output_dir=directory,
            device=str(loaded.device), architecture=loaded.architecture,
            max_edge=max_edge, threshold=threshold, benchmark_csv="",
        ), loaded)
        with Image.open(image_path) as image:
            original = image.convert("RGB")
        if result["prediction_type"] == "density":
            density = np.load(root / "frame_density.npy")
        else:
            density = np.zeros((128, 128), dtype=np.float32)
            with (root / "frame_points.csv").open() as stream:
                for row in csv.DictReader(stream):
                    x, y = float(row["x_original_px"]), float(row["y_original_px"])
                    if 0 <= x < original.width and 0 <= y < original.height:
                        density[int(y / original.height * 128), int(x / original.width * 128)] += 1
            density = gaussian_filter(density, sigma=2)
        return {
            "count": result["estimated_count"], "forward_ms": result["mean_forward_ms"],
            "frame": frame, "heatmap": heatmap_overlay(original, density),
            "prediction_type": result["prediction_type"],
        }


def history_csv(rows: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["timestamp_utc", "estimated_count", "processing_ms"])
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
