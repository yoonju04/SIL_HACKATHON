"""10 clip extraction: front.mp4 (or view.mp4) -> [anchor-2 s, anchor+6 s] at 4 fps, <=720p, timestamp overlay.

* front.mp4 frame k <-> front_frames.jsonl line k (sim time t), written once per sensor frame
  (sim-eval integration/benchmark_recording.py). Exact.
* view.mp4 has no timing file (chase_view.py writes one frame per agent tick once the camera has an
  image). Alignment is approximated end-aligned to the last tick; recorded as alignment='approx_end_aligned'.
* Frames are cut with ffmpeg (system ffmpeg, $FFMPEG, or the imageio-ffmpeg wheel), overlay drawn
  with Pillow at the bottom ("t=12.35 s"), then re-encoded as a 4 fps mp4 + a contact sheet image
  (one image for endpoints that accept a single image per request).
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


def ffmpeg_exe() -> str | None:
    if os.environ.get("FFMPEG") and Path(os.environ["FFMPEG"]).is_file():
        return os.environ["FFMPEG"]
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


@dataclass
class Clip:
    video_path: Path | None
    sheet_path: Path | None
    frame_paths: list[Path]
    frame_times: list[float]
    start_s: float
    end_s: float
    anchor_s: float
    source: str
    alignment: str
    overlay: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(video=str(self.video_path) if self.video_path else None,
                    contact_sheet=str(self.sheet_path) if self.sheet_path else None,
                    n_frames=len(self.frame_paths), frame_times_s=self.frame_times, start_s=self.start_s,
                    end_s=self.end_s, anchor_s=self.anchor_s, source=self.source, alignment=self.alignment,
                    overlay=self.overlay, notes=self.notes)


def count_frames(exe: str, video: Path) -> int:
    out = subprocess.run([exe, "-i", str(video), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    m = re.findall(r"frame=\s*(\d+)", out)
    return int(m[-1]) if m else 0


def frame_timeline(run_dir: Path, source: Path, front_frames: list[dict], tick_times: list[float], dt: float,
                   exe: str) -> tuple[list[float], str]:
    if source.name == "front.mp4" and front_frames:
        return [float(r["t"]) for r in front_frames], "front_frames.jsonl"
    n = count_frames(exe, source)
    if n == 0 or not tick_times:
        return [], "unknown"
    t_end = tick_times[-1]
    return [t_end - (n - 1 - i) * dt for i in range(n)], "approx_end_aligned"


def pick_indices(times: list[float], start: float, end: float, fps: float) -> list[int]:
    out, t = [], start
    step = 1.0 / fps
    while t <= end + 1e-9:
        if times and times[0] - 0.06 <= t <= times[-1] + 0.06:
            idx = min(range(len(times)), key=lambda i: abs(times[i] - t))
            if not out or out[-1] != idx:
                out.append(idx)
        t += step
    return out


OVERLAY_FONTS = ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "DejaVuSans.ttf", "arial.ttf")


def overlay_font(size: int):
    """A scalable font for the timestamp. BUG-017: PIL's default bitmap font is ~10 px high, so on a 450-720 px frame
    the "t=.. s" text was unreadable once the model downscaled the video, and the model guessed every time."""
    from PIL import ImageFont
    for name in OVERLAY_FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _overlay(paths: list[Path], times: list[float], max_h: int) -> bool:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return False
    font = None
    for p, t in zip(paths, times):
        im = Image.open(p).convert("RGB")
        if im.height > max_h:
            im = im.resize((int(im.width * max_h / im.height) // 2 * 2, max_h))
        d = ImageDraw.Draw(im)
        size = max(18, im.height // 12)            # ~37 px on 450 px, 60 px on 720 px
        font = font or overlay_font(size)
        bar = size + size // 2
        d.rectangle([0, im.height - bar, im.width // 2, im.height], fill=(0, 0, 0))
        d.text((size // 3, im.height - bar + size // 4), f"t={t:.2f} s", fill=(255, 255, 0), font=font)
        im.save(p)
    return True


def _contact_sheet(paths: list[Path], times: list[float], out: Path, fps_src: float, fps_sheet: float, cols: int,
                   tile_w: int) -> Path | None:
    try:
        from PIL import Image
    except Exception:
        return None
    step = max(1, int(round(fps_src / fps_sheet)))
    sel = paths[::step]
    if not sel:
        return None
    first = Image.open(sel[0])
    tile_h = int(first.height * tile_w / first.width)
    rows = (len(sel) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (0, 0, 0))
    for i, p in enumerate(sel):
        im = Image.open(p).convert("RGB").resize((tile_w, tile_h))
        sheet.paste(im, ((i % cols) * tile_w, (i // cols) * tile_h))
    sheet.save(out, quality=85)
    return out


def extract_clip(run_dir: Path, out_dir: Path, *, anchor_s: float, front_frames: list[dict], tick_times: list[float],
                 dt: float = 0.05, before_s: float = 2.0, after_s: float = 6.0, fps: float = 4.0, max_height: int = 720,
                 prefer: tuple[str, ...] = ("front.mp4", "view.mp4"), sheet_cfg: dict | None = None) -> Clip | None:
    exe = ffmpeg_exe()
    src = next((run_dir / n for n in prefer if (run_dir / n).is_file()), None)
    if exe is None or src is None:
        return None
    times, alignment = frame_timeline(run_dir, src, front_frames, tick_times, dt, exe)
    if not times:
        return None
    start, end = max(times[0], anchor_s - before_s), min(times[-1], anchor_s + after_s)
    idx = pick_indices(times, start, end, fps)
    if not idx:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out_dir))
    expr = "+".join(f"eq(n\\,{i})" for i in idx)
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src), "-vf", f"select='{expr}'", "-vsync", "0",
                    str(tmp / "f_%04d.png")], check=True)
    frames = sorted(tmp.glob("f_*.png"))
    ftimes = [round(times[i], 3) for i in idx][: len(frames)]
    notes = []
    overlay = _overlay(frames, ftimes, max_height)
    if not overlay:
        notes.append("Pillow missing: no timestamp overlay")
    final_frames = []
    for k, (p, t) in enumerate(zip(frames, ftimes)):
        q = out_dir / f"frame_{k:03d}.png"
        p.replace(q)
        final_frames.append(q)
    shutil.rmtree(tmp, ignore_errors=True)
    video = out_dir / "clip_4fps.mp4"
    (out_dir / "frames.json").write_text(json.dumps({"frame_times_s": ftimes, "source": src.name,
                                                     "alignment": alignment}), encoding="utf-8")
    subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                    str(out_dir / "frame_%03d.png"), "-vf", f"scale=-2:'min({max_height},ih)'", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    sc = sheet_cfg or {}
    sheet = _contact_sheet(final_frames, ftimes, out_dir / "contact_sheet.jpg", fps, sc.get("fps", 2), sc.get("columns", 4),
                           sc.get("tile_width_px", 320))
    return Clip(video if video.is_file() else None, sheet, final_frames, ftimes, ftimes[0], ftimes[-1], anchor_s,
                src.name, alignment, overlay, notes)


def data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()
