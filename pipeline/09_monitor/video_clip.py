"""Cut the part of a run video that matters (fault onset -> end of the graded window) and pull a few frames.

Video sources in an av_stack run folder (`run_benchmark.py --record-front`):
  view.mp4  + view_frames.jsonl   chase camera with the STEERING: MPC/LQR label (frame times: see note)
  front.mp4 + front_frames.jsonl  roof camera, one json line per frame with the simulation time `t`
The frames file maps every video frame to simulation time, so the clip lines up with the monitor's
times. Without it the clip falls back to "frame k = t0 + k / fps" and is marked approximate.
Note: av_stack writes view.mp4 without a frames file today; add one line per written frame
({"t": <sim time>}) in ChaseView.show() to get exact timing for the chase view.

Output clips are H.264 mp4 (plays in every browser). ffmpeg comes from PATH or, if missing, from the
`imageio-ffmpeg` pip package (which bundles a binary).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

SOURCES = (("view.mp4", "view_frames.jsonl"), ("front.mp4", "front_frames.jsonl"))


def ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def _probe_fps(exe: str, video: Path) -> float:
    out = subprocess.run([exe, "-hide_banner", "-i", str(video)], capture_output=True, text=True).stderr
    m = re.search(r"([\d.]+) fps", out)
    return float(m.group(1)) if m else 20.0


def find_video(run_dir: str | Path) -> dict | None:
    """{"video", "times" (list of sim times per frame or None), "exact"} for the best video in the run folder."""
    d = Path(run_dir)
    for vid, frames in SOURCES:
        v = d / vid
        if not v.exists() or v.stat().st_size == 0:
            continue
        times = None
        f = d / frames
        if f.exists():
            times = []
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    t = row.get("t", row.get("elapsed"))
                    if t is not None:
                        times.append(float(t))
        return {"video": v, "times": times or None, "exact": bool(times)}
    return None


def _video_seconds(times, fps, t_sim):
    """Video position [s] of simulation time t_sim (nearest logged frame; else assume the video starts at t = 0)."""
    if times:
        k = min(range(len(times)), key=lambda i: abs(times[i] - t_sim))
        return k / fps
    return max(0.0, t_sim)


def make_clip(run_dir: str | Path, t_start: float, t_end: float, out_mp4: str | Path,
              frames_dir: str | Path | None = None, n_frames: int = 8, width: int = 960) -> dict | None:
    """Cut [t_start, t_end] (simulation seconds) into an H.264 clip; optionally save n_frames JPEGs evenly spread.

    Returns {"clip", "frames": [{"path", "t"}], "source", "exact_timing"} or None when there is no video/ffmpeg."""
    src = find_video(run_dir)
    exe = ffmpeg_exe()
    if src is None or exe is None:
        return None
    fps = _probe_fps(exe, src["video"])
    a = _video_seconds(src["times"], fps, t_start)
    b = _video_seconds(src["times"], fps, t_end)
    if b <= a:
        b = a + 1.0
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-i", str(src["video"]), "-t", f"{b - a:.3f}",
           "-vf", f"scale={width}:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
           "-movflags", "+faststart", "-an", str(out_mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out_mp4.exists():
        return None
    frames = []
    if frames_dir is not None and n_frames > 0:
        fd = Path(frames_dir)
        fd.mkdir(parents=True, exist_ok=True)
        dur = b - a
        # one pass: n_frames evenly spread (one per slice of dur / n_frames seconds)
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(out_mp4), "-vf", f"fps={n_frames / dur:.6f}",
                        "-frames:v", str(n_frames), "-q:v", "4", str(fd / "f%02d.jpg")], capture_output=True)
        for i in range(n_frames):
            p = fd / f"f{i + 1:02d}.jpg"
            if p.exists():
                pos = dur * (i + 0.5) / n_frames
                frames.append({"path": str(p), "t": round(t_start + pos * (t_end - t_start) / dur, 2)})
    return {"clip": str(out_mp4), "frames": frames, "source": src["video"].name, "exact_timing": src["exact"]}
