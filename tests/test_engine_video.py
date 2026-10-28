"""Clip extraction + explanation verification on a SYNTHETIC run folder."""
import random

import pytest

from engine.explainer import build_messages, stub_explanation, verify
from engine.fixtures.synthetic_sim import write_run_folder
from engine.monitor import compute_run_record, load_av_stack_run, monitor_timeline
from engine.video import extract_clip, ffmpeg_exe

TAX = ["departed_before_switch", "steering_angle_saturation", "stable_recovery", "collision_after_departure",
       "pre_switch_lateral_drift", "late_detection", "steering_rate_limit"]

pytestmark = pytest.mark.skipif(ffmpeg_exe() is None, reason="ffmpeg not available")


@pytest.fixture(scope="module")
def clip_and_timeline(tmp_path_factory):
    d = tmp_path_factory.mktemp("v")
    run_dir = write_run_folder(d / "run", route="bench2drive:26956", max_speed_kmh=30,
                               fault=dict(mode="saturate", start=20.0, duration=500.0, magnitude=1.0), stop_after_fault_s=15)
    run = load_av_stack_run(run_dir)
    rec = compute_run_record(run)
    tl = monitor_timeline(run, rec)
    clip = extract_clip(run_dir, d / "clip", anchor_s=rec["switch_time"], front_frames=run.front_frames,
                        tick_times=[t["t"] for t in run.ticks], dt=run.dt, before_s=2, after_s=6, fps=4, max_height=720,
                        sheet_cfg={"enabled": True, "fps": 2, "cols": 4})
    return clip, tl


def test_clip_files(clip_and_timeline):
    clip, tl = clip_and_timeline
    assert clip is not None and clip.video_path.is_file() and clip.sheet_path.is_file()
    assert 8 <= len(clip.frame_paths) <= 34   # run may end 1 s after a collision
    assert clip.start_s <= tl["switch_time_s"] <= clip.end_s
    assert all(b > a for a, b in zip(clip.frame_times, clip.frame_times[1:]))


def test_messages_put_media_first(clip_and_timeline):
    clip, tl = clip_and_timeline
    msgs = build_messages(clip, tl, TAX, "video", "think_answer")
    user = msgs[-1]["content"]
    assert user[0]["type"] == "video_url" and user[-1]["type"] == "text"
    msgs = build_messages(clip, tl, TAX, "contact_sheet", "think_answer")
    assert msgs[-1]["content"][0]["type"] == "image_url"


def test_verification_tolerance(clip_and_timeline):
    clip, tl = clip_and_timeline
    good = stub_explanation(tl, clip, random.Random(0), hallucinate_p=0.0)
    assert verify(good, tl, clip, TAX, 0.5)["accepted"]
    bad = dict(good, switch_time_s=tl["switch_time_s"] + 1.2)
    v = verify(bad, tl, clip, TAX, 0.5)
    assert not v["accepted"] and not v["checks"]["switch_time_s"]["pass"]
    assert not verify(dict(good, mechanism_tags=["made_up"]), tl, clip, TAX)["accepted"]
    assert not verify(None, tl, clip, TAX)["accepted"]


def test_prompt_hides_observations(clip_and_timeline):
    import json as _j
    clip, tl = clip_and_timeline
    text = build_messages(clip, tl, TAX, "video", "think_answer")[-1]["content"][-1]["text"]
    assert "lane_departure_intervals_s" not in text and "recovery_outcome" not in text
    assert "switch_time_s\": " in text or "switch_time_s" in text
    good = stub_explanation(tl, clip, random.Random(0), hallucinate_p=0.0)
    assert not verify(dict(good, collision_seen=not good["collision_seen"]), tl, clip, TAX)["accepted"]


def test_departure_reference_is_first_visible_frame():
    """BUG-017: the model sees only the 4 fps frames. Reference = first frame inside a departure interval;
    a departure entirely between two frames is not visible (null is right, a claim must sit on the interval)."""
    from engine.explainer import visible_departure
    from engine.video import Clip
    ft = [35.1 + 0.25 * k for k in range(27)]
    clip = Clip(None, None, [], ft, ft[0], ft[-1], 37.1, "front.mp4", "front_frames.jsonl", True)
    tl = {"switch_time_s": 37.1, "lane_departure_intervals_s": [[37.15, 37.3]], "collision_times_s": []}
    assert visible_departure(tl, ft) == (None, [(37.15, 37.3)])
    base = {"switch_time_s": 37.1, "departure_time_s": None, "collision_seen": False, "evidence_frames_s": [37.1],
            "mechanism_tags": []}
    assert verify(base, tl, clip, TAX)["accepted"]
    assert verify(dict(base, departure_time_s=37.35), tl, clip, TAX)["accepted"]
    assert not verify(dict(base, departure_time_s=38.5), tl, clip, TAX)["accepted"]
    tl2 = dict(tl, lane_departure_intervals_s=[[40.1, 41.3]])
    assert visible_departure(tl2, ft)[0] == 40.1 + 1e-9 or abs(visible_departure(tl2, ft)[0] - 40.1) < 1e-6
    assert not verify(base, tl2, clip, TAX)["accepted"]                 # visible departure, claimed none


def test_prompt_lists_frame_times(clip_and_timeline):
    clip, tl = clip_and_timeline
    text = build_messages(clip, tl, TAX, "video", "think_answer")[-1]["content"][-1]["text"]
    assert f"{clip.frame_times[0]:.2f}" in text and f"{clip.frame_times[-1]:.2f}" in text and "Frame list" in text


def test_overlay_font_is_large():
    from engine.video import overlay_font
    f = overlay_font(37)
    box = f.getbbox("t=12.35 s")
    assert box[3] - box[1] >= 20      # PIL's default bitmap font was ~10 px


def test_dense_frames_near_key_moments_and_events_are_covered(tmp_path_factory):
    """BUG-020: 10 fps within 1 s of the switch / departure / collision, 4 fps elsewhere; the clip window is widened
    so a departure or collision that the log contains is always inside the clip."""
    from engine.video import cover_times, key_times
    d = tmp_path_factory.mktemp("dense")
    run_dir = write_run_folder(d / "run", route="bench2drive:26956", max_speed_kmh=30,
                               fault=dict(mode="saturate", start=20.0, duration=500.0, magnitude=1.0), stop_after_fault_s=15)
    run = load_av_stack_run(run_dir)
    rec = compute_run_record(run)
    tl = monitor_timeline(run, rec)
    sw = rec["switch_time"]
    late = sw + 7.5                                     # an event after the normal anchor+6 s window
    tl_cov = dict(tl, collision_times_s=[late])
    clip = extract_clip(run_dir, d / "clip", anchor_s=sw, front_frames=run.front_frames,
                        tick_times=[t["t"] for t in run.ticks], dt=run.dt, before_s=2, after_s=6, fps=4,
                        dense_times=[sw], dense_fps=10, dense_window_s=1.0, cover=cover_times(tl_cov))
    assert sw in key_times(tl)
    ft = clip.frame_times
    gaps_near = [b - a for a, b in zip(ft, ft[1:]) if abs(a - sw) < 0.9 and abs(b - sw) < 0.9]
    gaps_far = [b - a for a, b in zip(ft, ft[1:]) if b < sw - 1.05]
    assert gaps_near and max(gaps_near) <= 0.11
    assert gaps_far and min(gaps_far) >= 0.2
    if late <= run.ticks[-1]["t"]:
        assert clip.end_s >= late
    assert clip.video_path.name == "clip.mp4" and len(list((d / "clip").glob("frame_*.png"))) == len(ft)
    text = build_messages(clip, tl, TAX, "video", "think_answer")[-1]["content"][-1]["text"]
    assert "spacing is not uniform" in text


def test_shallow_departure_is_not_required():
    """BUG-020: a frame counts as a visible departure only when the car is >= margin outside the lane there."""
    from engine.explainer import visible_departure
    from engine.video import Clip
    ft = [19.75 + 0.25 * k for k in range(10)]
    clip = Clip(None, None, [], ft, ft[0], ft[-1], 19.75, "front.mp4", "front_frames.jsonl", True)
    tl = {"switch_time_s": None, "lane_departure_intervals_s": [[21.75, 22.05]], "collision_times_s": [],
          "lane_departure_depth_m": [[21.75, 0.017], [21.8, 0.036], [22.0, 0.111], [22.05, 0.12]]}
    assert visible_departure(tl, ft)[0] == 21.75                       # no margin: old behaviour
    assert visible_departure(tl, ft, margin_m=0.10)[0] == 22.0         # first frame really outside by >= 10 cm
    assert visible_departure(tl, ft, margin_m=0.20)[0] is None
    base = {"switch_time_s": None, "departure_time_s": None, "collision_seen": False, "evidence_frames_s": [21.0],
            "mechanism_tags": []}
    assert verify(base, tl, clip, TAX, 0.5, visible_margin_m=0.20)["accepted"]
    assert not verify(base, tl, clip, TAX, 0.5, visible_margin_m=0.10)["accepted"]
    assert verify(dict(base, departure_time_s=21.75), tl, clip, TAX, 0.5, visible_margin_m=0.10)["accepted"]
    assert not verify(dict(base, departure_time_s=20.0), tl, clip, TAX, 0.5, visible_margin_m=0.20)["accepted"]


def test_prompt_explains_tags_collision_and_hides_depth(clip_and_timeline):
    clip, tl = clip_and_timeline
    text = build_messages(clip, tl, TAX, "video", "think_answer")[-1]["content"][-1]["text"]
    assert "lane_departure_depth_m" not in text
    assert "- steering_angle_saturation: steering held at its angle limit" in text
    assert "frames_outside_lane_s" in text and "not a collision" in text
