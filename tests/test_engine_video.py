"""Clip extraction + explanation verification on a SYNTHETIC run folder."""
import random

import pytest

from pipeline.engine.explainer import build_messages, stub_explanation, verify
from pipeline.engine.fixtures.synthetic_sim import write_run_folder
from pipeline.engine.monitor import compute_run_record, load_av_stack_run, monitor_timeline
from pipeline.engine.video import extract_clip, ffmpeg_exe

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
