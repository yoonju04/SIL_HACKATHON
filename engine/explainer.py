"""10 video explainer: commentator, not judge.

For runs the monitor already labelled (failed or boundary-near), send the clip (video first, then text)
plus the monitor event timeline to Cosmos (Dedicated Endpoint of nvidia/Cosmos3-Super-Reasoner;
model id configurable) or the fallback nvidia/Nemotron-3.5-Super-VL. Ask for <think>..</think>
<answer>{JSON}</answer>. The explanation is stored next to - never inside - the run_record; the
explainer cannot change recovery_outcome / labels. Claimed event times are verified against the
monitor log (time_tolerance_s, default 0.5 s); only accepted explanations feed back.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .llm import LLMError, extract_json
from .video import Clip, data_url

ANSWER_FIELDS = ("switch_time_s", "departure_time_s", "collision_seen", "mechanism", "mechanism_tags", "novel_tags",
                 "evidence_frames_s", "counterfactual", "uncertainty")

SYSTEM = """You are a vehicle-dynamics incident analyst for a CARLA simulation of an MPC->LQR steering handover. \
The safety label of this run was ALREADY decided by a deterministic monitor; do NOT change, judge or restate it as your \
own conclusion. Explain the visible mechanism: when the steering owner switched, whether and when the car body left the \
lane, and why. Every frame shows simulation time at the bottom ("t=.. s"); use those times."""

# Role separation: the explainer gets the CONTEXT the monitor knows before looking (fault, trigger, switch time) but NOT
# the observations it must make from the pixels (departure, collision, offsets, saturation, final labels). Those hidden
# monitor values are what verify() checks the claims against, so a model cannot pass by copying the prompt.
VISIBLE_TIMELINE_KEYS = ("fault_mode", "fault_magnitude", "planned_fault_onset_s", "actual_fault_onset_s",
                         "trigger_time_s", "trigger_source", "switch_time_s")

THINK_FORMAT = "Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>."


def answer_spec(taxonomy: list[str]) -> dict:
    return {
        "switch_time_s": "number or null - time the steering owner changed to the LQR fallback, read from the overlay",
        "departure_time_s": "number or null - t of the FIRST frame (from the frame list) where a wheel/body is outside the lane lines; null if no frame shows it",
        "collision_seen": "true or false - does the ego car hit anything in the clip",
        "mechanism": "one or two sentences: cause and sequence",
        "mechanism_tags": f"array, ONLY tags from this list: {taxonomy} (anything else goes to novel_tags)",
        "novel_tags": "array of short snake_case tags for patterns not in the list (may be empty)",
        "evidence_frames_s": "array of frame times copied from the frame list (only times inside this clip) that show the key events",
        "counterfactual": "null or {parameter: one of [onset_offset_s, max_speed_kmh, fault_magnitude, fault_duration_s], "
                          "direction: increase|decrease, amount: number, expected_effect: string}",
        "uncertainty": "low|medium|high",
    }


def build_messages(clip: Clip, timeline: dict, taxonomy: list[str], media: str, prompt_format: str,
                   visible_keys: tuple[str, ...] | list[str] | None = VISIBLE_TIMELINE_KEYS) -> list[dict]:
    shown = timeline if visible_keys is None else {k: timeline.get(k) for k in visible_keys}
    content: list[dict] = []
    if media == "video" and clip.video_path:
        content.append({"type": "video_url", "video_url": {"url": data_url(clip.video_path, "video/mp4")}})
    elif media == "contact_sheet" and clip.sheet_path:
        content.append({"type": "image_url", "image_url": {"url": data_url(clip.sheet_path, "image/jpeg")}})
    elif media == "frames":
        for p in clip.frame_paths[::2]:
            content.append({"type": "image_url", "image_url": {"url": data_url(p, "image/png")}})
    else:
        raise ValueError(f"media {media} not available for this clip")
    media_note = {"video": f"a {len(clip.frame_paths)}-frame clip at 4 fps",
                  "contact_sheet": "one contact-sheet image; tiles are in time order left-to-right, top-to-bottom",
                  "frames": "individual frames in time order"}[media]
    times = ", ".join(f"{t:.2f}" for t in clip.frame_times)
    text = (f"Input: {media_note} covering t={clip.start_s:.2f}..{clip.end_s:.2f} s (camera: {clip.source}).\n"
            f"Frame list - frame i is at t = [{times}] s (same as the yellow t= label on each frame). Every time you "
            "report must be one of these frame times; times in the context below may lie before the clip and are not "
            "frames.\n"
            f"Known context from the logs (deterministic):\n{json.dumps(shown, default=str)}\n"
            "Lane departure, collision and offsets are NOT given: read them from the frames.\n\n"
            "Questions: Did the steering owner switch, and when? Was the car already displaced at the switch? "
            "Did any part of the car leave the lane, and when? Was the car in a curve? Did steering look saturated or "
            "rate-limited? Return in <answer> ONLY a JSON object with these keys:\n"
            f"{json.dumps(answer_spec(taxonomy), indent=1)}")
    if prompt_format == "think_answer":
        text += "\n" + THINK_FORMAT
    content.append({"type": "text", "text": text})   # media before text (Cosmos training format)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def explain(client, settings: dict, clip: Clip, timeline: dict, taxonomy: list[str], tag: str = "C",
            visible_keys=VISIBLE_TIMELINE_KEYS) -> dict:
    """Try media in the configured order; returns {explanation|None, media_used, model, errors}."""
    errors = []
    for media in settings.get("media", ["video", "contact_sheet"]):
        try:
            msgs = build_messages(clip, timeline, taxonomy, media, settings.get("prompt_format", "think_answer"), visible_keys)
        except ValueError as e:
            errors.append(str(e)); continue
        try:
            res = client.chat(settings["model"], msgs, temperature=settings.get("temperature"), top_p=settings.get("top_p"),
                              max_tokens=settings.get("max_tokens"), thinking=settings.get("thinking"),
                              extra_body=settings.get("extra_body"), tag=f"{tag}:{media}")
        except LLMError as e:
            errors.append(f"{media}: {str(e)[:160]}"); continue
        try:
            obj = extract_json(res.content)
        except ValueError as e:
            errors.append(f"{media}: parse {e}"); continue
        if not isinstance(obj, dict):
            errors.append(f"{media}: answer is not an object"); continue
        obj = {k: obj.get(k) for k in ANSWER_FIELDS}
        return {"explanation": obj, "media_used": media, "model": settings["model"], "errors": errors,
                "usage": res.usage}
    return {"explanation": None, "media_used": None, "model": settings.get("model"), "errors": errors}


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def monitor_departure_in_clip(timeline: dict, start: float, end: float) -> float | None:
    for a, b in timeline.get("lane_departure_intervals_s") or []:
        if b >= start and a <= end:
            return max(a, start)
    return None


def visible_departure(timeline: dict, frame_times: list[float], eps: float = 1e-6) -> tuple[float | None, list]:
    """BUG-017: (first clip FRAME inside a monitor departure interval, intervals that overlap the clip).
    The model only sees the 4 fps frames; a 0.15 s departure between two frames cannot be seen at all."""
    ts = sorted(frame_times or [])
    if not ts:
        return None, []
    over = [(a, b) for a, b in (timeline.get("lane_departure_intervals_s") or []) if b >= ts[0] and a <= ts[-1]]
    for t in ts:
        if any(a - eps <= t <= b + eps for a, b in over):
            return t, over
    return None, over


def verify(expl: dict | None, timeline: dict, clip: Clip, taxonomy: list[str], tol: float = 0.5) -> dict:
    """Compare claimed times with the monitor log. accepted = every checkable claim passes."""
    if expl is None:
        return {"accepted": False, "checks": {}, "reason": "no_parseable_explanation"}
    checks: dict[str, Any] = {}
    sw_m, sw_c = timeline.get("switch_time_s"), _num(expl.get("switch_time_s"))
    if sw_m is None:
        checks["switch_time_s"] = {"pass": sw_c is None, "monitor": None, "claimed": sw_c}
    else:
        checks["switch_time_s"] = {"pass": sw_c is not None and abs(sw_c - sw_m) <= tol, "monitor": sw_m,
                                   "claimed": sw_c, "error_s": None if sw_c is None else round(sw_c - sw_m, 3)}
    dep_c = _num(expl.get("departure_time_s"))
    if clip.frame_times:
        dep_m, over = visible_departure(timeline, clip.frame_times)
    else:   # no frame list (old clips): interval start clipped to the clip
        dep_m, over = monitor_departure_in_clip(timeline, clip.start_s, clip.end_s), []
    if dep_m is None and over:
        # departure happened only BETWEEN frames: not visible. null is right; a claim must sit on that interval.
        checks["departure_time_s"] = {"pass": dep_c is None or any(a - tol <= dep_c <= b + tol for a, b in over),
                                      "monitor": None, "claimed": dep_c, "not_visible_intervals": over}
    elif dep_m is None:
        checks["departure_time_s"] = {"pass": dep_c is None, "monitor": None, "claimed": dep_c}
    else:
        checks["departure_time_s"] = {"pass": dep_c is not None and abs(dep_c - dep_m) <= tol, "monitor": dep_m,
                                      "claimed": dep_c, "error_s": None if dep_c is None else round(dep_c - dep_m, 3)}
    col_m = any(clip.start_s - tol <= t <= clip.end_s + tol for t in (timeline.get("collision_times_s") or []))
    col_c = expl.get("collision_seen")
    if isinstance(col_c, str):
        col_c = {"true": True, "false": False}.get(col_c.strip().lower())
    if col_c is not None:
        checks["collision_seen"] = {"pass": bool(col_c) == col_m, "monitor": col_m, "claimed": col_c}
    ev = [_num(x) for x in (expl.get("evidence_frames_s") or [])]
    checks["evidence_frames_s_in_clip"] = {"pass": all(x is not None and clip.start_s - tol <= x <= clip.end_s + tol for x in ev)}
    tags = expl.get("mechanism_tags") or []
    checks["tags_in_taxonomy"] = {"pass": isinstance(tags, list) and all(t in taxonomy for t in tags),
                                  "unknown": [t for t in tags if t not in taxonomy] if isinstance(tags, list) else tags}
    return {"accepted": all(c["pass"] for c in checks.values()), "checks": checks, "tolerance_s": tol}


def stub_explanation(timeline: dict, clip: Clip, rng: random.Random, hallucinate_p: float = 0.25) -> dict:
    """SYNTHETIC explanation for --mock: rule-based from the timeline, sometimes wrong on purpose."""
    tags = []
    sw = timeline.get("switch_time_s")
    dep = visible_departure(timeline, clip.frame_times)[0] if clip.frame_times else \
        monitor_departure_in_clip(timeline, clip.start_s, clip.end_s)
    if sw is not None and dep is not None and dep <= sw + 1e-6:
        tags.append("departed_before_switch")
    if (timeline.get("angle_saturation_ratio_1s") or 0) > 0.3:
        tags.append("steering_angle_saturation")
    if (timeline.get("rate_saturation_ratio_1s") or 0) > 0.2:
        tags.append("steering_rate_limit")
    if timeline.get("collision_times_s"):
        tags.append("collision_after_departure")
    if sw is not None and timeline.get("planned_fault_onset_s") is not None and sw - timeline["planned_fault_onset_s"] > 2.0:
        tags.append("late_detection")
    if not tags:
        tags.append("stable_recovery" if dep is None else "pre_switch_lateral_drift")
    bad = rng.random() < hallucinate_p
    shift = 1.2 if bad else rng.uniform(-0.2, 0.2)
    novel = ["post_switch_weave"] if rng.random() < 0.3 else []
    cf = None
    if sw is not None and rng.random() < 0.5:
        cf = {"parameter": "onset_offset_s", "direction": "decrease", "amount": 0.5,
              "expected_effect": "SYNTHETIC: earlier fault -> switch before the curve"}
    return {"switch_time_s": None if sw is None else round(sw + shift, 2),
            "departure_time_s": None if dep is None else round(dep + rng.uniform(-0.2, 0.2), 2),
            "collision_seen": bool(timeline.get("collision_times_s")),
            "mechanism": "SYNTHETIC stub explanation (no model called): " + ", ".join(tags),
            "mechanism_tags": tags, "novel_tags": novel,
            "evidence_frames_s": [round(x, 2) for x in clip.frame_times[:: max(1, len(clip.frame_times) // 3)]][:3],
            "counterfactual": cf, "uncertainty": "high" if bad else "medium"}
