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

ANSWER_FIELDS = ("switch_time_s", "frames_outside_lane_s", "departure_time_s", "collision_seen", "mechanism",
                 "mechanism_tags", "novel_tags", "evidence_frames_s", "counterfactual", "uncertainty")

# What each taxonomy tag means (BUG-020: the bare list was ignored and tags were invented). Unknown tags get no text.
TAG_HELP = {
    "pre_switch_lateral_drift": "the car was already sliding to one side before the switch",
    "departed_before_switch": "the car was already over the lane line at the switch",
    "curve_entry_during_handover": "the switch happened while entering or driving through a curve",
    "steering_angle_saturation": "steering held at its angle limit",
    "steering_rate_limit": "steering could not turn fast enough to follow",
    "overshoot_after_switch": "after the switch the car came back and crossed to the other side",
    "oscillation_after_switch": "after the switch the car weaved left and right",
    "late_detection": "the fault went unnoticed for a long time",
    "collision_after_departure": "the car hit something after leaving the lane",
    "stable_recovery": "the car returned to the lane without problems",
}

SYSTEM = """You are a vehicle-dynamics incident analyst for a CARLA simulation of an MPC->LQR steering handover. \
The safety label of this run was ALREADY decided by a deterministic monitor; do NOT change, judge or restate it as your \
own conclusion. Explain the visible mechanism: when the steering owner switched, whether and when the car body left the \
lane, and why. Every frame shows simulation time at the bottom ("t=.. s"); use those times."""

# Role separation: the explainer gets the CONTEXT the monitor knows before looking (fault, trigger, switch time) but NOT
# the observations it must make from the pixels (departure, collision, offsets, saturation, final labels). Those hidden
# monitor values are what verify() checks the claims against, so a model cannot pass by copying the prompt.
VISIBLE_TIMELINE_KEYS = ("fault_mode", "fault_magnitude", "planned_fault_onset_s", "actual_fault_onset_s",
                         "trigger_time_s", "trigger_source", "switch_time_s")

# integration4: log_grounded mode (default in video_explainer.yaml prompt.mode). In result_8, 23 of 24 explanations were
# rejected on departure_time_s: the monitor logged no departure in any of the 24 runs, yet the model claimed one in 23
# (12 times at the given switch time). A front camera does not show the wheels against the lane line. The monitor's
# departure / collision / offset values are now GIVEN as facts; the model explains how the clip shows them and why,
# and verify() checks that the explanation agrees with the log (tags, evidence frames at the given events).
GROUNDED_TIMELINE_KEYS = VISIBLE_TIMELINE_KEYS + ("lane_departure_intervals_s", "first_departure_after_switch_s",
                                                  "collision_times_s", "max_abs_lane_offset_m", "max_abs_lane_offset_t_s",
                                                  "angle_saturation_ratio_1s", "rate_saturation_ratio_1s", "safe_stop_time_s")
GROUNDED_FIELDS = ("mechanism", "mechanism_tags", "novel_tags", "evidence_frames_s", "counterfactual", "uncertainty")

SYSTEM_GROUNDED = """You are a vehicle-dynamics incident analyst for a CARLA simulation of an MPC->LQR steering handover. A deterministic monitor already measured this run; its events (switch, lane departures, collisions, offsets, saturation) are given below as facts. Do not re-measure, judge or restate them as your own conclusion, and do not assume that anything went wrong. Explain what the clip shows around those events and why they happened (road geometry, steering, timing). Every frame shows simulation time at the bottom ("t=.. s"); use those times."""

THINK_FORMAT = "Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>."


def answer_spec(taxonomy: list[str]) -> dict:
    return {
        "switch_time_s": "number or null - time the steering owner changed to the LQR fallback, read from the overlay",
        "frames_outside_lane_s": "array of frame times (copied from the frame list) where a wheel or the body is visibly "
                                 "over the lane line or road edge in THAT frame; [] if no frame shows it",
        "departure_time_s": "number or null - the first time in frames_outside_lane_s; null if that list is empty. Decide "
                            "it from the pixels only: it is usually NOT the fault onset, trigger or switch time from the context",
        "collision_seen": "true or false - true ONLY if a frame shows the ego car touching another vehicle, person or "
                          "object; a car passing close or parked nearby is not a collision",
        "mechanism": "one or two sentences: cause and sequence",
        "mechanism_tags": "array of tags copied exactly from the allowed tag list below; never invent a tag here",
        "novel_tags": "array of short snake_case tags for patterns not in the list (may be empty)",
        "evidence_frames_s": "array of frame times copied from the frame list (only times inside this clip) that show the key events",
        "counterfactual": "null or {parameter: one of [onset_offset_s, max_speed_kmh, fault_magnitude, fault_duration_s], "
                          "direction: increase|decrease, amount: number, expected_effect: string}",
        "uncertainty": "low|medium|high",
    }


def build_messages(clip: Clip, timeline: dict, taxonomy: list[str], media: str, prompt_format: str,
                   visible_keys: tuple[str, ...] | list[str] | None = VISIBLE_TIMELINE_KEYS,
                   mode: str = "pixel_check") -> list[dict]:
    if mode == "log_grounded" and visible_keys == VISIBLE_TIMELINE_KEYS:
        visible_keys = GROUNDED_TIMELINE_KEYS
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
    gaps = sorted({round(b - a, 3) for a, b in zip(clip.frame_times, clip.frame_times[1:])})
    rate = (f"{1 / gaps[-1]:g} fps, {1 / gaps[0]:g} fps around key moments" if len(gaps) > 1 and gaps[0] > 0
            else "4 fps")
    media_note = {"video": f"a {len(clip.frame_paths)}-frame clip ({rate}; spacing is not uniform, use the frame list)"
                  if len(gaps) > 1 else f"a {len(clip.frame_paths)}-frame clip at {rate}",
                  "contact_sheet": "one contact-sheet image; tiles are in time order left-to-right, top-to-bottom",
                  "frames": "individual frames in time order"}[media]
    times = ", ".join(f"{t:.2f}" for t in clip.frame_times)
    text = (f"Input: {media_note} covering t={clip.start_s:.2f}..{clip.end_s:.2f} s (camera: {clip.source}).\n"
            f"Frame list - frame i is at t = [{times}] s (same as the yellow t= label on each frame). Every time you "
            "report must be one of these frame times; times in the context below may lie before the clip and are not "
            "frames.\n"
            f"Known context from the logs (deterministic):\n{json.dumps(shown, default=str)}\n"
            + ("These logged events are facts (an empty list means it did not happen). Do not re-measure them.\n"
               if mode == "log_grounded" else "Lane departure, collision and offsets are NOT given: read them from the frames.\n")
            + "Allowed mechanism_tags (use only these exact strings; anything else goes to novel_tags):\n"
            + "\n".join(f"- {t}: {TAG_HELP.get(t, '')}".rstrip(": ") for t in taxonomy) + "\n\n"
            + ("Questions: What happened in this clip and why? Which frames show the logged events? Was the car in a "
               "curve? How did the steering behave before and after the switch? If nothing went wrong, say so "
               "(stable_recovery). Return in <answer> ONLY a JSON object with these keys:\n"
               if mode == "log_grounded" else
               "Questions: Did the steering owner switch, and when? Was the car already displaced at the switch? "
               "Did any part of the car leave the lane, and when? Was the car in a curve? Did steering look saturated or "
               "rate-limited? Return in <answer> ONLY a JSON object with these keys:\n")
            + json.dumps({k: v for k, v in answer_spec(taxonomy).items() if mode != "log_grounded" or k in GROUNDED_FIELDS},
                         indent=1))
    if prompt_format == "think_answer":
        text += "\n" + THINK_FORMAT
    content.append({"type": "text", "text": text})   # media before text (Cosmos training format)
    return [{"role": "system", "content": SYSTEM_GROUNDED if mode == "log_grounded" else SYSTEM},
            {"role": "user", "content": content}]


def explain(client, settings: dict, clip: Clip, timeline: dict, taxonomy: list[str], tag: str = "C",
            visible_keys=VISIBLE_TIMELINE_KEYS, mode: str = "pixel_check") -> dict:
    """Try media in the configured order; returns {explanation|None, media_used, model, errors}."""
    errors = []
    for media in settings.get("media", ["video", "contact_sheet"]):
        try:
            msgs = build_messages(clip, timeline, taxonomy, media, settings.get("prompt_format", "think_answer"), visible_keys,
                                  mode=mode)
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
        obj = {k: obj.get(k) for k in (GROUNDED_FIELDS if mode == "log_grounded" else ANSWER_FIELDS)}
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


def visible_departure(timeline: dict, frame_times: list[float], eps: float = 1e-6,
                      margin_m: float = 0.0) -> tuple[float | None, list]:
    """BUG-017: (first clip FRAME inside a monitor departure interval, intervals that overlap the clip).
    The model only sees the frames; a 0.15 s departure between two frames cannot be seen at all.
    BUG-020: with margin_m > 0 and the per-tick depth in the timeline, a frame counts only if the car is at least
    margin_m outside the lane there (a wheel 2 cm over a painted line is not visible on camera)."""
    ts = sorted(frame_times or [])
    if not ts:
        return None, []
    over = [(a, b) for a, b in (timeline.get("lane_departure_intervals_s") or []) if b >= ts[0] and a <= ts[-1]]
    depth = timeline.get("lane_departure_depth_m") or []
    def deep_enough(t):
        if margin_m <= 0 or not depth:
            return True
        near = [d for u, d in depth if abs(u - t) <= 0.026]
        return bool(near) and max(near) >= margin_m - 1e-9
    for t in ts:
        if any(a - eps <= t <= b + eps for a, b in over) and deep_enough(t):
            return t, over
    return None, over


def tag_conflicts(tags: list, timeline: dict) -> list[str]:
    """Mechanism tags that contradict the monitor log (log_grounded mode)."""
    dep = timeline.get("lane_departure_intervals_s") or []
    sw = timeline.get("switch_time_s")
    col = timeline.get("collision_times_s") or []
    bad = []
    for t in tags:
        if t == "departed_before_switch" and not (sw is not None and any(a <= sw + 1e-6 for a, _ in dep)):
            bad.append(t)
        elif t == "collision_after_departure" and not col:
            bad.append(t)
        elif t == "stable_recovery" and (col or timeline.get("first_departure_after_switch_s") is not None):
            bad.append(t)
        elif t == "steering_angle_saturation" and not (timeline.get("angle_saturation_ratio_1s") or 0) > 0:
            bad.append(t)
        elif t == "steering_rate_limit" and not (timeline.get("rate_saturation_ratio_1s") or 0) > 0:
            bad.append(t)
    return bad


def verify_grounded(expl: dict | None, timeline: dict, clip: Clip, taxonomy: list[str], tol: float = 0.5) -> dict:
    """log_grounded: the events were given, so check that the explanation agrees with them instead of re-measuring.
    accepted = tags exist and do not contradict the log, evidence frames are in the clip and at least one sits at a
    given event (switch, lane-departure interval, collision) when the clip has one, and a mechanism is written."""
    if expl is None:
        return {"accepted": False, "checks": {}, "reason": "no_parseable_explanation", "mode": "log_grounded"}
    checks: dict[str, Any] = {}
    tags = expl.get("mechanism_tags") or []
    ok_list = isinstance(tags, list)
    checks["tags_in_taxonomy"] = {"pass": ok_list and all(t in taxonomy for t in tags),
                                  "unknown": [t for t in tags if t not in taxonomy] if ok_list else tags}
    conflicts = tag_conflicts(tags, timeline) if ok_list else []
    checks["tags_consistent_with_log"] = {"pass": not conflicts, "conflicting": conflicts}
    ev = [_num(x) for x in (expl.get("evidence_frames_s") or [])]
    checks["evidence_frames_s_in_clip"] = {"pass": all(x is not None and clip.start_s - tol <= x <= clip.end_s + tol for x in ev)}
    lo, hi = clip.start_s - tol, clip.end_s + tol
    points = [t for t in [timeline.get("switch_time_s"), *(timeline.get("collision_times_s") or [])]
              if t is not None and lo <= t <= hi]
    spans = [(a, b) for a, b in (timeline.get("lane_departure_intervals_s") or []) if b >= lo and a <= hi]
    if points or spans:
        near = [x for x in ev if x is not None and (any(abs(x - p) <= tol for p in points)
                                                   or any(a - tol <= x <= b + tol for a, b in spans))]
        checks["evidence_at_given_events"] = {"pass": bool(near), "events_s": points, "departure_intervals_s": spans}
    mech = expl.get("mechanism")
    checks["mechanism_written"] = {"pass": isinstance(mech, str) and len(mech.strip()) >= 20}
    return {"accepted": all(c["pass"] for c in checks.values()), "checks": checks, "tolerance_s": tol, "mode": "log_grounded"}


def verify(expl: dict | None, timeline: dict, clip: Clip, taxonomy: list[str], tol: float = 0.5,
           visible_margin_m: float = 0.0, mode: str = "pixel_check") -> dict:
    """Compare claimed times with the monitor log. accepted = every checkable claim passes."""
    if mode == "log_grounded":
        return verify_grounded(expl, timeline, clip, taxonomy, tol)
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
        dep_m, over = visible_departure(timeline, clip.frame_times, margin_m=visible_margin_m)
    else:   # no frame list (old clips): interval start clipped to the clip
        dep_m, over = monitor_departure_in_clip(timeline, clip.start_s, clip.end_s), []
    if dep_m is None and over:
        # departure only BETWEEN frames or too shallow to see: null is right; a claim must sit on that interval.
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
    return {"accepted": all(c["pass"] for c in checks.values()), "checks": checks, "tolerance_s": tol,
            "visible_margin_m": visible_margin_m}


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
        clean = timeline.get("first_departure_after_switch_s") is None and not timeline.get("collision_times_s")
        tags.append("stable_recovery" if dep is None and clean else "pre_switch_lateral_drift")
    bad = rng.random() < hallucinate_p
    shift = 1.2 if bad else rng.uniform(-0.2, 0.2)
    if bad and not timeline.get("collision_times_s"):
        tags = tags + ["collision_after_departure"]       # SYNTHETIC log contradiction (log_grounded catches it)
    novel = ["post_switch_weave"] if rng.random() < 0.3 else []
    keys = [t for t in [sw, dep, *(timeline.get("collision_times_s") or [])] if t is not None]
    near = sorted({min(clip.frame_times, key=lambda f: abs(f - k)) for k in keys}) if clip.frame_times and keys else []
    cf = None
    if sw is not None and rng.random() < 0.5:
        cf = {"parameter": "onset_offset_s", "direction": "decrease", "amount": 0.5,
              "expected_effect": "SYNTHETIC: earlier fault -> switch before the curve"}
    return {"switch_time_s": None if sw is None else round(sw + shift, 2),
            "departure_time_s": None if dep is None else round(dep + rng.uniform(-0.2, 0.2), 2),
            "collision_seen": bool(timeline.get("collision_times_s")),
            "mechanism": "SYNTHETIC stub explanation (no model called): " + ", ".join(tags),
            "mechanism_tags": tags, "novel_tags": novel,
            "evidence_frames_s": [round(x, 2) for x in (near or clip.frame_times[:: max(1, len(clip.frame_times) // 3)])][:3],
            "counterfactual": cf, "uncertainty": "high" if bad else "medium"}
