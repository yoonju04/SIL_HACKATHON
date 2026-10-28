"""SYNTHETIC run-folder generator for --mock mode (no CARLA). NOT MEASURED DATA.

Writes the same file layout as sim-eval's PDM agent (integration/pdm_agent.py, monitor_logs.py,
benchmark_recording.py) so that 09 monitor / 10 video explainer can be exercised end to end:
  agent_controls.jsonl, steps.jsonl, behavior.jsonl, monitor_probe.jsonl, collisions.json,
  summary.json, agent_config.json, manifest.json, results.json, evaluation_events.json,
  front_frames.jsonl + front.mp4 (tiny top-down rendering, only if ffmpeg is available)

The vehicle model is a kinematic error model (e_y' = v sin e_psi, e_psi' = v tan(delta)/L - v kappa)
with a linear "MPC", the av_stack supervisor rules (tracking_divergence 1.0 m / 0.6 rad held 0.25 s,
solver_failure same tick, latched) and an "LQR" with angle clip + 90 deg/s rate limit. Gains, route
shapes and the collision rule are invented so that both recoveries and failures occur. Every file
carries synthetic_fixture: true.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

DT = 0.05
WHEELBASE = 2.9
MAX_STEER = 0.70           # rad, synthetic
HALF_W, HALF_L, LANE_W = 0.93, 2.4, 3.5
COLLISION_OFFSET_M = 3.2   # synthetic: |offset| beyond this = hit roadside object

# synthetic route curvature profiles: list of (length_m, curvature_1pm)
ROUTES = {
    "bench2drive:26956": [(90, 0.0), (22, -1 / 14), (120, 0.0), (30, 1 / 30), (120, 0.0)],
    "bench2drive:28035": [(70, 0.0), (18, 1 / 10), (150, 0.0), (25, -1 / 20), (100, 0.0)],
    "town10_long": [(120, 0.0), (35, 1 / 25), (80, 0.0), (35, -1 / 25), (73, 0.0)],
    "town04_highway": [(300, 0.0), (200, 1 / 250), (300, 0.0), (200, -1 / 300), (400, 0.0)],
    # rough shapes from the 2026-10-28 CARLA 30 km/h baselines (curve timing / |kappa| only, SYNTHETIC)
    "bench2drive:24206": [(30, 0.0), (25, 1 / 45), (180, 0.0)],
    "bench2drive:28048": [(200, 0.0), (20, 1 / 15), (60, 0.0)],
    "bench2drive:28154": [(120, 0.0), (25, -1 / 40), (10, 0.0)],
}


def kappa_at(route: str, s: float) -> float:
    acc = 0.0
    for length, k in ROUTES.get(route, ROUTES["town10_long"]):
        if s < acc + length:
            return k
        acc += length
    return 0.0


def route_length(route: str) -> float:
    return float(sum(length for length, _ in ROUTES.get(route, ROUTES["town10_long"])))


def _gains(v: float, wn: float, zeta: float) -> tuple[float, float]:
    v = max(v, 2.0)
    return wn * wn * WHEELBASE / (v * v), 2 * zeta * wn * WHEELBASE / v


def simulate(route: str, max_speed_kmh: float, fault: dict[str, Any] | None, seed: int = 17,
             stop_after_fault_s: float | None = None, max_time_s: float = 120.0,
             supervised: bool = True) -> dict[str, Any]:
    """Return per-tick arrays + events. fault = {mode, start, duration, magnitude, on_s, off_s} or None.
    supervised=False is the no-reaction run (`--controller mpc_unprotected`): the MPC keeps steering (SR1 FTTI)."""
    rng = random.Random(hash((route, round(max_speed_kmh, 3), seed)) & 0xFFFFFFFF)
    v_cap = max_speed_kmh / 3.6
    s = 0.0; v = 0.0; e_y = 0.0; e_psi = 0.0; delta = 0.0
    active = "mpc"; reason = None; switched_at = None; switch_count = 0; bad_since = None
    history: list[float] = []
    rows = []
    collisions = []
    t = 0.0
    mode = (fault or {}).get("mode", "none")
    start = (fault or {}).get("start", 1e9)
    duration = (fault or {}).get("duration", 0.0)
    mag = (fault or {}).get("magnitude", 0.0)
    end_s = route_length(route)
    while t < max_time_s and s < end_s - 1.0:
        if stop_after_fault_s is not None and mode != "none" and t >= start + stop_after_fault_s:
            break
        k = kappa_at(route, s + max(v, 1.0) * 1.0)  # PDM looks ~1 s ahead for speed
        v_target = min(v_cap, math.sqrt(4.0 / abs(k)) if abs(k) > 1e-6 else v_cap)
        v += max(-3.0, min(1.5, (v_target - v) / 1.0)) * DT
        v = max(v, 0.0)
        kappa = kappa_at(route, s)
        delta_ref = math.atan(WHEELBASE * kappa)
        fault_on = mode != "none" and start <= t < start + duration
        status = "ok"
        if active == "mpc":
            a, b = _gains(v, 1.2, 0.9)
            u = delta_ref - (a * e_y + b * e_psi)
            history.append(u)
            reported = (e_y, e_psi)
            ok = True
            if fault_on:
                if mode == "solver_fail":
                    ok = False
                elif mode == "bias":
                    u += mag
                elif mode == "drift":
                    u += mag * (t - start)
                elif mode == "intermittent_bias":
                    on, off = fault.get("on_s", 0.2), fault.get("off_s", 0.2)
                    if round(t - start, 6) % (on + off) < on - 1e-9:
                        u += mag
                elif mode == "saturate":
                    u = math.copysign(MAX_STEER, mag)
                elif mode == "output_delay":
                    u = history[max(0, len(history) - 1 - int(round(mag)))]
            if not supervised:
                if ok:  # no supervisor: the (faulty) MPC command is applied; a failed solve holds the last command
                    lim = math.radians(60.0) * DT
                    delta = max(delta - lim, min(delta + lim, max(-MAX_STEER, min(MAX_STEER, u))))
            else:
                if not ok:
                    reason = "solver_failure"
                elif abs(reported[0]) > 1.0 or abs(reported[1]) > 0.6:
                    bad_since = t if bad_since is None else bad_since
                    if t - bad_since >= 0.25 - 1e-9:
                        reason = "tracking_divergence"
                else:
                    bad_since = None
                if reason is not None:
                    active, switched_at, switch_count = "lqr", round(t, 4), 1
                else:
                    lim = math.radians(60.0) * DT
                    delta = max(delta - lim, min(delta + lim, max(-MAX_STEER, min(MAX_STEER, u))))
        if active == "lqr":
            a, b = _gains(v, 1.6, 0.75)
            u = delta_ref - (a * e_y + b * e_psi)
            u = max(-MAX_STEER, min(MAX_STEER, u))
            lim = math.radians(90.0) * DT
            delta = max(delta - lim, min(delta + lim, u))
            status = "lqr"
        psi_dot = v * math.tan(delta) / WHEELBASE
        e_psi_dot = psi_dot - v * kappa
        rows.append(dict(t=round(t, 4), s=s, v=v, e_y=e_y, e_psi=e_psi, delta=delta, kappa=kappa,
                         yaw_rate=psi_dot, ay=v * psi_dot + rng.gauss(0, 0.05), active=active,
                         reason=reason, switched_at=switched_at, switch_count=switch_count,
                         fault_on=fault_on, status=status))
        if abs(e_y) > COLLISION_OFFSET_M and not collisions:
            collisions.append(dict(t=round(t, 4), other="static.prop.synthetic_roadside"))
        e_y += v * math.sin(e_psi) * DT + rng.gauss(0, 0.003)
        e_psi += e_psi_dot * DT + rng.gauss(0, 0.0015)
        e_psi = (e_psi + math.pi) % (2 * math.pi) - math.pi
        s += v * math.cos(e_psi) * DT
        t += DT
        if collisions and t > collisions[0]["t"] + 1.0:
            break  # synthetic: run ends shortly after a collision
    return dict(rows=rows, collisions=collisions)


def _render_frames(rows: list[dict], width: int = 320, height: int = 180):
    """Tiny synthetic top-down view: lane lines, car box shifted by e_y and tilted by e_psi."""
    px_per_m = 22.0
    cx = width // 2
    for r in rows:
        img = np.full((height, width, 3), 60, np.uint8)
        for side in (-1, 1):
            x = int(cx + side * LANE_W / 2 * px_per_m)
            img[:, max(0, x - 1):x + 2] = 230
        for side in (-3, 3):  # neighbouring lane edge (dashed)
            x = int(cx + side * LANE_W / 2 * px_per_m)
            if 0 <= x < width:
                img[(np.arange(height) // 10) % 2 == 0, max(0, x - 1):x + 1] = 150
        color = (40, 200, 40) if r["active"] == "mpc" else (240, 150, 0)
        car_x = cx - r["e_y"] * px_per_m          # left offset (+) drawn to the left
        for dy in range(int(-HALF_L * px_per_m), int(HALF_L * px_per_m)):
            y = height // 2 + dy
            x0 = int(car_x + dy * math.tan(r["e_psi"]) - HALF_W * px_per_m)
            x1 = int(car_x + dy * math.tan(r["e_psi"]) + HALF_W * px_per_m)
            if 0 <= y < height:
                img[y, max(0, x0):max(0, min(width, x1))] = color
        img[:6, :] = (200, 0, 200)  # magenta band = synthetic marker
        yield img


def _ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def write_run_folder(out: Path, *, route: str, max_speed_kmh: float, fault: dict[str, Any] | None,
                     seed: int = 17, stop_after_fault_s: float | None = None, record_front: bool = True,
                     controller: str = "supervised") -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sim = simulate(route, max_speed_kmh, fault, seed, stop_after_fault_s, supervised=controller == "supervised")
    rows = sim["rows"]
    f = fault or {"mode": "none", "start": 20.0, "duration": 5.0, "magnitude": 0.05}
    marker = {"synthetic_fixture": True, "generator": "engine/fixtures/synthetic_sim.py"}
    config = dict(model=None, planner="pdm", record_front=record_front, controller=controller,
                  mpc_fault=f["mode"], fault_start=f["start"], fault_duration=f["duration"],
                  fault_magnitude=f["magnitude"], fault_on_s=f.get("on_s", 0.2), fault_off_s=f.get("off_s", 0.2),
                  stop_after_fault=stop_after_fault_s, max_speed_kmh=max_speed_kmh, dt=DT, out=str(out),
                  geometry=dict(max_steer_rad=MAX_STEER, lf=1.35, lr=1.55), **marker)
    (out / "agent_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    town = {"bench2drive:26956": "Town05", "bench2drive:28035": "Town01", "bench2drive:24206": "Town03",
            "bench2drive:28048": "Town02", "bench2drive:28154": "Town10HD"}.get(route, route)
    (out / "manifest.json").write_text(json.dumps(dict(route_id=route, town=town, planner="pdm",
                                                       controller=controller, seed=seed, **marker), indent=2))
    with (out / "agent_controls.jsonl").open("w") as ac, (out / "steps.jsonl").open("w") as st, \
            (out / "behavior.jsonl").open("w") as be, (out / "monitor_probe.jsonl").open("w") as mp:
        for r in rows:
            steer = max(-1.0, min(1.0, -r["delta"] / MAX_STEER))
            ac.write(json.dumps(dict(t=r["t"], steer=steer, throttle=0.3, brake=0.0, measured_speed=r["v"],
                                     planner="PDM-Lite", source=r["active"], tracking_error=r["e_y"],
                                     heading_error=r["e_psi"], lateral_status=r["status"],
                                     fault_injected=r["fault_on"], safe_stop=False, fault_mode=f["mode"],
                                     fault_magnitude=f["magnitude"], active_controller=r["active"],
                                     supervisor_reason=r["reason"], switched_at_s=r["switched_at"],
                                     switch_count=r["switch_count"], primary_status="ok",
                                     primary_elapsed_ms=8.0 + 2.0 * math.sin(r["t"]))) + "\n")
            st.write(json.dumps(dict(t=r["t"], vx=r["v"], vy=0.0, yaw_rate=r["yaw_rate"],
                                     e_y_lane=r["e_y"], e_psi=r["e_psi"])) + "\n")
            be.write(json.dumps(dict(t=r["t"], active_controller=r["active"], supervisor_reason=r["reason"],
                                     fault_injected=r["fault_on"])) + "\n")
            mp.write(json.dumps(dict(t=r["t"], lane_offset_m=r["e_y"], lane_heading_err_rad=r["e_psi"],
                                     lane_width_m=LANE_W, intended_lane=[1, -1], map_lane_now=[1, -1],
                                     ay_mps2=r["ay"], bbox_half_length_m=HALF_L, bbox_half_width_m=HALF_W,
                                     fault_effective=None)) + "\n")
    (out / "collisions.json").write_text(json.dumps(sim["collisions"]))
    (out / "summary.json").write_text(json.dumps(dict(
        supervisor=dict(safe_stop=False),
        meta=dict(planner="PDM-Lite", controller=controller, fault_mode=f["mode"],
                  fault_start=f["start"], fault_magnitude=f["magnitude"], **marker)), indent=2))
    (out / "evaluation_events.json").write_text("[]")
    (out / "results.json").write_text(json.dumps(dict(synthetic_fixture=True, note="no Bench2Drive scoring in mock")))
    if record_front:
        exe = _ffmpeg()
        if exe is not None:
            with (out / "front_frames.jsonl").open("w") as ff:
                for i, r in enumerate(rows):
                    ff.write(json.dumps(dict(sensor_frame=1000 + i, t=r["t"], speed=r["v"])) + "\n")
            proc = subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                     "-s", "320x180", "-r", "20", "-i", "-", "-c:v", "libx264", "-preset",
                                     "ultrafast", "-pix_fmt", "yuv420p", str(out / "front.mp4")],
                                    stdin=subprocess.PIPE)
            assert proc.stdin is not None
            for img in _render_frames(rows):
                proc.stdin.write(img.tobytes())
            proc.stdin.close()
            proc.wait()
    return out
