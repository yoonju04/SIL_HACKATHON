"""laya-judge: B gives the GP what it lacks (blind spot, first-tripping check, proposals), label rescue of truncated
runs, acquisition diversity, LQR first-command logging."""
import math
import random

import pytest

from engine import judges as J
from engine.surrogate import enforce_diversity


def test_gp_columns_blind_spot_and_first_check():
    bank = J.load_bank()
    ans = {"J1": {"probs": {"immediate_check": 0.2, "deviation_accumulation": 0.5, "no_dedicated_check": 0.3, "unknown": 0.0}}}
    x, names = J.jev_features(ans, bank)
    d = dict(zip(names, x))
    assert d["B_blind_spot"] == pytest.approx(0.3)
    assert d["B_first_check_immediate"] == pytest.approx(0.2) and d["B_first_check_accumulated"] == pytest.approx(0.5)
    assert not any(n.startswith("J1=") for n in names)


def _seed(mode="bias", **params):
    return {"scenario_id": f"s-{mode}", "route_id": "r", "environment": {"max_speed_kmh": 30.0},
            "fault": {"fault_id": mode, "mode": mode, "params": params}, "onset_event": "straight", "onset_offset_s": 0.0}


def test_b_proposals_code_grounded_and_limited():
    space = {"environment": {"max_speed_kmh": {"range": [10, 40]}}}
    rows = [({"answers": {"J1": {"probs": {"deviation_accumulation": 0.8, "immediate_check": 0.2}}}}, _seed("bias", magnitude_rad=0.05)),
            ({"answers": {"J1": {"probs": {"no_dedicated_check": 0.9}}}}, _seed("output_delay", delay_ticks=2)),
            ({"answers": {"J1": {"probs": {"immediate_check": 0.95}}}}, _seed("saturate", magnitude=0.5)),
            ({"answers": {"J1": {"probs": {"deviation_accumulation": 0.7}}}}, dict(_seed("bias", magnitude_rad=0.04), scenario_id="s2"))]
    made = []
    def make(seed, params, speed, why):
        made.append(why)
        return {"source": "B", "fault": {"mode": seed["fault"]["mode"], "params": params}, "speed": speed, "why": why}
    out = J.propose_scenarios(rows, {"per_round": 2, "min_prob": 0.4}, space, make)
    assert len(out) == 2 and {c["fault"]["mode"] for c in out} == {"output_delay", "bias"}      # one per mode, saturate skipped
    b = next(c for c in out if c["fault"]["mode"] == "bias")
    assert b["fault"]["params"]["magnitude_rad"] == pytest.approx(0.9 * math.radians(90) * 0.05, abs=1e-4) and b["speed"] == 35.0
    assert all("MPCSupervisor.inspect" in w for w in made)
    assert J.propose_scenarios(rows, {"enabled": False}, space, make) == []


def test_enforce_diversity_dedupe_and_fault_share():
    mk = lambda i, mode, x: {"scenario_id": str(i), "fault": {"mode": mode}, "x": x}
    sel = [mk(0, "bias", [0.0, 0.0]), mk(1, "bias", [0.5, 0.0]), mk(2, "bias", [0.9, 0.0]), mk(3, "drift", [0.0, 0.01])]
    reserve = [mk(4, "saturate", [0.3, 0.9]), mk(5, "drift", [0.6, 0.6])]
    ran = [[0.0, 0.0]]                                                   # sel[0] and sel[3] repeat a run point
    out = enforce_diversity(sel, reserve, ran, 4, lambda c: c["x"], 0.05, 0.5)
    ids = [c["scenario_id"] for c in out]
    assert "0" not in ids and "3" not in ids and ids[:2] == ["1", "2"]   # bias capped at 2 of 4
    assert set(ids) == {"1", "2", "4", "5"} and out[2]["selected_by"] == "diversity_refill"


def _ticks(n, dt, switch_at, end_offset, lane=0.2, ay=0.0):
    rows = []
    for i in range(n):
        t = i * dt
        rows.append({"t": t, "lane_width_m": 3.5, "lane_offset_m": lane if t >= switch_at else 0.0, "bbox_half_width_m": 1.0,
                     "lane_heading_err_rad": 0.0, "active_controller": "lqr" if t >= switch_at else "mpc", "vx": 8.0,
                     "ay_mps2": ay, "ctrl_steer": 0.0})
    return rows


def test_truncated_window_rescue_and_lqr_logging():
    from engine import monitor as MON
    import inspect
    src = inspect.getsource(MON.compute_run_record)
    assert "truncated_window_failure" in src and "route_ended_before_window_end" in src
    assert "lqr_unclipped_steer_rad" in inspect.getsource(MON)
    from pathlib import Path
    sl = (Path(J.__file__).resolve().parents[1] / "pipeline/08_carla_paired_run/av_stack/integration/supervised_lateral.py").read_text()
    assert "self.lqr_unclipped = float(result.steer)" in sl and "lqr_unclipped_steer_rad=self.lqr_unclipped" in sl
