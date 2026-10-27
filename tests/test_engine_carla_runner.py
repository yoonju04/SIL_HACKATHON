"""Real-mode command line (no CARLA is started)."""
import os
import re
from pathlib import Path

import pytest

from engine import scenarios as SC
from engine.carla_runner import CarlaRunner, baseline_key, key_hash
from engine.config import FIXTURE_SPACE, load_stage_configs, load_yaml, resolve_av_stack


def make(tmp_path, av_stack):
    cfg = load_stage_configs()
    space = load_yaml(FIXTURE_SPACE)
    routes = cfg["paired_runs"]["carla"]["routes"]
    r = CarlaRunner(tmp_path, routes, 17, space, 15.0, av_stack=av_stack, python_exe="python")
    r.tool = None
    c = SC.make_candidate(space, route_id="bench2drive:26956", speed=30, fault_id="F-04",
                          params={"magnitude_rad": 0.05, "duration_s": 2.0}, onset_event="curve_entry", offset=0.2, source="t")
    return r, c


def test_command_layout(tmp_path):
    r, c = make(tmp_path, tmp_path / "av")
    base = r.command(c, tmp_path / "b", None)
    assert base[1].endswith("run_benchmark.py") and "--mpc-fault" not in base
    assert base[base.index("--routes") + 1] == "bench2drive" and base[base.index("--route-id") + 1] == "26956"
    assert "--no-view" in base and "--record-front" in base and base[base.index("--planner") + 1] == "pdm"
    f = r.command(c, tmp_path / "f", r.fault_args(c, 31.0))
    assert f.count("--max-speed") == 1 and f[f.index("--fault-start") + 1] == "31.00"
    assert f[f.index("--stop-after-fault") + 1] == "15.0"


def test_baseline_key_is_stable():
    k1 = baseline_key({"route_id": "x", "environment": {"max_speed_kmh": 30.0}}, {"town": "T"}, 17)
    k2 = baseline_key({"route_id": "x", "environment": {"max_speed_kmh": 30.0}}, {"town": "T"}, 17)
    assert key_hash(k1) == key_hash(k2)
    assert key_hash(k1) != key_hash(dict(k1, max_speed_kmh=35.0))


@pytest.mark.skipif(resolve_av_stack() is None, reason="sim-eval av_stack not available (SIM_EVAL_ROOT/AV_STACK_ROOT)")
def test_flags_exist_in_run_benchmark(tmp_path):
    av = resolve_av_stack()
    src = (av / "run_benchmark.py").read_text(encoding="utf-8")
    declared = set(re.findall(r"add_argument\(\s*['\"](--[a-z0-9-]+)", src))
    r, c = make(tmp_path, av)
    for cmd in (r.command(c, tmp_path / "b", None), r.command(c, tmp_path / "f", r.fault_args(c, 31.0))):
        used = {a for a in cmd if a.startswith("--")}
        assert used <= declared, used - declared
