"""integration3 #8b: route_onset_exclusions (scenario_space_av_stack.yaml) - onsets a short route can never grade."""
import glob
import json
import random
from pathlib import Path

import pytest

from engine import scenarios as SC
from engine.config import REPO_ROOT, load_stage_configs, load_yaml, resolve_scenario_space
from engine.monitor import load_av_stack_run

ROUTES = ["bench2drive:28035", "town10_long", "bench2drive:24206", "bench2drive:28048", "bench2drive:28154"]
POOL = ["F-01", "F-02", "F-03", "F-04", "F-05", "F-06", "F-07"]


@pytest.fixture(scope="module")
def space():
    return load_yaml(resolve_scenario_space(False)[0])


def test_allowed_onset_events(space):
    assert SC.allowed_onset_events(space, "town10_long") == ["straight", "curve_entry", "curve_apex", "curve_exit"]
    assert SC.allowed_onset_events(space, "bench2drive:28035") == ["straight", "curve_entry"]      # speed unknown
    assert SC.allowed_onset_events(space, "bench2drive:28035", speed_kmh=25) == ["straight", "curve_entry"]
    assert SC.allowed_onset_events(space, "bench2drive:28035", speed_kmh=30) == ["straight"]
    assert SC.allowed_onset_events(space, "bench2drive:28048", speed_kmh=15) == ["curve_entry", "curve_apex", "curve_exit"]
    assert SC.allowed_onset_events(space, "bench2drive:28048", speed_kmh=25) == ["curve_entry"]
    # control faults sample only curve_entry / curve_apex (handover_axes): 28035 at >= 30 km/h has none of them left
    f01 = SC.faults_by_id(space)["F-01"]
    assert SC.allowed_onset_events(space, "bench2drive:28035", f01, 25) == ["curve_entry"]
    assert SC.allowed_onset_events(space, "bench2drive:28035", f01, 35) == []


def test_validate_rejects_excluded_combo_and_history_may_skip_it(space):
    c = SC.make_candidate(space, route_id="bench2drive:28154", speed=20, fault_id="F-04", params={"magnitude_rad": 0.05},
                          onset_event="curve_apex", offset=0.0, source="nemotron_a")
    probs = SC.validate_against_space(space, c, ROUTES)
    assert any("excluded on bench2drive:28154" in p for p in probs)
    assert SC.validate_against_space(space, c, ROUTES, check_exclusions=False) == []
    ok, tags, problems = SC.gate(space, c, ROUTES, None, 4.41)
    assert not ok and "G-02_reject" in tags
    c2 = dict(c, onset_event="straight")
    assert SC.validate_against_space(space, c2, ROUTES) == []


def test_space_filling_never_picks_an_excluded_onset(space):
    for seed in range(5):
        cands = SC.space_filling(space, ROUTES, POOL, 60, random.Random(seed))
        assert len(cands) >= 55
        for c in cands:
            assert c["onset_event"] not in SC.onset_exclusions(space, c["route_id"], c["environment"]["max_speed_kmh"]), c
            assert SC.validate_against_space(space, c, ROUTES) == []


def test_stub_a_moves_an_excluded_onset(space):
    from engine import nemotron as NM
    schema = NM.candidate_schema(space, ROUTES, POOL, ["S1"], ["O-001"], 3)["properties"]["candidates"]["items"]
    c = {"fault_id": "F-04", "route_id": "bench2drive:28035", "max_speed_kmh": 20.0, "onset_event": "curve_exit"}
    NM._stub_respect_exclusions(c, schema, space["route_onset_exclusions"])
    assert c["onset_event"] == "straight"


def _baselines():
    ed = load_stage_configs()["paired_runs"].get("event_detection") or {}
    out, seen = [], set()
    for d in sorted(glob.glob(str(REPO_ROOT / "result" / "result_*" / "runs" / "baselines" / "*" / ""))):
        d = Path(d)
        if d.name in seen or not (d / "manifest.json").is_file():
            continue
        seen.add(d.name)
        man, ac = json.loads((d / "manifest.json").read_text()), json.loads((d / "agent_config.json").read_text())
        if "bench2drive" not in str(man.get("routes_file", "")).lower():
            continue
        run = load_av_stack_run(d)
        ev = SC.detect_route_events(run.ticks, run.dt, **ed)
        out.append((f"bench2drive:{man['route_id']}", float(ac["max_speed_kmh"]), ev))
    return out


def test_exclusions_match_the_measured_baselines(space):
    rows = _baselines()
    if not rows:
        pytest.skip("no recorded baselines in result/")
    for route, entries in space["route_onset_exclusions"].items():
        mine = [(kmh, ev) for r, kmh, ev in rows if r == route]
        assert mine, route
        for e in entries:
            name, lo = (e, None) if isinstance(e, str) else (e["event"], e["min_kmh"])
            left = [(kmh, ev["t_end"] - ev[name] if ev.get(name) is not None else None) for kmh, ev in mine]
            if lo is None:   # always excluded: >= 80 % of the measured baselines leave < 6 s (or miss the event)
                short = [x is None or x < 6.0 for _, x in left]
                assert sum(short) >= 0.8 * len(short), (route, name, left)
            else:            # excluded from lo km/h up: every baseline at >= lo is short, every one below is not
                for kmh, x in left:
                    assert (x is None or x < 6.0) == (kmh >= lo), (route, name, kmh, x)


def test_result6_v03_discards_are_caught_before_carla(space):
    stats = REPO_ROOT / "result" / "result_6" / "loop_stats.json"
    if not stats.is_file():
        pytest.skip("result_6 not in this checkout")
    pool = {}
    for f in glob.glob(str(REPO_ROOT / "result" / "result_6" / "round_*" / "03_04_pool.json")):
        for c in json.loads(Path(f).read_text()):
            pool[c["scenario_id"]] = c
    skipped = [r["scenario_id"] for r in json.loads(stats.read_text())["rejected"]
               if any("baseline: V-03" in p for p in r.get("problems", []))]
    assert len(skipped) == 7
    caught = [s for s in skipped
              if pool[s]["onset_event"] in SC.onset_exclusions(space, pool[s]["route_id"], pool[s]["environment"]["max_speed_kmh"])]
    unconditional = [s for s in skipped if pool[s]["onset_event"] in SC.onset_exclusions(space, pool[s]["route_id"], None)]
    assert len(unconditional) == 6 and len(caught) == 7
