"""laya-judge: --judge-context digest. engine/code_facts.py builds short factual digests from the av_stack source;
they must never carry an answer option of the question they serve, a conclusion, or a run outcome."""
import json
import re

import pytest
import yaml

from engine import code_facts as CF
from engine import judges as J
from engine.config import REPO_ROOT
from engine.fixtures.synthetic_sim import SUPERVISOR_PROFILES

J_YAML = REPO_ROOT / "pipeline" / "05b_judges" / "judges.yaml"
AV = REPO_ROOT / "pipeline" / "08_carla_paired_run" / "av_stack"
FAULT_MODES = ("solver_fail", "stale", "bias", "drift", "intermittent_bias", "saturate", "output_delay")
PARAMS = {"bias": {"magnitude_rad": 0.0801}, "drift": {"rate_rad_s": 0.0477},
          "intermittent_bias": {"magnitude_rad": 0.0581, "on_s": 0.4695, "off_s": 0.8853},
          "saturate": {"duration_s": 0.39, "side": "positive", "magnitude": 0.67, "ramp_s": 0.11},
          "output_delay": {"delay_ticks": 2}, "solver_fail": {}, "stale": {}}
# a finished run record's outcome fields and typical values: none may reach a judge state
OUTCOME_KEYS = ("switch_time", "actual_fault_onset", "final_safe_recovery", "recovery_labels", "departed_at_switch",
                "pre_switch_max_lateral_error_m", "handover_saturation", "rate_saturation_ratio", "safety_margin",
                "recovery_outcome", "sim_status", "metrics", "expected_from_code", "matches_code")


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(J_YAML.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bank():
    return J.load_bank()


def _ctx(mode, sup="tight"):
    return {"fault_mode": mode, "fault_params": PARAMS[mode], "onset_event": "curve_entry", "planner": "pdm",
            "controller": "supervised", "supervisor": sup, "max_speed_kmh": 30.0, "route_id": "r1"}


def _option_hits(text: str, options: list[str]) -> list[str]:
    low = text.lower()
    hits = []
    for o in options:
        for form in {o.lower(), o.lower().replace("_", " ")}:
            if re.search(r"(?<![a-z0-9_])" + re.escape(form) + r"(?![a-z0-9_])", low):
                hits.append(o)
    return hits


def test_profile_values_are_read_from_source():
    tight, legacy = CF.supervisor_profile(AV, "tight"), CF.supervisor_profile(AV, "legacy")
    assert tight == {"max_lateral_error_m": 0.6, "max_heading_error_rad": 0.35, "divergence_hold_s": 0.15,
                     "max_steer_rate_dps": 90.0}
    assert legacy["max_steer_rate_dps"] is None and legacy["divergence_hold_s"] == 0.25
    for sup, p in (("tight", tight), ("legacy", legacy)):   # same numbers as the fixture profiles
        f = SUPERVISOR_PROFILES[sup]
        assert (p["max_lateral_error_m"], p["max_heading_error_rad"], p["divergence_hold_s"], p["max_steer_rate_dps"]) == \
            (f["e_y_m"], f["e_psi_rad"], f["hold_s"], f["max_steer_rate_dps"])


def test_digest_facts_present():
    d = CF.digest(AV, _ctx("bias"), list(CF.TOPICS))
    assert "radians(90)" in d and "0.0785 rad" in d and ">= 0.15" in d
    assert d.index("'stale_sequence'") < d.index("'steer_rate_exceeded'") < d.index("'tracking_divergence'")
    assert "self._shift(value, self.config.magnitude)" in d and "MPCOutput fields replaced: steer" in d
    assert "self.rate_dps = 60" in d and "self.last.steer - limit" in d
    assert "step(): self.active = 'lqr'" in d and "__init__(): self.active = 'mpc'" in d
    legacy = CF.digest(AV, _ctx("bias", "legacy"), ["supervisor"])
    assert "check disabled in this profile: max_steer_rate_dps is None" in legacy and "0.0785" not in legacy


@pytest.mark.parametrize("mode", FAULT_MODES)
@pytest.mark.parametrize("sup", ("tight", "legacy"))
def test_digest_has_no_answer_options_or_outcomes(cfg, bank, mode, sup):
    """Every request of the digest context: no option name of any question it asks (with _ or space), no
    'unknown', no outcome field, and the scenario block passes the independence check."""
    qs, meta = J.judge_questions(bank, cfg, J.state_views(cfg, "laya"), context="digest")
    reqs = J.make_requests(cfg, "laya", AV, _ctx(mode, sup), SUPERVISOR_PROFILES[sup],
                           J.questions_for(qs, meta, bank, mode), meta, "digest", fewshot=False)
    by_id = {q["id"]: q for q in bank}
    assert len(reqs) >= 4
    for state, rq in reqs:
        assert isinstance(state, str) and "[facts." in state and "def " not in state
        opts = sorted({str(o) for n in rq for o in by_id[meta[n]["qid"]]["answer"]})
        assert _option_hits(state, opts) == [], (mode, sup, _option_hits(state, opts))
        for k in OUTCOME_KEYS:
            assert k not in state
        for word in ("fires first", "is caught", "therefore", "so the", "blind spot"):
            assert word not in state.lower()
        assert len(state) < 4000          # a few hundred tokens


def test_digest_builder_never_reads_a_record():
    """Outcome values put into a ctx by mistake still do not reach the digest (builder reads named inputs only)."""
    ctx = dict(_ctx("drift"), switch_time=31.2, final_safe_recovery=False,
               metrics={"pre_switch_max_lateral_error_m": 0.93})
    d = CF.digest(AV, ctx, list(CF.TOPICS))
    assert "31.2" not in d and "0.93" not in d and "final_safe_recovery" not in d


def test_full_context_unchanged(cfg, bank):
    qs, meta = J.judge_questions(bank, cfg, J.state_views(cfg, "laya"))
    ctx = _ctx("bias")
    reqs = J.make_requests(cfg, "laya", AV, ctx, SUPERVISOR_PROFILES["tight"], qs, meta)
    assert len(reqs) == 1 and reqs[0][0] == J.make_state(cfg, "laya", AV, ctx, SUPERVISOR_PROFILES["tight"])
    assert cfg.get("context", "full") == "full" and not cfg.get("fewshot", False)


def test_digest_questions_point_at_facts(cfg, bank):
    qs, meta = J.judge_questions(bank, cfg, J.state_views(cfg, "laya"), context="digest")
    assert all("`code" not in q["instructions"] for q in qs.values())
    assert "`facts.supervisor`" in qs["J1__focus_supervisor"]["instructions"]
    assert set(qs) == set(J.judge_questions(bank, cfg, J.state_views(cfg, "laya"))[0])


def test_fewshot_examples_are_synthetic(cfg, bank):
    ex = J.load_fewshot(cfg)
    by_id = {q["id"]: q for q in bank}
    assert set(ex) == {"J1", "J3", "J4", "J7", "J8", "J9"}
    for qid, es in ex.items():
        assert len(es) == 2 and by_id[qid]["label_source"] == "code"
        for e in es:
            assert e["answer"] in [str(o) for o in by_id[qid]["answer"]]
            text = (e["scenario"] + " " + e["reasoning"]).lower()
            for m in FAULT_MODES:      # never a real fault mode: no example is a campaign state or its answer
                assert not re.search(r"'" + m + r"'", text), (qid, m)
    qs, meta = J.judge_questions(bank, cfg, J.state_views(cfg, "laya"), context="digest")
    reqs = J.make_requests(cfg, "laya", AV, _ctx("bias"), SUPERVISOR_PROFILES["tight"],
                           J.questions_for(qs, meta, bank, "bias"), meta, "digest", fewshot=True)
    for state, rq in reqs:
        qids = {meta[n]["qid"] for n in rq}
        if "[worked_examples]" in state:
            assert len(qids) == 1 and qids <= set(ex)
            head = state.split("[worked_examples]")[0]
            opts = [str(o) for o in by_id[next(iter(qids))]["answer"]]
            assert _option_hits(head, opts) == []


def test_ask_requests_merges_and_stub_reads_digest(cfg, bank):
    qs, meta = J.judge_questions(bank, cfg, J.state_views(cfg, "laya"), context="digest")
    stub = J.StubJev(cfg["backends"]["laya"], bank, meta, seed=0)
    rq = J.questions_for(qs, meta, bank, "saturate")
    reqs = J.make_requests(cfg, "laya", AV, _ctx("saturate"), SUPERVISOR_PROFILES["tight"], rq, meta, "digest")
    resp = J.ask_requests(stub, reqs)
    assert set(resp["answers"]) == set(rq) and resp["n_requests"] == len(reqs)
    assert resp["usage"]["input_tokens"] > 0
    parsed = J.parse_answers(resp, meta)
    assert set(parsed) == {meta[n]["qid"] for n in rq}
    json.dumps(resp)
