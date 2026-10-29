"""integration3 #6: Nemotron B redesign (question scope, YAML yes/no, hallucination check, citation relevance) + DF-06."""
import json
import math
import subprocess
import sys
import textwrap

import pytest
import yaml

from engine import judgment as J
from engine import nemotron as NM
from engine.config import REPO_ROOT

B_YAML = REPO_ROOT / "pipeline" / "05b_nemotron_b" / "nemotron_b.yaml"
AV_STACK = REPO_ROOT / "pipeline" / "08_carla_paired_run" / "av_stack"
MODES = ["solver_fail", "stale", "bias", "drift", "intermittent_bias", "saturate", "output_delay"]


@pytest.fixture(scope="module")
def bank():
    return NM.normalize_questions(yaml.safe_load(B_YAML.read_text(encoding="utf-8"))["B_question_answering"]["question_bank"])


def _q(bank, qid):
    return next(q for q in bank if q["id"] == qid)


def test_yaml_yes_no_are_strings(bank):
    raw = yaml.safe_load(B_YAML.read_text(encoding="utf-8"))["B_question_answering"]["question_bank"]
    for q in raw:      # the file itself is quoted (no YAML 1.1 booleans)
        assert all(isinstance(a, str) for a in q["answer"]), q["id"]
    # and the loader repairs an old unquoted bank
    old = yaml.safe_load("- {id: Q-X, answer: [yes, no, unknown], expected_from_code: no}")
    assert old[0]["answer"] == [True, False, "unknown"]
    fixed = NM.normalize_questions(old)[0]
    assert fixed["answer"] == ["yes", "no", "unknown"] and fixed["expected_from_code"] == "no"
    assert fixed["scope"] == "per_scenario" and fixed["gp_feature"] is True
    assert {q["id"]: q["scope"] for q in bank}["Q-B2"] == "static"


def test_b_features_exclude_static_and_non_gp(bank):
    answers = {"Q-B1": {"answer": "corrupted_mpc"}, "Q-B2": {"answer": "no"}, "Q-B3": {"answer": "no"},
               "Q-B4": {"answer": "tracking_divergence"}, "Q-B5": {"answer": "steer_value_only"}}
    x, names = NM.b_features(answers, bank)
    assert {n.split("=")[0] for n in names} == {"Q-B1", "Q-B4"}
    assert x[names.index("Q-B1=corrupted_mpc")] == 1.0 and x[names.index("Q-B4=tracking_divergence")] == 1.0
    assert sum(x) == 2.0
    # an old boolean answer still encodes (old campaigns / --gp-history)
    old = NM.normalize_questions([{"id": "Q-Y", "answer": [True, False, "unknown"]}])
    assert NM.b_features({"Q-Y": {"answer": False}}, old)[0] == [0.0, 1.0, 0.0]


def test_static_questions_are_not_scenario_dependent(bank):
    for q in NM.questions_of_scope(bank, "static"):
        assert not q["gp_feature"]
        assert not isinstance(q["expected_from_code"], dict)       # one fixed code fact
        assert "param_bins" not in q


def test_expected_answers_by_mode_and_param(bank):
    b1, b4 = _q(bank, "Q-B1"), _q(bank, "Q-B4")
    assert NM.expected_answers(b1, {"fault_mode": "stale"}) == ["lqr_early"]
    assert NM.expected_answers(b1, {"fault_mode": "bias"}) == ["corrupted_mpc"]
    assert NM.expected_answers(b1, {"fault_mode": "output_delay"}) == ["stale_mpc"]
    # integration3 #8 (tight supervisor): <= 0.15 s cannot satisfy the divergence hold; the rate check still can
    assert NM.expected_answers(b4, {"fault_mode": "saturate", "fault_params": {"duration_s": 0.1}}) == ["steer_rate_exceeded", "none_during_fault"]
    assert "tracking_divergence" in NM.expected_answers(b4, {"fault_mode": "saturate", "fault_params": {"duration_s": 0.2}})
    assert "steer_rate_exceeded" not in NM.expected_answers(b4, {"fault_mode": "drift"})   # 0.05 rad/s + MPC 60 deg/s < 90
    assert "steer_rate_exceeded" in NM.expected_answers(b4, {"fault_mode": "intermittent_bias"})
    assert "tracking_divergence" in NM.expected_answers(b4, {"fault_mode": "saturate", "fault_params": {"duration_s": 0.6}})
    assert "steer_out_of_range" not in NM.expected_answers(b4, {"fault_mode": "saturate", "fault_params": {"duration_s": 0.6}})
    assert NM.expected_answers(b4, {"fault_mode": "unknown_mode"}) is None
    for q in NM.questions_of_scope(bank, "per_scenario"):
        for m in MODES:
            exp = NM.expected_answers(q, {"fault_mode": m, "fault_params": {"duration_s": 0.5}})
            assert exp is None or set(exp) <= set(q["answer"]), (q["id"], m, exp)


def test_b_key_bins_only_declared_params(bank):
    assert NM.param_bin_suffix(bank, {"duration_s": 0.1, "magnitude": 0.7}) == "|duration_s#0"
    assert NM.param_bin_suffix(bank, {"duration_s": 0.9}) == "|duration_s#1"
    assert NM.param_bin_suffix(bank, {"duration_s": 0.9, "ramp_s": 0.0}) == "|duration_s#1|ramp_s#0"
    assert NM.param_bin_suffix(bank, {"duration_s": 0.9, "ramp_s": 0.1}) == "|duration_s#1|ramp_s#1"
    # laya-judge: bias magnitude is binned at the tight supervisor's 90 deg/s rate check (4.5 deg = 0.0785 rad per tick)
    assert NM.param_bin_suffix(bank, {"magnitude_rad": 0.05}) == "|magnitude_rad#0"
    assert NM.param_bin_suffix(bank, {"magnitude_rad": 0.09}) == "|magnitude_rad#1"
    assert NM.param_bin_suffix(bank, {"on_s": 0.3, "rate_rad_s": 0.04}) == ""


def test_hallucination_check(bank):
    qs = [_q(bank, "Q-B1"), _q(bank, "Q-B2")]
    sizes = {"integration/supervised_lateral.py": 178, "integration/fault_injection.py": 122}
    ev = [{"file": "integration/fault_injection.py", "line": 112}]
    def sample(a1, a2):
        return {"answers": [{"id": "Q-B1", "answer": a1, "evidence": ev, "rationale": ""},
                            {"id": "Q-B2", "answer": a2, "evidence": ev, "rationale": ""}]}
    agg = NM.aggregate_b([sample("stale_mpc", "no"), sample("stale_mpc", "unknown"), sample("lqr_early", "no")], qs, sizes,
                         ctx={"fault_mode": "stale"})
    assert agg["Q-B1"]["answer"] == "stale_mpc" and agg["Q-B1"]["matches_code"] is False and agg["Q-B1"]["hallucinated"]
    assert agg["Q-B2"]["matches_code"] is True and not agg["Q-B2"]["hallucinated"]
    unk = NM.aggregate_b([sample("unknown", "no")], qs, sizes, ctx={"fault_mode": "stale"})
    assert unk["Q-B1"]["matches_code"] is False and unk["Q-B1"]["hallucinated"] is False     # unknown is not false
    s = NM.b_hallucination_summary([agg, unk])
    assert s["b_checked_answers"] == 4 and s["b_hallucinated_answers"] == 1
    assert s["b_hallucination_rate"] == pytest.approx(0.25) and s["b_unknown_when_code_decides"] == 1
    # boolean answers from the old unquoted schema still compare
    old = NM.aggregate_b([{"answers": [{"id": "Q-B2", "answer": False, "evidence": ev}]}], [qs[1]], sizes)
    assert old["Q-B2"]["answer"] == "no" and old["Q-B2"]["matches_code"] is True


def test_citation_relevance(bank):
    regions = NM.resolve_regions(AV_STACK, bank)
    for q in bank:
        assert regions[q["id"]], f"{q['id']}: evidence_regions do not resolve in av_stack (renamed function?)"
    sizes = {"integration/supervised_lateral.py": 178, "integration/fault_injection.py": 122, "lqr_controller/lqr.py": 174}
    inside = {"evidence": [{"file": "integration/supervised_lateral.py", "line": 74}]}          # MPCSupervisor.inspect
    outside = {"evidence": [{"file": "lqr_controller/lqr.py", "line": 1}]}
    out_of_range = {"evidence": [{"file": "integration/supervised_lateral.py", "line": 9999}]}
    r4 = regions["Q-B4"]
    assert NM.check_citations(inside, sizes) and NM.citations_relevant(inside, sizes, r4) is True
    assert NM.check_citations(outside, sizes) and NM.citations_relevant(outside, sizes, r4) is False
    assert NM.citations_relevant(out_of_range, sizes, r4) is False
    assert NM.citations_relevant(inside, sizes, []) is None


def test_prompt_does_not_leak_answer_key(bank):
    qs = NM.questions_of_scope(bank, "per_scenario")
    msgs = NM.b_messages("code", qs, {"fault_mode": "saturate", "fault_params": {"duration_s": 0.2}}, NM.b_schema(qs, ["f.py"]))
    user = msgs[-1]["content"]
    assert "expected_from_code" not in user and "evidence_regions" not in user and "none_during_fault" in user
    for word in ("recovery", "lane_recovery_robustness", "switch_time", "final_safe"):
        assert word not in user


def test_stub_b_is_schema_valid_and_marked(bank):
    qs = NM.questions_of_scope(bank, "per_scenario")
    schema = NM.b_schema(qs, ["integration/supervised_lateral.py", "integration/fault_injection.py"])
    h = NM.stub_handler(0)
    out = json.loads(h("B:x:0", "stub/B", NM.b_messages("", qs, {"fault_mode": "saturate", "fault_params": {"duration_s": 0.2}},
                                                         schema), {}))
    ids = [a["id"] for a in out["answers"]]
    assert ids == [q["id"] for q in qs]
    for a, q in zip(out["answers"], qs):
        assert a["answer"] in q["answer"] and "SYNTHETIC" in a["rationale"]


def test_df06_friction_utilization():
    profile = {}
    f = J.compute_df({"speed_mps": 10.0, "road_curvature_per_m": 0.05, "tire_friction_scale": 0.5}, profile)
    assert f.df05_nominal_lateral_acc_mps2 == pytest.approx(5.0)
    assert f.df06_friction_utilization == pytest.approx(5.0 / (0.9 * 0.5 * 9.81))
    d = J.compute_df({"speed_mps": 10.0, "road_curvature_per_m": 0.05}, profile)
    assert d.df06_friction_utilization == pytest.approx(5.0 / (0.9 * 9.81))        # default scale 1.0
    m = J.compute_df({"speed_mps": 10.0, "road_curvature_per_m": None, "tire_friction_scale": 0.5}, profile).as_dict()
    assert m["DF-06"] is None and m["DF-06_missing"] is True and m["DF-05_missing"] is True
    bad = J.compute_df({"speed_mps": 10.0, "road_curvature_per_m": 0.05, "tire_friction_scale": 0.0}, profile).as_dict()
    assert bad["DF-06"] is None and bad["DF-06_missing"] is True


@pytest.mark.skipif(not (AV_STACK / "integration" / "supervised_lateral.py").is_file(), reason="av_stack not merged")
def test_expected_answers_match_the_real_supervisor(bank):
    """Ground the answer key in the code itself: feed the real injector + supervisor synthetic healthy MPC outputs
    and check the first trigger on the first faulty tick (subprocess: av_stack has its own top-level packages)."""
    script = textwrap.dedent("""
        import json, numpy as np
        from integration.fault_injection import FaultConfig, MPCFaultInjector
        from integration.supervised_lateral import MPCSupervisor
        from mpc_controller import MPCOutput
        MAX = 0.5
        def out(seq, t, steer=0.1):
            return MPCOutput(steer=steer, raw_steer=steer, ok=True, status='solved', solve_time_s=0.001, qp_iterations=3,
                             seq=seq, stamp_s=t, state=np.zeros(4), s=0.0, kappa=0.0)
        res = {}
        for mode, kw in [('solver_fail', {}), ('stale', {}), ('bias', {'magnitude': 0.1}), ('drift', {'magnitude': 0.05}),
                         ('intermittent_bias', {'magnitude': 0.1}), ('saturate', {'magnitude': 1.0, 'level': 1.0}),
                         ('output_delay', {'magnitude': 3})]:
            inj = MPCFaultInjector(FaultConfig(mode=mode, at_s=1.0, duration_s=0.2, **kw), MAX)
            sup = MPCSupervisor(MAX, deadline_ms=50.0)
            first = None
            for k in range(30):
                t = round(k * 0.05, 6)
                o = inj.apply(out(k + 1, t, steer=0.45), t)
                r = sup.inspect(o, t, 1.0)
                if r and first is None:
                    first = (r, t)
            res[mode] = first
        print(json.dumps(res))
    """)
    p = subprocess.run([sys.executable, "-c", script], cwd=AV_STACK, capture_output=True, text=True, timeout=60)
    if p.returncode != 0 and "ModuleNotFoundError" in p.stderr:
        pytest.skip("av_stack deps missing: " + p.stderr.strip().splitlines()[-1])
    assert p.returncode == 0, p.stderr
    res = json.loads(p.stdout.strip().splitlines()[-1])
    b4 = _q(bank, "Q-B4")
    assert res["solver_fail"] == ["solver_failure", 1.0]
    assert res["stale"] == ["stale_sequence", 1.0]
    for m in ("solver_fail", "stale"):
        assert res[m][0] in NM.expected_answers(b4, {"fault_mode": m})
    # saturate at level 1.0 (exactly the limit) and output_delay never trip an instant check; with healthy reported
    # state nothing fires -> none_during_fault, which the answer key allows
    for m in ("saturate", "output_delay"):
        assert res[m] is None
        assert "none_during_fault" in NM.expected_answers(b4, {"fault_mode": m, "fault_params": {"duration_s": 0.2}})
    # bias 0.1 on a 0.45 rad command with a 0.5 rad limit -> steer_out_of_range on the first faulty tick (allowed answer)
    assert res["bias"] == ["steer_out_of_range", 1.0]
    assert res["bias"][0] in NM.expected_answers(b4, {"fault_mode": "bias"})
