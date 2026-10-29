"""laya-judge v2: 05b B judge = 25 System One judges over J1..J9 (Laya default, Jev / Nemotron optional) + per-question
regression. Code questions (J1 J3 J4 J7 J8 J9) are labelled by the code's answer key, outcome questions (J2 J5 J6) by
finished runs; what the parameters compute is DF-07..09 (engine/judgment.py fault_dfs)."""
import json
import math
import os

import numpy as np
import pytest
import yaml

from engine import judges as J
from engine import judgment as JM
from engine import nemotron as NM
from engine.config import REPO_ROOT
from engine.fixtures.synthetic_sim import SUPERVISOR_PROFILES

J_YAML = REPO_ROOT / "pipeline" / "05b_judges" / "judges.yaml"
AV = REPO_ROOT / "pipeline" / "08_carla_paired_run" / "av_stack"


@pytest.fixture(scope="module")
def bank():
    return J.load_bank()


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(J_YAML.read_text(encoding="utf-8"))


def _q(bank, qid):
    return next(q for q in bank if q["id"] == qid)


def _ctx(mode="bias", sup="tight", **params):
    return {"fault_mode": mode, "fault_params": params, "onset_event": "straight", "planner": "pdm",
            "controller": "supervised", "supervisor": sup, "max_speed_kmh": 30.0, "route_id": "r1",
            "tire_friction_scale": 1.0}


def _rec(**kw):
    r = {"sim_status": "ok", "switch_time": None, "actual_fault_onset": 10.0, "metrics": {}, "recovery_labels": {},
         "handover_saturation": {}}
    r.update(kw)
    return r


# ------------------------------------------------------------ bank: no numeric questions, 25 judges
def test_bank_j1_to_j9(bank, cfg):
    ids = [q["id"] for q in bank]
    assert sorted(ids) == [f"J{i}" for i in range(1, 10)]
    src = {q["id"]: q["label_source"] for q in bank}
    assert {k for k, v in src.items() if v == "outcome"} == {"J2", "J5", "J6"}
    assert {q["id"] for q in bank if q.get("regress") is False} == {"J8", "J9"}
    for q in bank:
        assert ("outcome_label" in q) == (q["label_source"] == "outcome")
        assert (q.get("expected_from_code") is not None) == (q["label_source"] == "code")
        assert "rad" not in str(q["answer"]) and not any(str(o)[0].isdigit() and "s" not in str(o) for o in q["answer"])
    assert sum(len(v) for v in cfg["judges"].values()) == 25 and set(cfg["judges"]) == set(ids)


def test_25_judges_and_applies_to(bank, cfg):
    for backend in J.BACKENDS:
        qs, meta = J.judge_questions(bank, cfg, J.state_views(cfg, backend))
        assert len(qs) == 25 and set(qs) == {f"{q}__{v}" for q, vs in cfg["judges"].items() for v in vs}
        for name, jq in qs.items():
            q = _q(bank, meta[name]["qid"])
            assert jq["instructions"].startswith(str(q["question"] if meta[name]["variant"] == "ko" else q["question_en"]))
            assert sorted(meta[name]["keys"][k] for k in jq["criteria"]) == sorted(str(o) for o in q["answer"])
            assert not set(jq["criteria"]) & {"yes", "no", "true", "false"}
    qs, meta = J.judge_questions(bank, cfg)
    n3 = len(cfg["judges"]["J3"])
    assert len(J.questions_for(qs, meta, bank, "saturate")) == 25
    assert len(J.questions_for(qs, meta, bank, "intermittent_bias")) == 25
    assert len(J.questions_for(qs, meta, bank, "bias")) == 25 - n3
    assert list(qs["J1__en_notes_rev"]["criteria"]) == list(qs["J1__en_notes"]["criteria"])[::-1]


def test_compact_focus_falls_back_to_code(bank, cfg):
    views = J.state_views(cfg, "laya")
    assert "mpc" not in views and "supervisor" in views
    qs, _ = J.judge_questions(bank, cfg, views)
    assert "code.supervisor" in qs["J1__focus_supervisor"]["instructions"]


# ------------------------------------------------------------ state: inputs only, never outcomes
def test_state_has_no_truth(bank, cfg):
    ctx = _ctx("bias", magnitude_rad=0.09)
    for backend in J.BACKENDS:
        blob = json.dumps(J.make_state(cfg, backend, AV, ctx, SUPERVISOR_PROFILES["tight"]))
        assert "expected_from_code" not in blob and "switch_time" not in blob and "outcome_label" not in blob
        assert "magnitude_rad" in blob and "max_steer_rate_dps" in blob
    with pytest.raises(ValueError):
        J.build_state({}, dict(ctx, fault_params={"expected_from_code": "x"}), None)
    with pytest.raises(ValueError):
        J.build_state({}, ctx, {"final_safe_recovery": True})


def test_compact_state_is_short_and_shows_the_switch(cfg):
    sat = J.make_state(cfg, "laya", AV, _ctx("saturate", level=0.7, ramp_s=0.0, duration_s=0.4, magnitude=1.0),
                       SUPERVISOR_PROFILES["tight"])
    bias = J.make_state(cfg, "laya", AV, _ctx("bias", magnitude_rad=0.05), SUPERVISOR_PROFILES["tight"])
    assert isinstance(sat, str) and len(sat) < 4500 and len(bias) < 4500
    assert "steer_rate_exceeded" in sat and "[code.switch]" in sat
    assert "def _saturate" in sat and "def _saturate" not in bias
    assert J._scenario_from_text(bias)["fault_params"] == {"magnitude_rad": 0.05}


# ------------------------------------------------------------ labels: code key (profile aware) + run outcomes
def test_code_keys_profile_aware(bank):
    k = lambda qid, ctx: NM.expected_answers(_q(bank, qid), ctx)
    assert k("J1", _ctx("bias", magnitude_rad=0.09)) == ["immediate_check"]
    assert k("J1", _ctx("bias", magnitude_rad=0.05)) == ["deviation_accumulation"]
    assert k("J1", _ctx("output_delay", delay_ticks=3)) == ["no_dedicated_check"]
    assert k("J4", _ctx("bias", magnitude_rad=0.09)) == ["lqr_early"]
    assert k("J4", _ctx("bias", "legacy", magnitude_rad=0.09)) == ["corrupted_mpc"]
    assert k("J7", _ctx("bias", magnitude_rad=0.09)) == ["continuous_from_clean"]
    assert k("J7", _ctx("bias", magnitude_rad=0.05)) == ["continuous_from_faulted"]
    assert k("J3", _ctx("intermittent_bias", "legacy", magnitude_rad=0.05, on_s=0.2)) == ["can_evade"]
    assert k("J8", _ctx("drift")) == ["mpc_output_only"] and k("J9", _ctx("stale")) == ["latched_no_return"]


def test_outcome_labels(bank):
    j2, j5, j6 = (_q(bank, q) for q in ("J2", "J5", "J6"))
    assert J.outcome_label(j2, _rec(switch_time=10.3)) == "under_0_5s"
    assert J.outcome_label(j2, _rec(switch_time=11.0)) == "0_5_to_2s"
    assert J.outcome_label(j2, _rec(switch_time=13.0)) == "2_to_5s"
    assert J.outcome_label(j2, _rec()) == "over_5s_or_never"
    assert J.outcome_label(j2, _rec(sim_status="error", switch_time=10.1)) is None
    assert J.outcome_label(j5, _rec(metrics={"pre_switch_max_lateral_error_m": 0.1})) == "low"
    assert J.outcome_label(j5, _rec(metrics={"pre_switch_max_lateral_error_m": 0.5})) == "high"
    assert J.outcome_label(j5, _rec(recovery_labels={"departed_at_switch": True})) == "near_or_departed"
    assert J.outcome_label(j6, _rec()) is None                                   # no switch -> no LQR command
    assert J.outcome_label(j6, _rec(switch_time=11, handover_saturation={"rate_saturation_ratio": 0.2})) == "rate_limited"
    assert J.outcome_label(j6, _rec(switch_time=11, handover_saturation={"rate_saturation_ratio": 0.0})) == "within_limit"
    lab = J.labels_for(bank, _ctx("bias", magnitude_rad=0.09), _rec(switch_time=10.1))
    assert lab["J1"] == ["immediate_check"] and lab["J2"] == ["under_0_5s"] and lab["J3"] is None
    assert J.labels_for(bank, _ctx("bias", magnitude_rad=0.09))["J2"] is None   # before the run: no outcome label


# ------------------------------------------------------------ DF-07..09: computable, not asked
def test_fault_dfs():
    step = math.radians(90) * 0.05
    d = JM.fault_dfs({"mode": "bias", "params": {"magnitude_rad": 0.09}}, "tight")
    assert d["DF-07"] == pytest.approx(0.09 / step) and d["DF-07"] > 1 and d["DF-09"] == 0.0 and d["DF-08"] is None
    d = JM.fault_dfs({"mode": "bias", "params": {"magnitude_rad": 0.05}}, "tight")
    assert d["DF-07"] < 1 and d["DF-09"] == pytest.approx(0.05 / (math.radians(60) * 0.05))
    assert JM.fault_dfs({"mode": "bias", "params": {"magnitude_rad": 0.09}}, "legacy")["DF-09"] > 0
    d = JM.fault_dfs({"mode": "saturate", "params": {"level": 0.5, "ramp_s": 0.5, "duration_s": 0.3}}, "tight")
    assert d["DF-07"] == pytest.approx(0.5 * 1.2217 * 0.1 / step) and d["DF-08"] == pytest.approx(2.0)
    assert JM.fault_dfs({"mode": "intermittent_bias", "params": {"magnitude_rad": 0.03, "on_s": 0.3}}, "legacy")["DF-08"] \
        == pytest.approx(1.2)
    assert JM.fault_dfs({"mode": "drift", "params": {"rate_rad_s": 0.1}})["DF-07"] == pytest.approx(0.005 / step)
    assert JM.fault_dfs({"mode": "output_delay", "params": {"delay_ticks": 3}})["DF-07"] is None
    assert JM.fault_dfs({"mode": "solver_fail", "params": {}}) == {"DF-07": 0.0, "DF-08": None, "DF-09": None}


# ------------------------------------------------------------ clients
def test_stub_is_deterministic_and_api_shaped(bank, cfg):
    qs, meta = J.judge_questions(bank, cfg, J.state_views(cfg, "laya"))
    st = J.make_state(cfg, "laya", AV, _ctx("saturate", level=0.7, ramp_s=0.0, duration_s=0.4), SUPERVISOR_PROFILES["tight"])
    a = J.StubJev(cfg["backends"]["laya"], bank, meta, seed=0).system_one(st, qs)
    assert a == J.StubJev(cfg["backends"]["laya"], bank, meta, seed=0).system_one(st, qs)
    assert len(a["answers"]) == 25 and a["model"] == "stub/judge"
    for name, ans in a["answers"].items():
        assert set(ans["probabilities"]) == set(qs[name]["criteria"]) and ans["choice"] in qs[name]["criteria"]


def test_make_client_fallbacks(bank, cfg, monkeypatch):
    _, meta = J.judge_questions(bank, cfg)
    c, label = J.make_client("laya", cfg, "mock", bank, meta)
    assert c.stub and label == "stub (mock)"
    monkeypatch.setenv("LAYA_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(J, "LayaInProcess", lambda *a, **k: (_ for _ in ()).throw(ImportError("no laya")))
    c, label = J.make_client("laya", cfg, "live", bank, meta)
    assert c.stub and "not reachable" in label
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    c, label = J.make_client("jev", cfg, "live", bank, meta)
    assert c.stub and "TYPESAFE_API_KEY" in label
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    c, label = J.make_client("nemotron", cfg, "live", bank, meta)
    assert c.stub and "NEBIUS_API_KEY" in label
    live = J.SystemOneClient(cfg["backends"]["laya"], backend="laya")
    assert live.endpoint == "http://127.0.0.1:9/v1/systemone" and live.has_key
    assert J.SystemOneClient(cfg["backends"]["jev"], backend="jev").endpoint == "https://api.typesafe.ai/v1/systemone"


class _FakeChat:
    has_key = True

    def __init__(self):
        self.calls = []

    def chat(self, model, messages, **kw):
        from engine.llm import ChatResult
        self.calls.append(kw)
        props = kw["response_format"]["json_schema"]["schema"]["properties"]
        obj = {n: {o: (3.0 if i == 0 else 1.0) for i, o in enumerate(p["required"])} for n, p in props.items()}
        return ChatResult(content=json.dumps(obj), reasoning=None, finish_reason="stop",
                          usage={"prompt_tokens": 100, "completion_tokens": 50}, model=model, latency_s=0.0)


def test_nemotron_system_one_shape(bank, cfg):
    llm = yaml.safe_load((REPO_ROOT / "pipeline" / "llm_models.yaml").read_text(encoding="utf-8"))
    qs, meta = J.judge_questions(bank, cfg, J.state_views(cfg, "nemotron"))
    fake = _FakeChat()
    c = J.NemotronSystemOne(cfg["backends"]["nemotron"], llm, client=fake)
    r = c.system_one("[scenario]\nfault_mode: bias", qs)
    assert set(r["answers"]) == set(qs) and r["usage"]["input_tokens"] == 100
    a = r["answers"]["J1__en_notes"]
    assert a["choice"] == list(qs["J1__en_notes"]["criteria"])[0] and sum(a["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert fake.calls[0]["response_format"]["type"] == "json_schema"
    p = J.parse_answers(r, meta)
    assert set(p) == {q["id"] for q in bank}


def test_client_never_logs_key(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test-SECRET")
    c = J.SystemOneClient(dict(cfg["backends"]["jev"], base_url="http://127.0.0.1:9", max_retries=0, timeout_s=1),
                          tmp_path / "calls.jsonl", backend="jev")
    with pytest.raises(J.JevError):
        c.system_one("s", {"q": {"type": "noul", "instructions": "x"}})
    assert "SECRET" not in (tmp_path / "calls.jsonl").read_text()


# ------------------------------------------------------------ parse + regression + aggregation
def test_parse_answers_maps_keys(bank, cfg):
    qs, meta = J.judge_questions(bank, cfg)
    crit = list(qs["J1__en_notes"]["criteria"])
    resp = {"answers": {"J1__en_notes": {"type": "choice", "choice": crit[0],
                                         "probabilities": {k: (0.7 if i == 0 else 0.1) for i, k in enumerate(crit)}}}}
    p = J.parse_answers(resp, meta)
    assert sum(p["J1"][0]["probs"].values()) == pytest.approx(1.0)
    assert p["J1"][0]["probs"][meta["J1__en_notes"]["keys"][crit[0]]] == pytest.approx(0.7)


def _synthetic_rows(n=40, seed=0):
    rng = np.random.default_rng(seed)
    opts = ["a", "b", "c"]
    rows = []
    for i in range(n):
        t = opts[i % 3]
        js = []
        for v in ("good", "flipped", "noise"):
            if v == "good":
                p = {o: (0.8 if o == t else 0.1) for o in opts}
            elif v == "flipped":       # systematically names the wrong option: regression learns to invert it
                w = opts[(opts.index(t) + 1) % 3]
                p = {o: (0.9 if o == w else 0.05) for o in opts}
            else:
                r = rng.dirichlet([1, 1, 1])
                p = dict(zip(opts, r.tolist()))
            js.append({"judge": v, "probs": p, "confidence": None})
        rows.append(({"Q": js}, {"Q": [t]}))
    return rows


def test_regression_cv_beats_mean_on_a_biased_judge():
    rows = _synthetic_rows()
    reg = J.JudgeRegression({"C": 1.0, "cv_folds": 5, "min_labels": 8}, {"Q": ["good", "flipped", "noise"]},
                            {"Q": ["a", "b", "c"]}).fit(rows)
    r = reg.report["Q"]
    assert r["method"] == "logistic_l2" and r["cv_folds"] == 5
    assert r["regression"]["accuracy"] >= 0.95 and r["regression"]["accuracy"] > r["judge_mean"]["accuracy"]
    assert r["regression"]["brier"] < r["judge_mean"]["brier"] and r["regression"]["ece"] is not None


def test_regression_falls_back_to_mean_with_few_labels():
    rows = _synthetic_rows(5)
    reg = J.JudgeRegression({"min_labels": 8}, {"Q": ["good", "flipped", "noise"]}, {"Q": ["a", "b", "c"]}).fit(rows)
    assert reg.report["Q"]["method"] == "mean" and reg.models["Q"] is None
    one = J.JudgeRegression({"min_labels": 2}, {"Q": ["good"]}, {"Q": ["a", "b", "c"]}).fit(
        [({"Q": [{"judge": "good", "probs": {"a": 0.9, "b": 0.05, "c": 0.05}}]}, {"Q": ["a"]})] * 10)
    assert one.report["Q"]["method"] == "mean"                         # one class only -> no fit


def test_report_only_is_not_regressed():
    rows = _synthetic_rows()
    reg = J.JudgeRegression({"min_labels": 8}, {"Q": ["good", "flipped", "noise"]}, {"Q": ["a", "b", "c"]},
                            report_only={"Q"}).fit(rows)
    assert reg.report["Q"]["method"] == "report_only" and reg.models.get("Q") is None


def test_aggregate_features_and_summary(bank, cfg):
    qs, meta = J.judge_questions(bank, cfg)
    variants_of = {q: list(v) for q, v in cfg["judges"].items()}
    options_of = {q: [str(o) for o in _q(bank, q)["answer"]] for q in variants_of}
    sources = {q["id"]: q["label_source"] for q in bank}
    stub = J.StubJev(cfg["backends"]["laya"], bank, meta, seed=0)
    rows = []
    cases = [("bias", {"magnitude_rad": 0.05}), ("bias", {"magnitude_rad": 0.09}), ("solver_fail", {}),
             ("drift", {"rate_rad_s": 0.04}), ("saturate", {"level": 0.7, "ramp_s": 0.0, "duration_s": 0.4})] * 3
    for i, (mode, par) in enumerate(cases):
        ctx = _ctx(mode, **par)
        st = J.make_state(cfg, "laya", AV, ctx, SUPERVISOR_PROFILES["tight"])
        p = J.parse_answers(stub.system_one(st, J.questions_for(qs, meta, bank, mode)), meta)
        rows.append((p, J.labels_for(bank, ctx, _rec(switch_time=10.0 + 0.4 * (i % 4)))))
    reg = J.JudgeRegression(cfg["regression"], variants_of, options_of, report_only={"J8", "J9"}).fit(rows)
    ents = [{"answers": J.aggregate(p, reg, key, 0.5, sources)} for p, key in rows]
    a = ents[1]["answers"]
    assert a["J1"]["expected_from_code"] == ["immediate_check"] and "J3" not in a and "J3" in ents[4]["answers"]
    assert a["J2"]["label_source"] == "outcome" and a["J1"]["label_source"] == "code"
    assert reg.report["J8"]["method"] == "report_only"
    x, names = J.jev_features(a, bank)
    assert len(x) == len(names) and "J2=ordinal" in names and not any(n.startswith(("J8", "J9")) for n in names)
    assert not any("unknown" in n for n in names) and all(0 <= v <= 1 for v in x)
    drop = J.constant_columns(ents, bank)
    _, n2 = J.jev_features(a, bank, drop)
    assert len(n2) == len(names) - len(drop)
    s = J.summary(ents, reg)
    assert 0 <= s["agreement_with_code"] <= 1 and 0 <= s["agreement_with_outcome"] <= 1


# ------------------------------------------------------------ run.py switch
def test_loop_judge_switch(tmp_path):
    from run import Loop
    lp = Loop(mode="mock", out=tmp_path / "o", ablation="X-5", video_on=False, seed=0, quiet=True)
    assert lp.judge_b == "laya" and [q["id"] for q in lp.questions][0] == "J1"
    lp._setup_clients()
    assert lp.jev_client.stub and len(lp.jev_questions) == 25
    c = {"fault": {"mode": "bias", "params": {"magnitude_rad": 0.09}}, "onset_event": "straight", "route_id": "r9",
         "environment": {"max_speed_kmh": 40.0, "tire_friction_scale": 0.8}}
    assert lp.b_key(c).endswith("|v#2|r9")
    assert "switch_time" not in json.dumps(lp.judge_ctx(c))
    nb = Loop(mode="mock", out=tmp_path / "o2", ablation="X-5", video_on=False, seed=0, quiet=True, judge_b="nemotron_b")
    assert nb.questions[0]["id"].startswith("Q-B") and "|v#" not in nb.b_key(c)
    assert Loop(mode="mock", out=tmp_path / "o4", ablation="X-5", video_on=False, seed=0, quiet=True,
                judge_b="nemotron").questions[0]["id"] == "J1"
    with pytest.raises(SystemExit):
        Loop(mode="mock", out=tmp_path / "o3", ablation="X-5", video_on=False, seed=0, quiet=True, judge_b="gpt")


# ------------------------------------------------------------ live (skipped unless a backend is there)
def _laya_up(cfg):
    return J.SystemOneClient(cfg["backends"]["laya"], backend="laya").reachable()


@pytest.mark.skipif(os.environ.get("LAYA_LIVE_TEST") != "1", reason="set LAYA_LIVE_TEST=1 with laya-serve running")
def test_live_laya_one_request(bank, cfg):
    if not _laya_up(cfg):
        pytest.skip("laya-serve not reachable")
    qs, meta = J.judge_questions(bank, cfg, J.state_views(cfg, "laya"))
    qs = {k: v for k, v in qs.items() if k.startswith("J9")}
    st = J.make_state(cfg, "laya", AV, _ctx("bias", magnitude_rad=0.09), SUPERVISOR_PROFILES["tight"])
    r = J.SystemOneClient(cfg["backends"]["laya"], backend="laya").system_one(st, qs)
    assert set(r["answers"]) == set(qs) and not (r.get("usage") or {}).get("truncated")
    assert J.parse_answers(r, meta)["J9"]


@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set")
def test_live_jev_one_tiny_request(bank, cfg):
    qs, meta = J.judge_questions(bank, cfg, J.state_views(cfg, "jev"))
    qs = {"J9__en_notes": qs["J9__en_notes"]}
    r = J.SystemOneClient(cfg["backends"]["jev"], backend="jev").system_one("[scenario]\nfault_mode: bias", qs)
    assert r["answers"]["J9__en_notes"]["type"] == "choice"
