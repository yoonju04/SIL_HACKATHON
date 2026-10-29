"""python run.py --mock end to end (SYNTHETIC; small budget)."""
import json

import yaml

from run import Loop, main


def test_mock_loop_small(tmp_path):
    out = tmp_path / "o"
    loop = Loop(mode="mock", out=out, ablation="X-6", video_on=True, seed=1, cold_start=8, batch_size=4, rounds=2, quiet=True)
    m = loop.run()
    assert m["n_runs"] >= 10
    recs = [json.loads(l) for l in (out / "run_records.jsonl").read_text().splitlines()]
    assert all(r["synthetic_fixture"] for r in recs) if "synthetic_fixture" in recs[0] else True
    for f in ("00_prior_payload.json", "02_objectives.json", "report.md", "evaluation_metrics.json", "llm_calls.jsonl",
              "failure_table.jsonl", "round_00/07_batch.json", "round_01/03_04_pool.json"):
        assert (out / f).exists(), f
    if m["explanations"]:
        ex = [json.loads(l) for l in (out / "explanations.jsonl").read_text().splitlines()]
        for e in ex:
            r = next(r for r in recs if r["run_id"] == e["run_id"])
            assert e["monitor_labels_snapshot"]["recovery_outcome"] == r["recovery_outcome"]
        props = yaml.safe_load((out / "monitor_rule_candidates.yaml").read_text())
        assert props["meta"]["auto_apply"] is False
    assert "SYNTHETIC" in (out / "report.md").read_text()


def test_mock_cli_video_off(tmp_path):
    rc = main(["--mock", "--out", str(tmp_path / "o"), "--ablation", "X-1", "--video-explainer", "off",
               "--cold-start", "6", "--batch-size", "3", "--rounds", "1"])
    assert rc == 0
    assert not (tmp_path / "o" / "explanations.jsonl").exists()
    cfg = json.loads((tmp_path / "o" / "run_config.json").read_text())
    assert cfg["video_explainer"] is False and cfg["synthetic"] is True


def test_reserve_follows_acquisition_order_not_pool_order(tmp_path):
    # BUG-029: a rejected GP pick was replaced by the first Nemotron A candidate of the pool, not the next-best one
    loop = Loop(mode="mock", out=tmp_path / "o", ablation="X-5", video_on=False, seed=1, cold_start=4, batch_size=2,
                rounds=1, quiet=True)
    m = loop.obj_cfg[loop.objectives[0]]["metric"]
    gp = lambda mu, sd: {"gp_mean": {m: mu}, "gp_std": {m: sd}}
    pool = [dict(scenario_id="A", source="nemotron_a", **gp(1.5, 0.1)),        # far from the boundary: straddle < 0
            dict(scenario_id="B", source="space_filling", **gp(0.0, 0.9)),
            dict(scenario_id="C", source="space_filling", **gp(0.0, 0.5)),
            dict(scenario_id="P", source="space_filling", **gp(0.0, 1.0))]
    assert [c["scenario_id"] for c in loop.reserve_order(pool, [pool[3]], 1)] == ["B", "C", "A"]
    assert [c["scenario_id"] for c in loop.reserve_order(pool, [pool[3]], 0)] == ["A", "B", "C"]   # cold start: pool order


def test_prior_campaign_runs_join_the_gp_training_data(tmp_path):
    # integration4: each campaign used to start the GP from zero (4-9 points); --prior-runs adds earlier campaigns
    first = tmp_path / "first"
    Loop(mode="mock", out=first, ablation="X-5", video_on=False, seed=3, cold_start=4, batch_size=2, rounds=1, quiet=True).run()
    n_first = sum(1 for l in (first / "run_records.jsonl").read_text().splitlines() if '"sim_status": "ok"' in l)
    loop = Loop(mode="mock", out=tmp_path / "second", ablation="X-5", video_on=False, seed=4, cold_start=4, batch_size=2,
                rounds=1, quiet=True, prior_runs=[first])
    assert len(loop.prior) == n_first > 0 and loop.records == []
    assert all(c["source"] == "prior:first" for c, _ in loop.prior)
    gps = loop.fit_gp(1)
    counts = loop.stats["gp_train_counts"][-1]
    assert counts["prior_rows"] == n_first and counts["switched_indicator"] == n_first
    assert loop.obj_cfg[loop.objectives[0]]["metric"] in gps.models
    assert loop.stats["gp_kernels"][-1]["round"] == 1
