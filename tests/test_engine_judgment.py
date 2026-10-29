"""Pure judgment functions (formerly the runner skeleton --self-test). Numbers are arbitrary test values, not repo values."""
from engine.config_check import PIPELINE_FILES, cmd_check, collect_missing, load_pipeline
from engine.config import PIPELINE_DIR
from engine.judgment import (PROVISIONAL_RECOVERY, beta_schedule, candidate_lcb, classify_outcome, classify_recovery, cold_start_split,
                             compute_df, excess_over_baseline, feasibility_gate, final_safe_recovery, gp_targets,
                             lane_margin_m, lane_recovery_robustness, lateral_jerk, lcb, margin_delta_m,
                             recovery_labels, robustness_p01, robustness_p02, select_batch, travelled_distance)


def test_judgment_self_test():
    profile = {
        "controllers": {
            "lqr": {"feedback_design": {"design_speed_mps": 10.0, "design_curvature_per_m": 0.01}},
            "mpc": {"solver_runtime": {"configured_time_limit_ms": 20}},
        },
        "switching": {"failure_detection_period_ms": 20},
    }
    feats = compute_df(
        {"speed_mps": 15.0, "road_curvature_per_m": 0.02,
         "fault": {"delay_ms": 50}, "onset_time": 10.0, "curve_entry_time": 10.01},
        profile)
    d = feats.as_dict()
    assert abs(d.pop("DF-06") - 4.5 / (0.9 * 9.81)) < 1e-12   # integration3 #6: friction utilization (scale 1.0)
    assert d == {
        "DF-01": 5.0, "DF-02": 0.01, "DF-03": 30, "DF-04": True, "DF-05": 4.5,
    }, d
    assert beta_schedule(0, 10) == 2.0 and abs(beta_schedule(9, 10) - 0.5) < 1e-9
    assert lcb(0.0, 1.0, 2.0) == -2.0
    ok, tags = feasibility_gate(
        {"onset_event": "curve_entry", "nominal_lateral_acc_mps2": 5.0},
        ["curve_entry"], lateral_acc_threshold=4.0, carla_mapping_ok=True)
    assert ok and tags == ["stress_test"]
    assert robustness_p01([0.5, -0.1, 0.3]) == -0.1
    assert robustness_p02([1.0, -2.0], 3.0) == 1.0
    assert classify_outcome(simulation_invalid=False, triggered=True, switched=True,
                            final_safe=True) == "switched_safe"
    assert classify_outcome(simulation_invalid=False, triggered=True, switched=True,
                            final_safe=None) == "switched_unsafe"
    assert classify_outcome(simulation_invalid=True, triggered=True, switched=True,
                            final_safe=True) == "simulation_invalid"
    # recovery_outcome: 100틱 창 (기본 3 s 창보다 길게 줘도 됨), 1 s 유지 = 20틱
    ok_m, ok_e = [0.5] * 100, [0.1] * 100
    kw = dict(switched=True, window_complete=True, collision=False)
    assert classify_recovery(margins=ok_m, e_y=ok_e, e_psi=[0.0] * 100, **kw) == "safe_recovery"
    assert classify_recovery(margins=ok_m, e_y=[0.5] * 100, e_psi=[0.0] * 100, **kw) == "unsettled"
    dep = [-0.1] * 30 + [0.5] * 70
    assert classify_recovery(margins=dep, e_y=ok_e, e_psi=[0.0] * 100, **kw) == "recovered_after_departure"
    assert classify_recovery(margins=[0.5] * 70 + [-0.1] * 30, e_y=ok_e, e_psi=[0.0] * 100,
                             **kw) == "unrecovered"
    assert classify_recovery(margins=ok_m, e_y=ok_e, e_psi=[0.0] * 100, switched=True,
                             window_complete=True, collision=True) == "unrecovered"
    assert classify_recovery(margins=ok_m, e_y=ok_e, e_psi=[0.0] * 100, switched=True,
                             window_complete=False, collision=False) == "simulation_invalid"
    assert classify_recovery(margins=ok_m, e_y=ok_e, e_psi=[0.0] * 100, switched=False,
                             window_complete=True, collision=False) == "no_switch"
    labels = recovery_labels(acc_excess=1.0, acc_total_max=5.0, jerk_excess=2.0,
                             jerk_total_max=6.0, margin_at_switch=-0.2, safe_stop=False)
    assert labels["lat_acc_excess"] == {"primary": False, "secondary": True}
    assert labels["departed_at_switch"] is True
    assert final_safe_recovery("safe_recovery", labels) is True
    assert final_safe_recovery("unsettled", labels) is False
    assert lane_margin_m(3.5, 0.5, 0.9) == 0.35
    assert abs(lane_recovery_robustness([0.85, 0.0], [3.5, 3.5], 0.9, False) - 0.0) < 1e-9
    assert lane_recovery_robustness([0.85], [3.5], 0.9, True) == -2.0     # 팀원 SR4 충돌 값
    assert lane_recovery_robustness([5.0], [3.5], 0.9, False) == 1.0      # [-2, 1] 로 자름
    assert PROVISIONAL_RECOVERY["window_s"] == 3.0 and PROVISIONAL_RECOVERY["lat_acc_threshold_mps2"] == 4.41
    s = travelled_distance([10.0] * 5)
    assert s == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert excess_over_baseline([0.25, 1.0], [2.0, 5.0], [0.0, 0.5, 1.0], [1.0, 1.0, 1.0]) == 4.0
    assert abs(lateral_jerk([0.0, 0.5, 1.0], window_s=0.05)[2] - 10.0) < 1e-9
    assert margin_delta_m(-0.1, 0.4) == -0.5
    assert cold_start_split(24) == (12, 12)
    rules = {
        "recovery_success": {"support_direction": "negative",
                             "gp": {"target": "lane_recovery_robustness",
                                    "form": "absolute_post_switch", "scale": 0.1}},
        "switch_count": {"support_direction": "positive",
                         "gp": {"target": "switch_count",
                                "form": "paired_delta_if_fault_else_absolute", "scale": 1.0}},
    }
    z = gp_targets(rules, {"lane_recovery_robustness": -0.05, "switch_count": 4},
                   {"switch_count": 1}, has_fault=True)
    assert abs(z["recovery_success"] + 0.5) < 1e-9 and z["switch_count"] == -3.0, z
    assert gp_targets(rules, {"switch_count": 2}, None, has_fault=False) == {"switch_count": -2.0}
    assert gp_targets(rules, {"switch_count": 2}, None, has_fault=True) == {}  # baseline 없음
    metric_of = {"H-01": "recovery_success", "H-03": "switch_count"}
    cands = [
        {"id": "a", "hypothesis_ids": ["H-01"], "gp_mean": {"recovery_success": 0.0},
         "gp_std": {"recovery_success": 1.0}, "normalized_features": [0.0], "seed_scenario_id": "s1"},
        {"id": "b", "hypothesis_ids": ["H-03"], "gp_mean": {"switch_count": -1.0},
         "gp_std": {"switch_count": 0.1}, "normalized_features": [0.5], "seed_scenario_id": "s2"},
        {"id": "c", "hypothesis_ids": ["H-03"], "gp_mean": {}, "gp_std": {},
         "normalized_features": [1.0], "seed_scenario_id": "s3"},
    ]
    assert candidate_lcb(cands[2], metric_of, 2.0) is None
    batch = select_batch(cands, metric_of, beta=2.0, batch_size=4)
    assert [c["id"] for c in batch[:2]] == ["a", "b"] and len(batch) == 3


def test_config_check_loads_all_stage_yamls(capsys):
    cfg = load_pipeline(PIPELINE_DIR)
    assert set(cfg) == set(PIPELINE_FILES) and len(PIPELINE_FILES) == 15
    missing = collect_missing(cfg)
    assert all(isinstance(m, str) for m in missing)
    assert cmd_check(PIPELINE_DIR) == (2 if missing else 0)
    assert "YAML 15" in capsys.readouterr().out

