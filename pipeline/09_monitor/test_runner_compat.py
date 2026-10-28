"""Integration against actual new_pipeline sources, not a mock runner.

Set SIL_RUNNER_ROOT to a checkout containing pipeline/engine/{monitor,judgment}.py.
No simulation or API calls. Fixtures below are explicitly synthetic av_stack logs.
"""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import monitor
import runner_adapter
from test_monitor import trace, drift, CFG, ONSET

ROOT = os.environ.get("SIL_RUNNER_ROOT")


def folder(path, tr, *, fault=True, controls=True, max_steer=1.2):
    path.mkdir()
    def rows(name, values):
        (path/name).write_text("".join(json.dumps(v)+"\n" for v in values),encoding="utf-8")
    rows("steps.jsonl", [dict(t=float(t),vx=float(tr.vx[i]),vy=0,yaw_rate=0,
                              e_y_lane=float(tr.lane_offset_m[i]),e_psi=0) for i,t in enumerate(tr.t)])
    rows("behavior.jsonl", [dict(t=float(t),active_controller=tr.active_controller[i],
                                 supervisor_reason="solver_failure" if tr.active_controller[i]=="lqr" else None,
                                 fault_injected=bool(tr.fault_active[i])) for i,t in enumerate(tr.t)])
    rows("monitor_probe.jsonl", [dict(t=float(t),lane_offset_m=float(tr.lane_offset_m[i]),
                                      lane_heading_err_rad=0,lane_width_m=3.5,ay_mps2=0,
                                      bbox_half_width_m=.95,bbox_half_length_m=2.4) for i,t in enumerate(tr.t)])
    if controls:
        rows("agent_controls.jsonl",[dict(t=float(t),steer=float(tr.steer_cmd_rad[i])/max_steer,
                                          tracking_error=float(tr.lane_offset_m[i]),heading_error=0)
                                     for i,t in enumerate(tr.t)])
    (path/"agent_config.json").write_text(json.dumps(dict(dt=.05,planner="pdm",mpc_fault="bias" if fault else "none",
                fault_start=ONSET,fault_duration=20,geometry=dict(max_steer_rad=max_steer))),encoding="utf-8")
    (path/"collisions.json").write_text(json.dumps([dict(t=t) for t in tr.collisions_t]),encoding="utf-8")
    (path/"summary.json").write_text(json.dumps(dict(supervisor=dict(safe_stop=False))),encoding="utf-8")
    return path


class LogInputTests(unittest.TestCase):
    def test_normalized_command_is_converted_using_logged_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            tr=trace(fault_at=ONSET,switch_at=10.2,steer_fn=lambda t: .03 if t>=10.2 else 0)
            p=folder(Path(temp)/"run",tr)
            loaded=monitor.load_av_stack_run(p)
            np.testing.assert_allclose(loaded.steer_cmd_rad,tr.steer_cmd_rad)
            self.assertEqual(loaded.max_steer_rad,1.2)

    def test_missing_command_or_angle_scale_is_never_zero_filled(self):
        with tempfile.TemporaryDirectory() as temp:
            p=folder(Path(temp)/"run",trace(fault_at=ONSET),controls=False)
            self.assertIsNone(monitor.load_av_stack_run(p).steer_cmd_rad)
            p=folder(Path(temp)/"no_scale",trace(fault_at=ONSET))
            c=json.loads((p/"agent_config.json").read_text()); c.pop("geometry")
            (p/"agent_config.json").write_text(json.dumps(c))
            self.assertIsNone(monitor.load_av_stack_run(p).steer_cmd_rad)


@unittest.skipUnless(ROOT,"set SIL_RUNNER_ROOT for actual runner integration")
class RunnerIntegrationTests(unittest.TestCase):
    def compare(self, fault, *, controls=True):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            f=folder(root/"fault",fault,controls=controls)
            b=folder(root/"baseline",trace(),fault=False)
            runner=runner_adapter.load_runner(ROOT)
            candidate=dict(run_id="fixture",scenario_id="S1",hypothesis_ids=["H-01"])
            rules={"recovery_success": {"support_direction":"negative", "gp":{"target":"lane_recovery_robustness", "form":"absolute_post_switch", "scale":.1}},
                   "acc_excess": {"support_direction":"positive", "gp":{"target":"lat_acc_excess_mps2", "form":"absolute_post_switch", "scale":1.0}}}
            expected=runner.compute_run_record(runner.load_av_stack_run(f),runner.load_av_stack_run(b),
                                              candidate=candidate,run_id="fixture",metric_rules=rules)
            got=runner_adapter.evaluate_folder(f,b,runner_root=ROOT,cfg=CFG,run_info=candidate,metric_rules=rules)
            self.assertEqual(tuple(got),runner.J.RUN_RECORD_FIELDS)
            self.assertEqual(len(got),37)
            stripped=copy.deepcopy(got); detail=stripped["metrics"].pop("monitor")
            self.assertEqual(stripped,expected)  # every original field/metric/label survives exactly
            json.dumps(got,allow_nan=False)
            return got,detail

    def test_safe_recovery_preserves_all_37_fields(self):
        rec,d=self.compare(trace(drift(.3,.2),fault_at=ONSET,switch_at=10.2))
        self.assertEqual(rec["recovery_outcome"],"safe_recovery")
        self.assertEqual(d["failure_type"],"pass")
        self.assertTrue(d["assessment_complete"])
        self.assertIn("recovery_success",rec["gp_targets"])
        self.assertEqual(monitor.view(rec)["runner_outcome"],rec["outcome_category"])

    def test_sr6_failure_is_visible_even_when_runner_recovery_is_safe(self):
        rec,d=self.compare(trace(drift(.3,.2),fault_at=ONSET,switch_at=10.2,
                                steer_fn=lambda t: math.radians(10) if t>=12 else 0))
        self.assertEqual(rec["outcome_category"],"switched_safe")
        self.assertEqual(d["failure_type"],"F3_not_recovered")
        self.assertFalse(d["sr"]["SR6"]["pass"])

    def test_missing_commands_marks_sr_incomplete(self):
        rec,d=self.compare(trace(fault_at=ONSET,switch_at=10.2),controls=False)
        self.assertFalse(d["assessment_complete"])
        self.assertIn("SR2",d["unassessed_requirements"])

    def test_incomplete_runner_recovery_window_is_not_safe(self):
        rec,d=self.compare(trace(T=14,fault_at=ONSET,switch_at=10.2))
        self.assertEqual(rec["recovery_outcome"],"simulation_invalid")
        self.assertEqual(rec["outcome_category"],"simulation_invalid")

    def test_collision_stays_unsafe(self):
        rec,d=self.compare(trace(fault_at=ONSET,switch_at=10.2,collide_at=12))
        self.assertEqual(rec["recovery_outcome"],"unrecovered")
        self.assertEqual(rec["outcome_category"],"switched_unsafe")
        self.assertFalse(d["sr"]["SR4"]["pass"])


if __name__=="__main__":
    unittest.main()
