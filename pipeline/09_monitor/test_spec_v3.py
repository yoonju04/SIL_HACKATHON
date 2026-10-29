"""Analytic and regression checks for the v3 safety/diagnostic separation."""
import copy
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
import monitor
import explain
import feedback
from test_monitor import CFG, ONSET, trace, drift, dangerous_no_reaction


class SpecV3Tests(unittest.TestCase):
    def grade(self, fault, cfg=CFG):
        return monitor.view(monitor.evaluate_pair(fault, trace(), ONSET, cfg, dangerous_no_reaction(), {'run_id':'r'}))

    def test_spec_evidence_roles_and_provenance_are_loaded(self):
        path = Path(__file__).with_name('safety_spec.yaml')
        cfg = monitor.load_spec(path, lane_width_default_m=3.5, vehicle_half_width_m=.95,
                                vehicle_half_length_m=2.4, max_steer_rad=1.2)
        self.assertEqual(cfg.spec_provenance['schema_version'],'3.1')
        self.assertEqual(len(cfg.spec_provenance['sha256']),64)
        self.assertTrue(cfg.sr6_enforce)
        self.assertEqual(cfg.comfort_reference_limits['max_abs_lateral_accel_mps2'],4.89)
        with tempfile.TemporaryDirectory() as d:
            spec=yaml.safe_load(path.read_text(encoding='utf-8'))
            spec['requirements']['SR2']['limit_dps']=0
            p=Path(d)/'bad.yaml'; p.write_text(yaml.safe_dump(spec),encoding='utf-8')
            with self.assertRaises(ValueError): monitor.load_spec(p)

    def test_ay_reference_never_reclassifies_detailed_safety_label(self):
        fault=trace(drift(.3,.2),fault_at=ONSET,switch_at=10.2,vx=15,r=.4)
        r=self.grade(fault)
        self.assertEqual(r['outcome_category'],'pass')
        ref=r['diagnostics']['comfort_reference']['comparisons']['max_abs_lateral_accel_mps2']
        self.assertAlmostEqual(ref['value'],6)
        self.assertFalse(ref['within_reference'])
        self.assertIsNone(r['sr']['SR3']['pass'])

    def test_configured_ay_limit_is_consistent_between_aux_and_team_metrics(self):
        r=self.grade(trace(drift(.3,.2),fault_at=ONSET,switch_at=10.2,vx=15,r=.2),replace(CFG,a_y_max_mps2=2.0))
        self.assertEqual(r['aux']['P02_lat_acc']['threshold_mps2'],2.0)
        self.assertAlmostEqual(r['aux']['P02_lat_acc']['absolute_robustness'],-1)
        # paired: the straight baseline peak is below 2.0, so the limit is 2.0 + 0.3 allowance
        self.assertAlmostEqual(r['aux']['P02_lat_acc']['effective_limit_mps2'],2.3)
        self.assertAlmostEqual(r['aux']['P02_lat_acc']['robustness'],-0.7)
        self.assertLess(r['metrics']['recovery_robustness'],0)

    def test_sr6_violation_after_transition_is_enforced(self):
        steer=lambda t: 0 if t<12 else math.radians(10)
        r=self.grade(trace(drift(.3,.2),fault_at=ONSET,switch_at=10.2,steer_fn=steer))
        self.assertTrue(r['sr']['SR2']['pass'])
        self.assertFalse(r['sr']['SR6']['pass'])
        self.assertEqual(r['outcome_category'],'F3_not_recovered')
        self.assertTrue(r['unsafe'])
        self.assertLess(r['sr_scores']['overall'],0)

    def test_sr6_can_be_explicitly_report_only(self):
        steer=lambda t: 0 if t<12 else math.radians(10)
        r=self.grade(trace(drift(.3,.2),fault_at=ONSET,switch_at=10.2,steer_fn=steer),replace(CFG,sr6_enforce=False))
        self.assertEqual(r['outcome_category'],'pass')
        self.assertNotIn('SR6',r['failed_requirements'])

    def test_missing_commands_are_unassessed_and_excluded_from_analysis(self):
        fault=trace(drift(.3,.2),fault_at=ONSET,switch_at=10.2)
        fault.steer_cmd_rad=None
        r=self.grade(fault)
        self.assertFalse(r['assessment_complete'])
        self.assertIn('SR2',r['unassessed_requirements'])
        ds=explain.build_dataset([r])
        self.assertEqual(ds['summary']['graded_triggered_runs'],0)
        self.assertIsNone(ds['summary']['failure_rate'])
        self.assertFalse(feedback._valid(r))

    def test_diagnostics_tracking_and_paired_baseline_are_exact(self):
        r=self.grade(trace(lambda t:.2,fault_at=ONSET,switch_at=10.2))
        phase=r['diagnostics']['phases']['post_switch']
        self.assertAlmostEqual(phase['lateral_rmse_m'],.2)
        self.assertAlmostEqual(phase['lateral_mae_m'],.2)
        self.assertAlmostEqual(r['diagnostics']['paired_deltas']['post_switch']['lateral_rmse_m'],.2)
        self.assertIsNone(phase['tlc_linear_min_s'])
        self.assertEqual(phase['tlc_status'],'not_closing_under_local_model')
        json.dumps(r['diagnostics'],allow_nan=False)

    def test_tlc_linear_drift_and_infinite_horizon_are_not_confused(self):
        cfg=replace(CFG,window_s=1,min_window_s=.5)
        fault=trace(lambda t:.1*(t-10),fault_at=ONSET,switch_at=10.2)
        r=self.grade(fault,cfg)
        self.assertAlmostEqual(r['diagnostics']['phases']['evaluation']['tlc_linear_min_s'],7,places=6)
        fault.lane_heading_err=None
        r=self.grade(fault,cfg)
        self.assertEqual(r['diagnostics']['phases']['evaluation']['tlc_status'],'unavailable_geometry')
        self.assertIsNone(r['diagnostics']['phases']['evaluation']['tlc_linear_min_s'])

    def test_low_speed_sideslip_is_not_filled_with_zero(self):
        r=self.grade(trace(drift(.3,.2),fault_at=ONSET,switch_at=10.2,vx=0))
        self.assertIsNone(r['diagnostics']['phases']['post_switch']['max_abs_sideslip_rad'])
        self.assertEqual(r['diagnostics']['phases']['post_switch']['tlc_status'],'low_speed')

    def test_jerk_uses_measured_ay_and_actual_timestamps(self):
        fault=trace(fault_at=ONSET,switch_at=10.2)
        fault.ay_mps2=.5*np.asarray(fault.t)
        r=self.grade(fault)
        self.assertAlmostEqual(r['diagnostics']['phases']['post_switch']['max_abs_lateral_jerk_mps3'],.5,places=5)

    def test_p02_is_judged_against_a_baseline_that_already_turns_hard(self):
        # baseline and fault both corner at 6 m/s2: absolute P-02 fails, paired P-02 passes
        base=trace(vx=15,r=.4)
        r=monitor.view(monitor.evaluate_pair(trace(fault_at=ONSET,switch_at=10.2,vx=15,r=.4),base,ONSET,CFG,
                                             dangerous_no_reaction(),{'run_id':'r'}))
        p02=r['aux']['P02_lat_acc']
        self.assertLess(p02['absolute_robustness'],0)
        self.assertAlmostEqual(p02['robustness'],CFG.p02_allowance_mps2,places=6)
        self.assertEqual(p02['evaluation'],'paired_baseline')
        self.assertTrue(p02['signal'].startswith('butterworth'))

    def test_r79_filter_removes_one_tick_ay_spike(self):
        fault=trace(fault_at=ONSET,switch_at=10.2)
        ay=np.zeros(len(fault.t)); ay[230]=20.0
        fault.ay_mps2=ay
        r=self.grade(fault)
        self.assertLess(r['aux']['P02_lat_acc']['max_abs_mps2'],2.0)

    def test_bug021_no_switch_uses_the_fault_run_as_its_own_no_reaction_run(self):
        grade = lambda f: monitor.view(monitor.evaluate_pair(f, trace(), ONSET, CFG, None, {'run_id': 'r'}))
        harmless = grade(trace(drift(.3, .2), fault_at=ONSET))           # never switched, stays in the lane
        self.assertEqual(harmless['outcome_category'], 'pass')
        self.assertEqual(harmless['note'], 'undetected_harmless')
        self.assertFalse(harmless['sr']['SR1']['applies'])
        dangerous = grade(trace(drift(1.0, 2.0), fault_at=ONSET))        # never switched, leaves the lane
        self.assertEqual(dangerous['outcome_category'], 'F1_no_or_late_switch')
        self.assertEqual(dangerous['sr']['SR1']['limit_source'], 'ftti_from_fault_run_no_switch')
        self.assertIsNotNone(dangerous['sr']['SR1']['ftti_s'])

    def test_bug026_rate_held_at_the_limit_is_not_a_violation(self):
        lim = math.radians(CFG.sr6_max_rate_dps) * 0.05 * (1 + 3e-7)   # one tick at the bound + round-trip error
        steer = lambda t: 0.0 if t < 10.2 else min(0.3, lim * round((t - 10.2) / 0.05))
        r = self.grade(trace(drift(.3, .2), fault_at=ONSET, switch_at=10.2, steer_fn=steer))
        self.assertTrue(r['sr']['SR6']['pass'], r['sr']['SR6'])

    def test_recovery_cannot_look_past_evaluation_window(self):
        cfg=replace(CFG,window_s=1,min_window_s=.5)
        r=self.grade(trace(fault_at=ONSET,switch_at=10.8),cfg)
        self.assertFalse(r['sr']['SR5']['pass'])
        self.assertIsNone(r['sr']['SR5']['recover_time_s'])
        self.assertEqual(r['diagnostics']['phases']['post_switch']['status'],'unavailable')

    def test_sr2_cannot_measure_beyond_evaluation_horizon(self):
        cfg=replace(CFG,window_s=1,min_window_s=.5)
        fault=trace(fault_at=ONSET,switch_at=10.8,steer_fn=lambda t:0 if t<11.1 else .2)
        r=self.grade(fault,cfg)
        self.assertTrue(r['sr']['SR2']['pass'])
        self.assertFalse(r['sr']['SR2']['coverage_complete'])
        self.assertFalse(r['assessment_complete'])
        self.assertIn('SR2_transition_coverage',r['unassessed_requirements'])

    def test_command_rate_includes_interval_into_fault_onset(self):
        fault=trace(fault_at=ONSET,switch_at=10.2,steer_fn=lambda t:0 if t<ONSET else .2)
        r=self.grade(fault)
        self.assertTrue(r['sr']['SR2']['pass'])
        self.assertFalse(r['sr']['SR6']['pass'])
        self.assertGreater(r['sr']['SR6']['max_rate_dps'],90)
        self.assertEqual(r['outcome_category'],'F3_not_recovered')

    def test_subtick_ftti_budget_must_not_be_rounded_up_to_dt(self):
        nr=trace(lambda t:1.0 if t>=10.05 else 0.0,fault_at=ONSET)
        r=monitor.view(monitor.evaluate_pair(trace(fault_at=ONSET,switch_at=10.05),trace(),ONSET,CFG,nr))
        self.assertAlmostEqual(r['sr']['SR1']['limit_s'],.025)
        self.assertFalse(r['sr']['SR1']['pass'])

    def test_zero_ftti_is_a_failure_without_division_by_zero(self):
        nr=trace(lambda t:1.0 if t>=10.0 else 0.0,fault_at=ONSET)
        r=monitor.view(monitor.evaluate_pair(trace(fault_at=ONSET,switch_at=10.0),trace(),ONSET,CFG,nr))
        self.assertEqual(r['sr']['SR1']['limit_s'],0)
        self.assertFalse(r['sr']['SR1']['pass'])
        self.assertEqual(r['sr']['SR1']['robustness'],-1)

    def test_optional_misalignment_and_nonfinite_commands_are_invalid(self):
        fault=trace(fault_at=ONSET,switch_at=10.2)
        fault.steer_cmd_rad=np.zeros(2)
        self.assertEqual(self.grade(fault)['outcome_category'],'invalid')
        fault=trace(fault_at=ONSET,switch_at=10.2)
        fault.steer_cmd_rad[220]=np.nan
        self.assertEqual(self.grade(fault)['outcome_category'],'invalid')


if __name__=='__main__': unittest.main()
