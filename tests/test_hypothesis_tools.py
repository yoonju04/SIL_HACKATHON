import unittest,yaml,pathlib
from pipeline.hypothesis_tools.generate_hypotheses import safe_expr,generate,interpolate
from pipeline.hypothesis_tools.build_scenario_candidates import make
from pipeline.hypothesis_tools.space import normalize_space
ROOT=pathlib.Path(__file__).resolve().parents[1]/'pipeline'
REPO=ROOT.parent
def load(rel):return yaml.safe_load((ROOT/rel).read_text(encoding='utf-8'))
SO='02_hypotheses/search_objectives/'
class Tests(unittest.TestCase):
 def setUp(self):
  self.p=load('hypothesis_tools/examples/safety_profile.example.yaml')
  self.c=load('02_hypotheses/vulnerability_patterns.yaml')
  self.s=load('04_feasibility_expansion/scenario_space.yaml')
  # Generator mechanics are tested on the superseded (legacy) patterns; the active catalog is empty.
  legacy=sorted(self.c['superseded_patterns'],key=lambda p:p['target_id'])
  self.old={**self.c,'patterns':legacy}
 def test_safe_parser(self):
  self.assertTrue(safe_expr("15 > 3 and True"))
  with self.assertRaises(ValueError):safe_expr("__import__('os').system('echo bad')")
 def test_active_catalog_is_empty(self):
  self.assertEqual(self.c['patterns'],[])
  self.assertEqual(generate(self.p,self.c,self.s)['hypotheses'],[])
 def test_legacy_generation(self):
  x=generate(self.p,self.old,self.s)
  self.assertEqual([h['id'] for h in x['hypotheses']],[f'H-{i:02}' for i in range(1,12)])
  status={h['id']:h['status'] for h in x['hypotheses']}
  self.assertEqual(sorted(k for k,v in status.items() if v=='proposed'),['H-01','H-02','H-03','H-08'])
 def test_repo_space_has_every_dimension(self):
  x=generate(self.p,self.old,self.s)
  self.assertFalse([r for h in x['hypotheses'] for r in h.get('blocked_reason',[]) if 'scenario dimensions' in r])
 def test_repo_profile_legacy_h01_not_applicable(self):
  # sim-eval LQR is gain_scheduled -> the fixed-gain H-01 pattern does not apply
  x=generate(load('01_safety_profile/safety_profile.yaml'),self.old,self.s)
  self.assertFalse(any(h['id']=='H-01' for h in x['hypotheses']))
  self.assertFalse(next(a for a in x['pattern_application'] if a['pattern_id']=='P-OFF-DESIGN-H-01')['applies'])
 def test_missing_is_blocked(self):
  self.p['controllers']['lqr']['feedback_design']['design_speed_mps']=None
  x=generate(self.p,self.old,self.s)
  self.assertEqual(x['hypotheses'][0]['status'],'blocked')
 def test_false_not_generated(self):
  self.p['controllers']['lqr']['feedback_design']['gain_type']='scheduled'
  x=generate(self.p,self.old,self.s)
  self.assertFalse(any(h['id']=='H-01' for h in x['hypotheses']))
 def test_no_malicious_proposals(self):
  x=generate(self.p,self.old,self.s,[{'based_on':['O-001'],'condition':{},'control':{},'prediction':{'metric':'recovery_success','direction':'negative'}}])
  self.assertEqual(len(x['rejected_proposals']),1)
 def test_template_preserves_types(self):
  self.assertEqual(interpolate('{controllers.lqr.feedback_design.design_speed_mps}',self.p),15.0)
 def test_h01_thresholds_intact(self):
  h=next(h for h in generate(self.p,self.old,self.s)['hypotheses'] if h['id']=='H-01')
  self.assertEqual(h['condition']['curvature_offset'],'road_curvature_per_m - 0.0 > 0.01')
  self.assertEqual(h['control']['curvature_offset'],'road_curvature_per_m - 0.0 <= 0.005')
 def test_every_metric_has_gp_target(self):
  for name,rule in self.c['metric_rules'].items():
   self.assertIn(rule['gp']['form'],('paired_delta_if_fault_else_absolute','absolute_post_switch'),name)
   self.assertGreater(rule['gp']['scale'],0,name)
 def test_stress_tagged_not_rejected(self):
  h=next(h for h in generate(self.p,self.old,self.s)['hypotheses'] if h['id']=='H-08')
  rows=make(h,normalize_space(self.s),4,7,max_lateral_acc=1.0)
  self.assertTrue(all(r['lateral_screen']=='stress_test' and r['executable'] for r in rows))
 def test_inactive_fault_params_dropped(self):
  h=next(h for h in generate(self.p,self.old,self.s)['hypotheses'] if h['id']=='H-03')
  for r in make(h,normalize_space(self.s),4,7):
   faults={k.split('.')[0] for k in r['parameters'] if '.' in k}
   self.assertEqual(faults,{r['parameters']['fault_type']})
class SearchObjectiveTests(unittest.TestCase):
 def setUp(self):
  self.c=load('02_hypotheses/vulnerability_patterns.yaml')
  self.common=load(SO+'_common.yaml')
 def test_every_legacy_pattern_marked(self):
  ids=sorted(p['target_id'] for p in self.c['superseded_patterns'])
  self.assertEqual(ids,[f'H-{i:02}' for i in range(1,12)])
  for p in self.c['superseded_patterns']:
   if p['target_id']=='H-07':
    self.assertEqual((p['status'],p['superseded_by']),('on_hold',None))
   else:
    self.assertEqual(p['status'],'superseded')
    self.assertEqual(load('02_hypotheses/'+p['superseded_by'])['meta']['status'][:6] in ('design','option'),True)
 def test_set_files_and_sources(self):
  covered=set()
  for key,spec in self.common['set'].items():
   doc=load(SO+spec['file'])
   self.assertEqual(doc['id'],key)
   covered|=set(doc.get('legacy_ids',[]))
  self.assertEqual(covered|set(self.common['on_hold']),{f'H-{i:02}' for i in range(1,12)})
 def test_metrics_grounded(self):
  defs=self.c['outcome_definitions']
  for f in ('S1_operating_condition_recovery_boundary.yaml','S2_fault_timing_detection_delay_boundary.yaml','S3_natural_mpc_degradation.yaml'):
   doc=load(SO+f)
   for e in doc['evaluation']:
    if 'defined_in' in e:
     name=e['name']
     self.assertIn(name,defs,f+':'+name)
 def test_s1_names_exist(self):
  s1=load(SO+'S1_operating_condition_recovery_boundary.yaml')
  modes=self.common['injector_modes']
  modes=set(modes) if isinstance(modes,list) else {m for v in modes.values() for m in v}
  for m in s1['control_inputs']['degrade_fault']['mode']:self.assertIn(m,modes)
  trig=next(e for e in s1['evaluation'] if e['name']=='trigger_source')
  for names in trig['expect'].values():
   for n in names:self.assertIn(n,self.common['trigger_names'])
  self.assertTrue({'H-04','H-05'}<=set(s1['legacy_ids']))
  self.assertFalse({'H-04','H-05'}&set(load(SO+'A_switch_analysis_layer.yaml')['legacy_ids']))
 def test_s1_switch_axes_in_error_table(self):
  s1=load(SO+'S1_operating_condition_recovery_boundary.yaml')['derived_axes']
  rs=load('09_monitor/error_table.yaml')['results_schema']
  for k in ('e_y_at_switch','e_psi_at_switch','e_y_dot_at_switch','e_psi_dot_at_switch','normalized_switch_error'):
   self.assertIn(k,s1['switch_state']);self.assertIn(k,rs['switch_state_fields'])
  for k in ('angle_saturation_ratio','rate_saturation_ratio','time_in_saturation_s','predicted_first_lqr_command_ratio','required_steer_rate_ratio'):
   self.assertIn(k,s1['actuator_limits']);self.assertIn(k,rs['handover_saturation_fields'])
 def test_limitation_stated(self):
  self.assertTrue(any('수학적으로 보증' in x for x in self.common['limitations']['claims_not_made']))
  self.assertIn('성공 런과 실패 런을 모두',self.common['observation_requirement'])
class ProfileAndSchemaTests(unittest.TestCase):
 def setUp(self):
  self.prof=load('01_safety_profile/safety_profile.yaml')
  self.err=load('09_monitor/error_table.yaml')
  self.pr=load('08_carla_paired_run/paired_runs.yaml')
 def test_trigger_names_consistent(self):
  names=[t['name'] for t in self.prof['switching']['triggers']]
  self.assertEqual(names,load(SO+'_common.yaml')['trigger_names'])
  self.assertEqual(names+[None],self.err['results_schema']['trigger_source_values'])
 def test_every_filled_field_ref_has_evidence(self):
  obs={o['id'] for o in self.prof['code_evidence']['observations']}
  self.assertEqual(len(obs),len(self.prof['code_evidence']['observations']))
  for sec in (self.prof['controllers']['mpc'],self.prof['controllers']['lqr'],self.prof['switching']):
   self.assertTrue(set(sec['code_refs'])<=obs)
  for t in self.prof['switching']['triggers']:
   self.assertTrue(set(t['evidence_ids'])<=obs)
 def test_run_record_and_baseline_key_match_runner(self):
  import importlib.util
  spec=importlib.util.spec_from_file_location('runner_skeleton_t',REPO/'scripts/runner_skeleton.py')
  import sys
  m=importlib.util.module_from_spec(spec);sys.modules['runner_skeleton_t']=m;spec.loader.exec_module(m)
  self.assertEqual(list(m.RUN_RECORD_FIELDS),self.err['results_schema']['run_record'])
  self.assertEqual(list(m.BASELINE_MATCH_FIELDS),self.pr['paired_runs']['match'])
  for k in ('vehicle_mass_scale','maneuver','velocity_disturbance','steering_disturbance'):
   self.assertIn(k,self.pr['paired_runs']['match'])
  self.assertIn('trigger_source',self.err['results_schema']['run_record'])
class RecoveryJudgmentTests(unittest.TestCase):
 def setUp(self):
  self.c=load('02_hypotheses/vulnerability_patterns.yaml')
  self.od=self.c['outcome_definitions']
  self.common=load(SO+'_common.yaml')['recovery_judgment']
  import importlib.util,sys
  spec=importlib.util.spec_from_file_location('runner_skeleton_r',REPO/'scripts/runner_skeleton.py')
  self.m=importlib.util.module_from_spec(spec);sys.modules['runner_skeleton_r']=self.m;spec.loader.exec_module(self.m)
 def test_outcome_classes_consistent(self):
  classes=set(self.od['recovery_outcome']['classes'])
  self.assertEqual(classes,set(self.common['recovery_outcome']))
  self.assertEqual(classes,set(load('09_monitor/safety_spec.yaml')['safety_specification']['recovery_outcomes']))
  self.assertEqual(classes,set(load('09_monitor/error_table.yaml')['results_schema']['recovery_outcome_values']))
  spec=load('09_monitor/safety_spec.yaml')['safety_specification']
  self.assertEqual(set(spec['category_mapping']),set(spec['outcome_categories']))
 def test_provisional_values_marked_and_consistent(self):
  th=self.od['recovery_judgment']['thresholds']
  for k in ('e_settle_m','psi_settle_rad','settle_hold_s','window_s'):
   self.assertEqual(th[k]['status'],'provisional')
  pv=self.common['provisional_values'];R=self.m.PROVISIONAL_RECOVERY
  self.assertEqual((pv['e_settle_m'],pv['psi_settle_rad'],pv['settle_hold_s'],pv['window_s']),
                   (th['e_settle_m']['value'],th['psi_settle_rad']['value'],th['settle_hold_s']['value'],th['window_s']['value']))
  self.assertEqual((R['e_settle_m'],R['psi_settle_rad'],R['settle_hold_s'],R['window_s'],R['lat_acc_threshold_mps2'],R['lat_jerk_threshold_mps3']),
                   (pv['e_settle_m'],pv['psi_settle_rad'],pv['settle_hold_s'],pv['window_s'],pv['lat_acc_mps2'],pv['lat_jerk_mps3']))
  self.assertEqual(self.od['recovery_labels']['lat_acc_excess']['threshold_mps2']['status'],'provisional')
 def test_gp_targets_split(self):
  mr=self.c['metric_rules']
  self.assertEqual(mr['recovery_success']['gp']['target'],'lane_recovery_robustness')
  self.assertEqual(mr['lat_acc_excess_mps2']['gp']['target'],'lat_acc_excess_mps2')
  for k in self.common['gp_targets']:self.assertIn(k,self.od)
  self.assertIn('SUPERSEDED',self.od['recovery_robustness'])
 def test_external_refs_flagged(self):
  refs=self.c['references']
  for k in ('E-01','E-02'):
   self.assertIn('not verified',refs[k]['verification']);self.assertIn('인증 기준',refs[k]['caution'])
 def test_objectives_use_new_targets(self):
  for f in ('S1_operating_condition_recovery_boundary.yaml','S2_fault_timing_detection_delay_boundary.yaml','S3_natural_mpc_degradation.yaml','S4_stress_plus_fault.yaml','R_regression_verification.yaml'):
   text=(REPO/'pipeline/02_hypotheses/search_objectives'/f).read_text(encoding='utf-8')
   self.assertNotIn(' recovery_robustness',text.replace('lane_recovery_robustness','')) ;self.assertIn('lane_recovery_robustness',text)
 def test_runner_outcomes_are_declared(self):
  classes=set(self.od['recovery_outcome']['classes']);kw=dict(e_y=[0.0]*40,e_psi=[0.0]*40,collision=False)
  seen={self.m.classify_recovery(switched=s,window_complete=w,margins=m,**kw) for s,w,m in
        [(True,True,[0.5]*40),(True,True,[-0.1]*40),(False,True,[0.5]*40),(True,False,[0.5]*40),(True,True,[-0.1]*5+[0.5]*35)]}
  self.assertTrue(seen<=classes,seen)
if __name__=='__main__':unittest.main()
