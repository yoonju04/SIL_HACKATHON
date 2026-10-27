import unittest,yaml,pathlib
from pipeline.hypothesis_tools.generate_hypotheses import safe_expr,generate,interpolate
from pipeline.hypothesis_tools.build_scenario_candidates import make
from pipeline.hypothesis_tools.space import normalize_space
ROOT=pathlib.Path(__file__).resolve().parents[1]/'pipeline'
def load(rel):return yaml.safe_load((ROOT/rel).read_text(encoding='utf-8'))
class Tests(unittest.TestCase):
 def setUp(self):
  self.p=load('hypothesis_tools/examples/safety_profile.example.yaml')
  self.c=load('02_hypotheses/vulnerability_patterns.yaml')
  self.s=load('04_feasibility_expansion/scenario_space.yaml')
 def test_safe_parser(self):
  self.assertTrue(safe_expr("15 > 3 and True"))
  with self.assertRaises(ValueError):safe_expr("__import__('os').system('echo bad')")
 def test_generation(self):
  x=generate(self.p,self.c,self.s)
  self.assertEqual([h['id'] for h in x['hypotheses']],[f'H-{i:02}' for i in range(2,12)])
  status={h['id']:h['status'] for h in x['hypotheses']}
  self.assertEqual(sorted(k for k,v in status.items() if v=='proposed'),['H-02','H-03','H-08'])
 def test_repo_space_has_every_dimension(self):
  x=generate(self.p,self.c,self.s)
  self.assertFalse([r for h in x['hypotheses'] for r in h.get('blocked_reason',[]) if 'scenario dimensions' in r])
 def test_repo_profile_blocks_everything(self):
  x=generate(load('01_safety_profile/safety_profile.yaml'),self.c,self.s)
  self.assertTrue(all(h['status']=='blocked' for h in x['hypotheses']))
 def test_missing_is_blocked(self):
  # H-04 needs design_speed_mps (formerly checked via H-01, now a search objective)
  self.p['controllers']['lqr']['feedback_design']['design_speed_mps']=None
  x=generate(self.p,self.c,self.s)
  h=next(h for h in x['hypotheses'] if h['id']=='H-04')
  self.assertEqual(h['status'],'blocked')
  self.assertTrue(any('design_speed_mps' in r for r in h['blocked_reason']))
 def test_false_not_generated(self):
  cat={**self.c,'patterns':[{**self.c['superseded_patterns'][0],'status':None}]}
  self.p['controllers']['lqr']['feedback_design']['gain_type']='scheduled'
  x=generate(self.p,cat,self.s)
  self.assertEqual(x['hypotheses'],[])
  self.assertFalse(x['pattern_application'][0]['applies'])
 def test_no_malicious_proposals(self):
  x=generate(self.p,self.c,self.s,[{'based_on':['O-001'],'condition':{},'control':{},'prediction':{'metric':'recovery_success','direction':'negative'}}])
  self.assertEqual(len(x['rejected_proposals']),1)
 def test_template_preserves_types(self):
  self.assertEqual(interpolate('{controllers.lqr.feedback_design.design_speed_mps}',self.p),15.0)
 def test_h01_superseded_by_search_objective(self):
  self.assertFalse(any(p['target_id']=='H-01' for p in self.c['patterns']))
  old=self.c['superseded_patterns'][0]
  self.assertEqual((old['target_id'],old['status']),('H-01','superseded'))
  self.assertFalse(any(h['id']=='H-01' for h in generate(self.p,self.c,self.s)['hypotheses']))
  so=load('02_hypotheses/'+old['superseded_by'])
  self.assertEqual(so['id'],'H-01')
  self.assertEqual(so['search_dimensions'],['speed_mps','road_curvature_per_m'])
  self.assertEqual([d['name'] for d in so['derived_features']],['speed_deviation_mps','curvature_deviation_per_m'])
  self.assertEqual([e['name'] for e in so['evaluation']],['recovery_success','recovery_robustness','margin_delta_m'])
  self.assertEqual(so['search_strategy']['arms'],'none')
  dims=normalize_space(self.s)['dimensions']
  self.assertTrue(all(d in dims for d in so['search_dimensions']))
  for e in so['evaluation']:
   self.assertIn(e['name'],self.c['outcome_definitions'])
 def test_every_metric_has_gp_target(self):
  for name,rule in self.c['metric_rules'].items():
   self.assertIn(rule['gp']['form'],('paired_delta_if_fault_else_absolute','absolute_post_switch'),name)
   self.assertGreater(rule['gp']['scale'],0,name)
 def test_stress_tagged_not_rejected(self):
  h=next(h for h in generate(self.p,self.c,self.s)['hypotheses'] if h['id']=='H-08')
  rows=make(h,normalize_space(self.s),4,7,max_lateral_acc=1.0)
  self.assertTrue(all(r['lateral_screen']=='stress_test' and r['executable'] for r in rows))
 def test_inactive_fault_params_dropped(self):
  h=next(h for h in generate(self.p,self.c,self.s)['hypotheses'] if h['id']=='H-03')
  for r in make(h,normalize_space(self.s),4,7):
   faults={k.split('.')[0] for k in r['parameters'] if '.' in k}
   self.assertEqual(faults,{r['parameters']['fault_type']})
if __name__=='__main__':unittest.main()
