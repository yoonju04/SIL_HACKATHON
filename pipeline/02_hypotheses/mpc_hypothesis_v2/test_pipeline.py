import unittest,copy,yaml,pathlib
from generate_hypotheses import safe_expr,apply,generate,interpolate,validate_llm
ROOT=pathlib.Path(__file__).parent
class Tests(unittest.TestCase):
 def setUp(self):
  self.p=yaml.safe_load((ROOT/'safety_profile.example.yaml').read_text())
  self.c=yaml.safe_load((ROOT/'vulnerability_patterns.yaml').read_text())
  self.s=yaml.safe_load((ROOT/'scenario_space.example.yaml').read_text())
 def test_safe_parser(self):
  self.assertTrue(safe_expr("15 > 3 and True"))
  with self.assertRaises(ValueError):safe_expr("__import__('os').system('echo bad')")
 def test_generation(self):
  x=generate(self.p,self.c,self.s)
  self.assertEqual(len(x['hypotheses']),11)
  self.assertEqual([h['id'] for h in x['hypotheses']],[f'H-{i:02}' for i in range(1,12)])
  self.assertEqual(next(h for h in x['hypotheses'] if h['id']=='H-08')['status'],'proposed')
  self.assertEqual(next(h for h in x['hypotheses'] if h['id']=='H-11')['status'],'blocked')
 def test_missing_is_blocked(self):
  self.p['controllers']['lqr']['feedback_design']['design_speed_mps']=None
  x=generate(self.p,self.c,self.s)
  self.assertEqual(x['hypotheses'][0]['status'],'blocked')
 def test_false_not_generated(self):
  self.p['controllers']['lqr']['feedback_design']['gain_type']='scheduled'
  x=generate(self.p,self.c,self.s)
  self.assertFalse(any(h['id']=='H-01' for h in x['hypotheses']))
 def test_no_malicious_proposals(self):
  x=generate(self.p,self.c,self.s,[{'based_on':['O-001'],'condition':{},'control':{},'prediction':{'metric':'recovery_success','direction':'negative'}}])
  self.assertEqual(len(x['rejected_proposals']),1)
 def test_template_preserves_types(self):
  self.assertEqual(interpolate('{controllers.lqr.feedback_design.design_speed_mps}',self.p),15.0)
if __name__=='__main__':unittest.main()
