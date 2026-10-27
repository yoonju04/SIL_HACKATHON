import unittest
from engine.prior_tools.prior_tools import make_nemotron_payload, screen_lateral_demand


class PriorToolsTests(unittest.TestCase):
    def test_only_verified_observations(self):
        profile = {'code_evidence': {'observations': [
            {'id': 'O-1', 'verified': True}, {'id': 'O-2', 'verified': False}]}}
        result = make_nemotron_payload(profile, {}, {'principles': []}, {})
        self.assertEqual(result['unverified_observations_omitted'], 1)
        self.assertIn('O-1', result['messages'][1]['content'])
        self.assertNotIn('O-2', result['messages'][1]['content'])

    def test_stress_not_rejected(self):
        result = screen_lateral_demand({'speed_mps': 20, 'road_curvature_per_m': 0.05}, 8)
        self.assertEqual(result['nominal_lateral_acc_mps2'], 20)
        self.assertEqual(result['tag'], 'stress_test')

    def test_missing_is_not_invented(self):
        self.assertEqual(screen_lateral_demand({'speed_mps': 20})['tag'], 'insufficient_data')

    def test_bad_inputs(self):
        self.assertEqual(screen_lateral_demand({'speed_mps': float('nan'), 'road_curvature_per_m': 0.01})['tag'], 'invalid_input')

if __name__ == '__main__':
    unittest.main()
