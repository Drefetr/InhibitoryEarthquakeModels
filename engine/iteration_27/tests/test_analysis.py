from pathlib import Path
import sys,unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from analyze_adequacy import residual_summary,validate_observed

class AnalysisTests(unittest.TestCase):
    def test_literal_uniform_and_tail(self):
        import numpy as np
        r=residual_summary(-np.log1p(-np.array([.25,.75])),2.)
        self.assertAlmostEqual(r['uniform_KS_statistic'],.25)
        self.assertAlmostEqual(r['realized_integrated_intensity'],2.-np.log(.75*.25))
        self.assertEqual(r['n_complete_gaps'],2);self.assertEqual(r['acf'][0]['pairs'],1)
    def test_empty(self):
        r=residual_summary([],1.5);self.assertIsNone(r['uniform_KS_statistic']);self.assertEqual(r['count_minus_compensator'],-1.5)
    def test_invalid(self):
        with self.assertRaises(ValueError):residual_summary([-1.],0.)
    def test_incomplete_fold_quota(self):
        with self.assertRaises(ValueError):validate_observed(dict(T0=24394.,folds={}))

if __name__=='__main__':unittest.main(verbosity=2)
