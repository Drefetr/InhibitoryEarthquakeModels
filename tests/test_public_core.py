"""Fixed inputs only: no simulation, optimizer or research-history work."""
from pathlib import Path
import sys,math,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'engine'))
from api import load,MODULES
class CoreTests(unittest.TestCase):
 def test_import_graph(self):
  for name in MODULES:
   with self.subTest(name=name):self.assertIsNotNone(load(name))
 def test_kernel_normalisation(self):
  w,r=load('kernels').KernelSpec('exp',2.).mixture
  np.testing.assert_allclose(w,[1.],rtol=0,atol=0);np.testing.assert_allclose(r,[.5],rtol=0,atol=0)
 def test_poisson_exact_exposure(self):
  a=load('observed');record={'pars':[math.log(.2),0.],'trigger':None,'stress':None,'ratio':0.}
  out=a.evaluate_window(np.array([1.,3.]),np.array([.2,.4]),record,0.,10.,10.,2.,32)
  self.assertAlmostEqual(out['integrated_intensity'],2.,places=11)
  self.assertAlmostEqual(out['ground_loglik'],2*math.log(.2)-2,places=11)
  self.assertEqual(out['n_events'],2);self.assertTrue(out['final_tail_right_censored'])
if __name__=='__main__':unittest.main()
