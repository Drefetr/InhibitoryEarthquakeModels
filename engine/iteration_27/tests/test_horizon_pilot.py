from pathlib import Path
import sys,unittest,math
from decimal import Decimal
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'iteration_27/code'))
from horizon_pilot import transformed_background,prefix

class HorizonFixtures(unittest.TestCase):
    def test_physical_background_identity(self):
        a,d,T=-4.162108473051707,.23211265877958467,24394.
        for mult in [1,2,4]:
            H=mult*T;aa,dd=transformed_background(a,d,H,T)
            for t in np.linspace(0,H,41):self.assertAlmostEqual(aa+dd*(2*t/H-1),a+d*(2*t/T-1),places=14)
    def test_exact_prefix_exclusive_and_subulp(self):
        arrays=dict(times=np.array([1.,2.,2.,3.]),times_lo=np.array([0.,-1e-30,1e-30,0.]),event_gaps=np.array([1.,1.-1e-30,2e-30,1.]),excess=np.array([0.,.1,.2,.3]),exact_time_strings=np.array(['1','1.999999999999999999999999999999','2.000000000000000000000000000001','3']))
        q=prefix(arrays,2.)
        self.assertEqual(len(q['times']),2);self.assertEqual(q['times_lo'][-1],-1e-30)
        self.assertTrue(all(len(v)==2 for v in q.values()));self.assertEqual(len(prefix(arrays,1.)['times']),0)
    def test_prefix_no_mutation(self):
        a={k:np.array([1,2,3]) for k in ['times','excess']};a['exact_time_strings']=np.array(['1','2','3']);q=prefix(a,3.);q['times'][0]=100
        self.assertEqual(a['times'][0],1)

class LikelihoodFixture(unittest.TestCase):
    def test_independent_literal_event_compensator_and_gradient(self):
        import math,time
        from scipy.integrate import quad
        sys.path.insert(0,str(ROOT/'iteration_20/code'))
        from restricted_history import RestrictedHistory,KnownLaw,KernelSpec
        times=np.array([.1,.4,1.3]);q=np.array([.1,.5,.2]);T=2.;T0=10.;tau=.7;beta=3.;ratio=.4
        arrays=dict(times=times,times_lo=np.zeros(3),event_gaps=np.diff(np.r_[0.,times]),excess=q)
        kernel=KernelSpec('exp',tau);law=KnownLaw(-2.,.3,.2,beta,ratio,T,kernel,kernel,T0)
        pars=np.array([-2.,.3,.2,.4]);g=np.exp(.75*math.log(10)*q)/math.exp(.75*math.log(10));k=(1-ratio)*np.exp(beta*ratio*q)
        def lam(t):
            G=sum(gg*math.exp(-(t-ti)/tau) for ti,gg in zip(times,g) if ti<t)
            H=sum(kk/tau*math.exp(-(t-ti)/tau) for ti,kk in zip(times,k) if ti<t)
            return math.exp(pars[0]+pars[1]*(2*t/T0-1)-pars[3]*G)+pars[2]*H
        literal=-sum(math.log(lam(t)) for t in times)+sum(quad(lam,a,b,epsabs=1e-11,epsrel=1e-11)[0] for a,b in zip([0.]+times.tolist(),times.tolist()+[T]))
        h=RestrictedHistory(arrays,law,clock_mode='paired_clock',order=32,backend='cpu',deadline=time.perf_counter()+30)
        try:
            value,grad=h.objective(pars,kernel,ratio,True)
            self.assertAlmostEqual(value,literal,places=10)
            for i in range(4):
                p=pars.copy();m=pars.copy();p[i]+=1e-6;m[i]-=1e-6
                fd=(h.objective(p,kernel,ratio,False)-h.objective(m,kernel,ratio,False))/2e-6
                self.assertAlmostEqual(grad[i],fd,places=6)
        finally:h.close()

class ActualSmallFitFixture(unittest.TestCase):
    def test_complete_small_cuda_CPU32_pair(self):
        import time
        from decimal import Decimal
        sys.path.insert(0,str(ROOT/'iteration_20/code'))
        from restricted_history import KnownLaw,KernelSpec
        from horizon_pilot import fit_pair
        sys.path.insert(0,str(ROOT/'iteration_18/code'))
        from cuda_oracle import CudaOracleContext
        rng=np.random.default_rng(81027);times=np.sort(rng.uniform(.1,39.9,30));q=rng.exponential(.1,30)
        exact=[Decimal.from_float(float(t)) for t in times]
        gaps=np.array([float(t-(exact[i-1] if i else Decimal(0))) for i,t in enumerate(exact)])
        arrays=dict(times=times,times_lo=np.zeros(30),event_gaps=gaps,excess=q,exact_time_strings=np.array([str(t) for t in exact]))
        k=KernelSpec('exp',.7);law=KnownLaw(-2.,.1,.3,3.,.4,40.,k,k,40.)
        context=CudaOracleContext(max_allocated_bytes=2**30)
        try:
            context.warmup();r=fit_pair(arrays,law,dict(validation_reserve_seconds=5,maxls_tiers=[40,200],maxiter=200),context,time.perf_counter()+30)
            self.assertTrue(r['complete'],str(r));self.assertEqual(len(r['records']),4)
            self.assertLess(max(r['deltas']),1e-4)
        finally:context.close()

if __name__=='__main__':unittest.main(verbosity=2)
