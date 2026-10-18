"""Synthetic and literal checks only; never load the observed catalogue."""
from __future__ import annotations
import copy
import importlib.util
import math
from pathlib import Path
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.integrate import quad
from scipy.optimize._numdiff import approx_derivative

CODE=Path(__file__).resolve().parents[1]/'code'
sys.path.insert(0,str(CODE))
import adequacy as a
import run_adequacy as runner


def record(stress=False):
    k=a.KernelSpec('exp',2.)
    return {'pars':[math.log(.3),.2,.25]+([.5] if stress else []),
            'trigger':a.asdict(k),'stress':a.asdict(k) if stress else None,'ratio':.4}


class LiteralChecks(unittest.TestCase):
    def test_analytic_poisson_and_zero_slope(self):
        r={'pars':[math.log(.2),0.],'trigger':None,'stress':None,'ratio':0.}
        out=a.evaluate_window([1.,3.],[.2,.4],r,0.,5.,5.,2.)
        self.assertAlmostEqual(out['ground_loglik'],2*math.log(.2)-1.,places=13)
        self.assertAlmostEqual(out['common_mark_loglik'],2*math.log(2.)-1.2,places=13)
        self.assertEqual(len(out['rescaled_uniform']),2)
        self.assertAlmostEqual(out['final_no_event_compensator'],.4,places=13)
        for d in [0.,1e-10,-1.3,2.]:
            expected=quad(lambda t:math.exp(-2.+d*(2*t/5.-1)),.2,4.8,epsabs=1e-12)[0]
            self.assertAlmostEqual(a.background_integral(-2.,d,.2,4.8,5.)[0],expected,places=12)

    def test_literal_pretest_history_full_compensator_and_tail(self):
        times=np.array([1.,3.]);marks=np.array([.2,.4]);r=record(True)
        beta=2.;ratio=.4;eta=.5;n=.25;A=.3;d=.2;T=5.
        def intensity(t):
            past=times<t;lag=t-times[past]
            G=sum(np.exp(a.GAMMA*(marks[past]-1.))*np.exp(-lag/2.))
            H=sum((1-ratio)*np.exp(beta*ratio*marks[past])*.5*np.exp(-lag/2.))
            return A*math.exp(d*(2*t/T-1)-eta*G)+n*H
        expected_comp=quad(intensity,2.,3.,epsabs=1e-12)[0]+quad(intensity,3.,5.,epsabs=1e-12)[0]
        expected=math.log(intensity(3.))-expected_comp
        out=a.evaluate_window(times,marks,r,2.,5.,T,beta,32)
        self.assertAlmostEqual(out['ground_loglik'],expected,places=11)
        self.assertAlmostEqual(out['integrated_intensity'],expected_comp,places=11)
        self.assertAlmostEqual(out['event_compensator_increments'][0],quad(intensity,2.,3.,epsabs=1e-12)[0],places=11)
        self.assertAlmostEqual(out['final_no_event_compensator'],quad(intensity,3.,5.,epsabs=1e-12)[0],places=11)
        # Dropping the pre-test event changes intensity and is detectably wrong.
        dropped=a.evaluate_window(times[1:],marks[1:],r,2.,5.,T,beta,32)
        self.assertGreater(abs(dropped['ground_loglik']-expected),.001)

    def test_split_additivity_and_right_censoring(self):
        times=[1.,3.];marks=[.2,.4];r=record(True)
        full=a.evaluate_window(times,marks,r,0.,5.,5.,2.,32)
        left=a.evaluate_window(times,marks,r,0.,2.,5.,2.,32)
        right=a.evaluate_window(times,marks,r,2.,5.,5.,2.,32)
        for key in ('ground_loglik','common_mark_loglik','integrated_intensity'):
            self.assertAlmostEqual(full[key],left[key]+right[key],places=11)
        self.assertEqual(len(full['event_compensator_increments']),2)
        self.assertEqual(len(full['rescaled_uniform']),2)
        self.assertGreater(full['final_no_event_compensator'],0.)

    def test_event_at_window_start_and_future_marks(self):
        r=record(True)
        out=a.evaluate_window([1.,3.],[.2,.4],r,1.,5.,5.,2.,32)
        self.assertEqual(out['event_compensator_increments'][0],0.)
        h=a.PredictiveHistory([1.,3.],[.2,999.],2.,0.,3.,5.,32)
        self.assertEqual(len(h.times),1)
        self.assertEqual(h.stress(a.KernelSpec('exp',2.))[0][0],0.)

    def test_continuous_omori_value_gradient_independent_quad(self):
        times=np.array([1.,3.]);marks=np.array([.2,.4]);beta=2.;ratio=.4
        h=a.PredictiveHistory(times,marks,beta,2.,5.,5.,32)
        q=np.array([-1.3,.4,.35,math.log(.3),1.7])
        def lam(t):
            lag=t-times[times<t];kp=(1-ratio)*np.exp(beta*ratio*marks[times<t])
            c=math.exp(q[3]);return math.exp(q[0]+q[1]*(2*t/5.-1))+q[2]*sum(kp*(q[4]-1)/c*(1+lag/c)**(-q[4]))
        expected=quad(lam,2.,3.,epsabs=1e-12)[0]+quad(lam,3.,5.,epsabs=1e-12)[0]-math.log(lam(3.))
        value,gradient=a.omori_continuous_objective(h,q,ratio)
        self.assertAlmostEqual(value,expected,places=11)
        finite=approx_derivative(lambda x:a.omori_continuous_objective(h,x,ratio,False),q).ravel()
        np.testing.assert_allclose(gradient,finite,rtol=3e-7,atol=1e-8)
        fixed=a.KernelSpec('omori',math.exp(q[3]),p=q[4])
        self.assertAlmostEqual(value,h.objective(q[:3],fixed,ratio),places=12)


class SyntheticChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(2026101827)
        cls.times=np.sort(rng.uniform(.1,99.,40));cls.excess=rng.exponential(.3,40)
        cls.search={'ratios':[.4],'ml_alpha':[.5],'ml_tau':[1.],
                    'omori_p':[1.3],'omori_tau':[1.],'continuous_omori':True,
                    'optimizer':{'maxiter':400,'maxfun':20000,'ftol':1e-10,'gtol':2e-6,'maxls':40}}
        cls.fit=a.fit_training(cls.times,cls.excess,80.,100.,cls.search,time.perf_counter()+60.)

    def test_small_full_search_and_training_mark_isolation(self):
        self.assertTrue(self.fit['complete'],[(r['origin'],r['message']) for r in self.fit['records'] if not r['success']])
        self.assertTrue(a.validate_fit(self.fit,self.search))
        mask=self.times<80.
        self.assertEqual(self.fit['beta_train'],float(mask.sum()/self.excess[mask].sum()))
        changed=self.excess.copy();changed[~mask]=99.
        again=a.fit_training(self.times,changed,80.,100.,self.search,time.perf_counter()+60.)
        self.assertEqual(self.fit['beta_train'],again['beta_train'])
        for name in a.FAMILIES:
            np.testing.assert_array_equal(self.fit['best'][name]['pars'],again['best'][name]['pars'])
            self.assertEqual(self.fit['best'][name]['nll'],again['best'][name]['nll'])
        self.assertLessEqual(self.fit['best']['ml_shared']['nll'],self.fit['best']['ml_trigger']['nll'])

    def test_pinned_old_history_agreement_and_refinement(self):
        path=a.ROOT/'iteration_5/code/empirical_process.py'
        sys.path.insert(0,str(path.parent))
        spec=importlib.util.spec_from_file_location('_old27_oracle',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for name in a.FAMILIES:
            r=self.fit['best'][name]
            controls=a.selected_controls(self.times,self.excess,r,80.,100.,100.,self.fit['beta_train'])
            self.assertTrue(controls['pass'])
            if r['trigger']:
                old=module.EmpiricalHistory(self.times,self.excess,self.fit['beta_train'],100.,100.,start=80.,order=32)
                k=module.KernelSpec(**r['trigger']);s=module.KernelSpec(**r['stress']) if r['stress'] else None
                self.assertAlmostEqual(-old.objective(r['pars'],k,r['ratio'],s),controls['order32']['ground_loglik'],places=10)

    def test_gradient_and_empty_scoring(self):
        h=a.PredictiveHistory(self.times,self.excess,3.,40.,100.,100.,32)
        k=a.KernelSpec('ml',1.,alpha=.5);q=np.array([-2.,.2,.4,.6])
        value,gradient=h.objective(q,k,.4,k,True)
        finite=approx_derivative(lambda x:h.objective(x,k,.4,k),q).ravel()
        np.testing.assert_allclose(gradient,finite,rtol=2e-6,atol=2e-7)
        empty=a.evaluate_window([],[],record(True),0.,5.,5.,2.,32)
        self.assertEqual(empty['event_compensator_increments'],[])
        self.assertGreater(empty['final_no_event_compensator'],0.)

    def test_failure_deadline_and_tamper_policy(self):
        failed=copy.deepcopy(self.fit);failed['records'][1]['success']=False
        with self.assertRaises(ValueError):a.validate_fit(failed,self.search)
        tampered=copy.deepcopy(self.fit);tampered['best']['ml_trigger']=copy.deepcopy(tampered['best']['ml_trigger']);tampered['best']['ml_trigger']['pars'][0]+=1.
        with self.assertRaises(ValueError):a.validate_fit(tampered,self.search)
        shortened=copy.deepcopy(self.fit);shortened['records'].pop();shortened['expected_records']-=1
        with self.assertRaises(ValueError):a.validate_fit(shortened,self.search)
        kernel_tamper=copy.deepcopy(self.fit)
        next(r for r in kernel_tamper['records'] if r['origin']=='continuous_omori_optimizer')['optimizer_pars'][4]=3.1
        with self.assertRaises(ValueError):a.validate_fit(kernel_tamper,self.search)
        expired=a.fit_training(self.times,self.excess,80.,100.,self.search,time.perf_counter()-1.)
        self.assertTrue(expired['censored']);self.assertFalse(expired['complete'])
        with self.assertRaises(ValueError):a.validate_fit(expired,self.search)

    def test_input_memory_and_protocol_guards(self):
        with self.assertRaises(ValueError):a.PredictiveHistory([1.,1.],[.2,.3],2.,0.,3.,5.)
        with self.assertRaises(MemoryError):a.PredictiveHistory(self.times,self.excess,3.,0.,100.,100.,memory_limit_mb=.001)
        with self.assertRaises(ValueError):a.PredictiveHistory([1.],[.2],0.,0.,3.,5.)
        import json
        interface=json.loads((a.ROOT/'iteration_27/configs/D_interface.json').read_text())
        draft={'search':interface['search'],'T0':24394.,'folds':[
            {'id':'fold_50','train_fraction':.5,'test_end_fraction':.65},
            {'id':'fold_65','train_fraction':.65,'test_end_fraction':.8},
            {'id':'fold_80','train_fraction':.8,'test_end_fraction':1.}], 'wall_cap_seconds':1800.}
        with self.assertRaises(ValueError):runner.validate_protocol(draft,interface)
        runner.validate_protocol(draft,interface,benchmark=True)

    def test_uniform_failure_only_recovery_and_failed_return_exclusion(self):
        calls=[]
        def fake(fun,start,**kw):
            calls.append(kw)
            tier=len(calls)-1
            return SimpleNamespace(x=np.array([10. if tier<2 else 1.]),fun=-100. if tier<2 else 0.,
                success=tier==2,message='deliberate synthetic failure' if tier<2 else 'success',
                status=2 if tier<2 else 0,nit=1,nfev=2)
        target=lambda x:((x[0]-1.)**2,np.array([2*(x[0]-1.)]))
        with patch.object(a,'minimize',fake):
            chosen,lineage,initial,censored=a._bounded_solve(target,[0.],[(-12.,12.)],
                {'maxiter':150,'maxfun':20000,'ftol':1e-10,'gtol':2e-6,'maxls':40},
                time.perf_counter()+5.,lambda x,refined:target(x)[0])
        self.assertFalse(censored);self.assertEqual(initial,1.)
        self.assertEqual(chosen.x[0],1.)
        self.assertEqual([r['accepted'] for r in lineage],[False,False,True])
        self.assertEqual([r['method'] for r in lineage],['L-BFGS-B','L-BFGS-B','Powell'])
        self.assertTrue(all(r['start']==[0.] for r in lineage))
        self.assertEqual(lineage[1]['options']['maxls'],200)


if __name__=='__main__':unittest.main(verbosity=2)
