"""Exact value likelihood and explicit failure-only derivative-free amendment.

The n=0 one-sided derivative can genuinely exceed FP64. It is never clipped.
The extra optimizer tier uses bounded Powell on the same value likelihood and
also optimizes its exact n=0 face. Prior files and primary successful fits are
untouched. This is an algorithm amendment, not the old L-BFGS-B recipe.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import time
import types
import numpy as np
import numexpr as ne
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'iteration_5/code'))
from empirical_process import EmpiricalHistory
from kernels import KernelSpec
ne.set_num_threads(1)
_spec=importlib.util.spec_from_file_location('frozen_numerical9_stable12',ROOT/'iteration_9/code/numerical_fit.py')
f9=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(f9)

POWELL_OPTIONS=dict(maxiter=100,maxfev=2000,ftol=1e-10,xtol=1e-8)
FACE_OPTIONS=dict(maxiter=100,maxls=200,ftol=1e-10,gtol=2e-6)
EXPECTED_SOURCE='9d8bc923cc87eba586bd0dee721245be73f2bbf623c90e00e138a18198d45ea0'


class UnrepresentableDerivative(FloatingPointError):pass
class ProbeBudgetExceeded(RuntimeError):pass


class StableObjective:
    def __init__(self,history,trigger,ratio,stress=None):
        self.history=history;self.stress=stress
        he,self.integral=history.trigger_features(trigger,ratio)
        self.loghe=np.full_like(he,-np.inf);np.log(he,out=self.loghe,where=he>0)
        self.ge,self.gq=history.stress_features(stress) if stress else (0.,0.)

    def __call__(self,pars,gradient=False):
        h=self.history;a,d,n=np.asarray(pars,dtype=float)[:3];eta=float(pars[3]) if self.stress else 0.
        if n<0:raise ValueError('n must be nonnegative')
        logbe=a+d*h.xe-eta*self.ge
        loglam=np.logaddexp(logbe,self.loghe+np.log(n) if n>0 else np.full_like(self.loghe,-np.inf))
        w=h.wq;x=h.xq;g=self.gq
        wb=ne.evaluate('w*exp(a+d*x-eta*g)',local_dict=dict(w=w,x=x,g=g,a=a,d=d,eta=eta),optimization='moderate')
        integral=float(wb.sum());value=float(-loglam.sum()+integral+n*self.integral)
        if not np.isfinite(value):raise FloatingPointError('nonfinite unchanged likelihood')
        if not gradient:return value
        eb=np.exp(logbe-loglam)
        common=[float(-eb.sum()+integral),float(-eb@h.xe+wb@h.xq)]
        stress_grad=float(eb@self.ge-wb@self.gq) if self.stress else None
        if gradient=='face':
            result=np.asarray(common+([stress_grad] if self.stress else []))
        else:
            log_total=float(logsumexp(self.loghe-loglam))
            if log_total>np.log(np.finfo(float).max):
                raise UnrepresentableDerivative('Exact sum h/lambda exceeds FP64; value remains finite')
            with np.errstate(over='ignore'):
                derivative_n=float(self.integral-np.exp(log_total))
            result=np.asarray(common+[derivative_n]+([stress_grad] if self.stress else []))
        if not np.isfinite(result).all():raise UnrepresentableDerivative('Exact requested gradient exceeds FP64')
        return value,result


class StableHistory(EmpiricalHistory):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.stable_objectives={}
    def cached_objective(self,trigger,ratio,stress=None):
        key=(trigger.key,float(ratio),stress.key if stress else None)
        if key not in self.stable_objectives:self.stable_objectives[key]=StableObjective(self,trigger,ratio,stress)
        return self.stable_objectives[key]
    def objective(self,pars,trigger,ratio,stress=None,gradient=False):
        return self.cached_objective(trigger,ratio,stress)(pars,gradient)


def _snapshot(fit,elapsed):
    return f9._snapshot(fit,'amended_solver',{},elapsed)


def recover_failed_record(history,record,seconds=90.,deadline=None):
    """Return (new canonical record, full lineage); never mutate the old record.

    This tier is eligible only for a retained unsuccessful original start. The
    caller must preserve prior artifacts and run independent CPU32 selected gates.
    """
    if record.get('origin')!='optimizer' or record.get('success'):raise ValueError('Only unsuccessful optimizer records are eligible')
    if hashlib.sha256((ROOT/'iteration_5/code/empirical_process.py').read_bytes()).hexdigest()!=EXPECTED_SOURCE:
        raise RuntimeError('Original likelihood source differs')
    began=time.perf_counter();deadline=min(deadline or float('inf'),began+seconds)
    trigger=KernelSpec(**record['trigger']);stress=KernelSpec(**record['stress']) if record.get('stress') else None
    start=np.asarray(record['start'],dtype=float);bounds=np.asarray([[-12.,0.],[-2.,2.],[0.,.98]]+([[0.,3.]] if stress else []))
    if start.shape!=(len(bounds),) or not np.isfinite(start).all() or np.any(start<bounds[:,0]) or np.any(start>bounds[:,1]):raise ValueError('Original start not feasible')
    objective=StableObjective(history,trigger,record['ratio'],stress)
    n0_evaluations=0
    def value(pars):
        nonlocal n0_evaluations
        if time.perf_counter()>=deadline:raise ProbeBudgetExceeded('Declared objective wall cap reached')
        if pars[2]==0:n0_evaluations+=1
        return objective(pars)
    updated=copy.deepcopy(record)
    lineage=dict(algorithm='failure-only bounded Powell plus exact n=0 face',original_record=copy.deepcopy(record),
        original_start=start.tolist(),bounds=bounds.tolist(),trigger=asdict(trigger),stress=asdict(stress) if stress else None,
        powell_options=dict(POWELL_OPTIONS),face_options=dict(FACE_OPTIONS),accepted=False,requires_independent_CPU32_gate=True)
    lineage['implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        initial=value(start);lineage['start_nll']=initial
        prior_reference=min(initial,float(record['nll'])) if record.get('nll') is not None and np.isfinite(record['nll']) else initial
        lineage['nonworsening_reference_nll']=prior_reference
        n0=start.copy();n0[2]=0.;lineage['exact_n0_original_start_nll']=value(n0)
        tick=time.perf_counter()
        full=minimize(value,start.copy(),method='Powell',bounds=bounds,options=dict(POWELL_OPTIONS))
        lineage['full_box']=_snapshot(full,time.perf_counter()-tick)
        free=[0,1]+([3] if stress else [])
        def face(pars):
            nonlocal n0_evaluations
            if time.perf_counter()>=deadline:raise ProbeBudgetExceeded('Declared face wall cap reached')
            n0_evaluations+=1
            point=n0.copy();point[free]=pars
            return objective(point,'face')
        tick=time.perf_counter()
        boundary=minimize(face,start[free].copy(),jac=True,method='L-BFGS-B',bounds=bounds[free],options=dict(FACE_OPTIONS))
        boundary_record=_snapshot(boundary,time.perf_counter()-tick)
        lineage['exact_n0_face']=boundary_record
        full_valid=f9._valid_result(full,bounds);face_valid=f9._valid_result(boundary,bounds[free])
        lineage['returned_candidates_valid']=dict(full_box=full_valid,exact_n0_face=face_valid)
        if full.success and boundary.success and full_valid and face_valid:
            face_point=n0.copy();face_point[free]=boundary.x
            boundary_record['full_pars']=face_point.tolist()
            full_value=value(full.x);face_value=face(boundary.x)[0]
            full_agreement=abs(float(full.fun)-full_value)<1e-4
            face_agreement=abs(float(boundary.fun)-face_value)<1e-4
            chosen=full if full_value<=face_value else boundary
            pars=full.x.copy() if chosen is full else face_point
            nll=value(pars)
            original_candidate_value=EmpiricalHistory.objective(history,pars,trigger,record['ratio'],stress,False)
            feasible=bool(np.isfinite(pars).all() and np.all(pars>=bounds[:,0]) and np.all(pars<=bounds[:,1]))
            agreement=bool(np.isfinite(original_candidate_value) and abs(original_candidate_value-nll)<1e-4 and full_agreement and face_agreement)
            nonworsening=bool(max(nll,float(chosen.fun),float(original_candidate_value))<=prior_reference+1e-8)
            full_nonworsening=bool(max(full_value,float(full.fun))<=initial+1e-8)
            face_nonworsening=bool(max(face_value,float(boundary.fun))<=lineage['exact_n0_original_start_nll']+1e-8)
            lineage['acceptance']=dict(feasible=feasible,reference_original_nll=float(original_candidate_value),nll=float(nll),
                original_minus_new=float(original_candidate_value-nll),agreement_strict_1e_4=agreement,nonworsening_tolerance_1e_8=nonworsening,
                full_returned_fun_agreement=full_agreement,face_returned_fun_agreement=face_agreement,
                full_solver_nonworsening=full_nonworsening,face_solver_own_start_nonworsening=face_nonworsening,
                chosen='full_box' if chosen is full else 'exact_n0_face')
            try:
                _,gradient=objective(pars,True);lineage['final_gradient']=gradient.tolist()
            except UnrepresentableDerivative as exc:lineage['final_gradient_unrepresentable']=str(exc)
            if feasible and agreement and nonworsening and full_nonworsening and face_nonworsening:
                updated.update(pars=pars.tolist(),nll=float(nll),success=True,status=0,
                    message='AMENDED: full-box Powell and exact-n0-face solvers converged; independent CPU32 gate still required',
                    nit=int(full.nit+boundary.nit),nfev=int(full.nfev+boundary.nfev),
                    elapsed_seconds=time.perf_counter()-began,numerical_method='bounded_Powell_plus_exact_n0_face')
                lineage['accepted']=True
    except (ProbeBudgetExceeded,FloatingPointError,ValueError,OverflowError) as exc:
        lineage['error']=f'{type(exc).__name__}: {exc}'
    lineage.update(elapsed_seconds=time.perf_counter()-began,exact_n0_value_evaluations=n0_evaluations)
    return updated,lineage


def fit_with_recovery(history,maxiter=100,ratios=(.4,.7,.8),deadline=None,include_omori=True,history_factory=None):
    """F9 branches first; only total F9 failure invokes the new CPU value tier.

    Optional history_factory supplies the frozen9 CPU retry history, including
    callers whose event clock has a paired representation. No primary branch is
    switched to StableHistory. The cloned original loop retains 234/162 records.
    """
    if maxiter!=100:raise ValueError('Original maxiter100 required')
    if hashlib.sha256((ROOT/'iteration_5/code/empirical_process.py').read_bytes()).hexdigest()!=EXPECTED_SOURCE:
        raise RuntimeError('Frozen original source differs')
    controller=f9.RecoveryController(history,history_factory=history_factory);extra=[]
    def controlled(fun,x0,*args,**kwargs):
        old=None;old_error=None
        try:
            old=controller(fun,x0,*args,**kwargs)
            if bool(old.success):return old
        except (FloatingPointError,ValueError,OverflowError) as exc:old_error=exc
        context=dict(zip(fun.__code__.co_freevars,(cell.cell_contents for cell in (fun.__closure__ or ()))))
        if set(context)!={'history','kernel','ratio','stress'}:raise RuntimeError('Original objective closure differs')
        old_lineage=controller.lineages[-1]
        finite=[r.get('nll') for r in old_lineage['attempts'] if r.get('nll') is not None and np.isfinite(r['nll'])]
        failed=dict(origin='optimizer',success=False,start=np.asarray(x0).tolist(),trigger=asdict(context['kernel']),
            stress=asdict(context['stress']) if context['stress'] else None,ratio=float(context['ratio']),
            nll=min(finite) if finite else None,message=str(old_error) if old_error else str(old.message))
        repaired,lineage=recover_failed_record(controller.cpu_history(),failed,deadline=deadline)
        lineage['optimizer_call_index']=controller.calls-1;extra.append(lineage)
        if lineage['accepted']:
            return OptimizeResult(x=np.asarray(repaired['pars']),fun=repaired['nll'],success=True,status=0,
                message=repaired['message'],nit=repaired['nit'],nfev=repaired['nfev'])
        if old_error is not None:raise old_error
        return old
    original=f9.ep.fit_candidates
    cloned=types.FunctionType(original.__code__,dict(original.__globals__,minimize=controlled),
        name=original.__name__,argdefs=original.__defaults__,closure=original.__closure__)
    cloned.__kwdefaults__=original.__kwdefaults__
    fitted=cloned(history,maxiter=maxiter,ratios=ratios,deadline=deadline,include_omori=include_omori)
    records=[(i,r) for i,r in enumerate(fitted['subfits']) if r['origin']=='optimizer']
    if len(records)!=controller.calls:raise RuntimeError('Original optimizer count differs')
    for i,line in enumerate(controller.lineages):
        index,record=records[line['optimizer_call_index']];line['canonical_subfit_index']=index
        record['numerical_recovery_lineage_index']=i
    for i,line in enumerate(extra):
        index,record=records[line['optimizer_call_index']];line['canonical_subfit_index']=index
        record['derivative_free_lineage_index']=i
    fitted['numerical_recovery']=dict(policy=f9.POLICY,primary_optimizer_calls=controller.calls,lineages=controller.lineages,
        failed_primary_starts=len(controller.lineages),accepted_recoveries=sum(r['accepted_tier'] is not None for r in controller.lineages),
        agreement_tolerance_strict=f9.AGREEMENT_TOLERANCE,nonworsening_tolerance=f9.NONWORSENING_TOLERANCE,
        original_source_sha256=f9.EXPECTED_EMPIRICAL_SHA256)
    fitted['derivative_free_recovery']=dict(policy='Only after complete F9 failure; bounded Powell plus exact n0 face',
        lineages=extra,accepted=sum(r['accepted'] for r in extra),powell_options=dict(POWELL_OPTIONS),face_options=dict(FACE_OPTIONS))
    return fitted


def probe(seconds=90.):
    began=time.perf_counter();deadline=began+seconds
    path=ROOT/'iteration_11/results/full_bootstrap/runs/null_0066'
    before={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ('catalogue.npz','fit.json','simulation.json')}
    saved=json.loads((path/'fit.json').read_text());old=saved['fit']['subfits'][122]
    with np.load(path/'catalogue.npz',allow_pickle=False) as data:times=data['times'];excess=data['excess']
    beta=float(1/excess.mean());history=StableHistory(times,excess,beta,24394.,24394.,order=16,memory_limit_mb=1600.)
    repaired,lineage=recover_failed_record(history,old,seconds=seconds,deadline=deadline)
    result=dict(case='null_0066',index=122,n_events=len(times),record=repaired,lineage=lineage,input_hashes=before)
    if repaired['success'] and time.perf_counter()<deadline:
        trigger=KernelSpec(**old['trigger']);stress=KernelSpec(**old['stress'])
        fine=EmpiricalHistory(times,excess,beta,24394.,24394.,order=32,memory_limit_mb=3200.)
        fine_nll=fine.objective(repaired['pars'],trigger,old['ratio'],stress,False)
        result['original_CPU32']=dict(nll=fine_nll,delta=fine_nll-repaired['nll'],strict_1e_4_pass=abs(fine_nll-repaired['nll'])<1e-4)
    result['elapsed_seconds']=time.perf_counter()-began
    result['inputs_unchanged']=before=={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in before}
    out=ROOT/'iteration_12/results/numerical_probe';out.mkdir(parents=True,exist_ok=True)
    target=out/'null_0066_probe.json'
    if target.exists():raise FileExistsError('Preserve the first bounded probe; choose an explicit new probe version')
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(target),accepted=lineage['accepted'],elapsed_seconds=result['elapsed_seconds'],inputs_unchanged=result['inputs_unchanged'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');parser.add_argument('--seconds',type=float,default=90.)
    args=parser.parse_args()
    if args.probe:probe(args.seconds)
