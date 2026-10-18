"""Failure-only numerical recovery inside the unchanged finite-grid fit loop.

The original fit_candidates code object runs with a private globals dictionary;
only its minimize callable is replaced. Successful primary starts are untouched.
An unsuccessful start gets original-CPU FP64 retries from its original start,
first with the original options, then with maxls=200. No original globals/files
are patched. Full retry lineage is separate from the canonical 234 records.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from dataclasses import asdict
import hashlib
import time
import types
import numpy as np

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'iteration_5/code'))
import empirical_process as ep

EXPECTED_EMPIRICAL_SHA256='9d8bc923cc87eba586bd0dee721245be73f2bbf623c90e00e138a18198d45ea0'
AGREEMENT_TOLERANCE=1e-4
NONWORSENING_TOLERANCE=1e-8
POLICY='primary-original-options; unsuccessful -> originalCPU40 -> originalCPU200; original start, bounds, maxiter and tolerances'


def _finite(value):
    try:return bool(np.isfinite(value))
    except (TypeError,ValueError):return False


def _number(value):return float(value) if _finite(value) else None


def _vector(value):return [_number(item) for item in np.asarray(value,dtype=float).ravel()]


def _valid_result(result,bounds):
    pars=np.asarray(result.x,dtype=float);bounds=np.asarray(bounds,dtype=float)
    return bool(pars.shape==(len(bounds),) and np.isfinite(pars).all() and _finite(result.fun) and
        np.all(pars>=bounds[:,0]) and np.all(pars<=bounds[:,1]) and
        (getattr(result,'jac',None) is None or np.isfinite(result.jac).all()))


def _snapshot(result,backend,options,elapsed,error=None):
    value=dict(backend=backend,options=dict(options),elapsed_seconds=elapsed)
    if error is not None:return dict(value,success=False,error=f'{type(error).__name__}: {error}')
    for key in ('success','status','nit','nfev','njev'):
        if key in result:value[key]=bool(result[key]) if key=='success' else int(result[key])
    value.update(pars=_vector(result.x),pars_finite=bool(np.isfinite(result.x).all()),nll=_number(result.fun),
                 nll_finite=_finite(result.fun),message=str(result.message))
    if getattr(result,'jac',None) is not None:
        value.update(gradient=_vector(result.jac),gradient_finite=bool(np.isfinite(result.jac).all()))
    return value


class RecoveryController:
    """Per-fit state; constructor injection supports non-scientific unit tests."""
    def __init__(self,history,minimize_impl=None,history_factory=None):
        self.history=history;self.minimize=minimize_impl or ep.minimize
        self.history_factory=history_factory or ep.EmpiricalHistory
        self.cpu=None;self.lineages=[];self.calls=0

    def cpu_history(self):
        if self.cpu is None:
            h=self.history
            self.cpu=self.history_factory(h.times,h.excess,h.beta,h.end,h.coordinate_horizon,
                start=h.start,order=h.order,memory_limit_mb=h.memory_limit_mb)
        return self.cpu

    def __call__(self,fun,x0,*args,**kwargs):
        number=self.calls;self.calls+=1;start=np.asarray(x0,dtype=float).copy();options=dict(kwargs['options'])
        if options!={'maxiter':100,'ftol':1e-10,'gtol':2e-6,'maxls':40}:
            raise RuntimeError('Recovery requires the frozen original optimizer options')
        began=time.perf_counter();original_error=None;primary=None
        try:
            primary=self.minimize(fun,start.copy(),*args,**kwargs)
            if bool(primary.success) and _valid_result(primary,kwargs['bounds']):return primary
        except (FloatingPointError,ValueError,OverflowError) as exc:original_error=exc
        primary_record=_snapshot(primary,'primary',options,time.perf_counter()-began,original_error)
        if primary is not None:primary_record['numerically_valid_result']=_valid_result(primary,kwargs['bounds'])
        closure=dict(zip(fun.__code__.co_freevars,(cell.cell_contents for cell in (fun.__closure__ or ()))))
        if set(closure)!={'history','kernel','ratio','stress'} or closure['history'] is not self.history:
            raise RuntimeError('Frozen original objective closure differs')
        kernel=closure['kernel'];ratio=closure['ratio'];stress=closure['stress'];bounds=np.asarray(kwargs['bounds'],dtype=float)
        expected=np.asarray([(-12.,0.),(-2.,2.),(0.,.98)]+([(0.,3.)] if stress else []))
        if kwargs.get('method')!='L-BFGS-B' or kwargs.get('jac') is not True or not np.array_equal(bounds,expected):
            raise RuntimeError('Frozen original optimizer method/bounds differ')
        lineage=dict(optimizer_call_index=number,start=start.tolist(),trigger=asdict(kernel),ratio=float(ratio),
            stress=asdict(stress) if stress else None,bounds=bounds.tolist(),attempts=[primary_record],accepted_tier=None)
        self.lineages.append(lineage)
        try:
            initial_value=_number(fun(start)[0]);lineage['primary_start_nll']=initial_value
        except (FloatingPointError,ValueError,OverflowError) as exc:
            initial_value=None;lineage['primary_start_error']=f'{type(exc).__name__}: {exc}'
        comparisons=[value for value in (initial_value,primary_record.get('nll')) if _finite(value)]
        reference=min(comparisons) if comparisons else None;lineage['nonworsening_reference_nll']=reference
        cpu=self.cpu_history()
        cpu_fun=lambda pars:cpu.objective(pars,kernel,ratio,stress,True)
        for maxls in (40,200):
            retry_options=dict(options,maxls=maxls);retry_kwargs=dict(kwargs,options=retry_options);began=time.perf_counter()
            attempt=None
            try:
                retry=self.minimize(cpu_fun,start.copy(),*args,**retry_kwargs)
                attempt=_snapshot(retry,'original_cpu_fp64',retry_options,time.perf_counter()-began)
                lineage['attempts'].append(attempt)
                if not bool(retry.success):continue
                pars=np.asarray(retry.x,dtype=float);feasible=bool(pars.shape==start.shape and np.isfinite(pars).all() and
                    np.all(pars>=bounds[:,0]) and np.all(pars<=bounds[:,1]))
                cpu_value,cpu_gradient=cpu_fun(pars);primary_value,primary_gradient=fun(pars)
                returned_valid=_valid_result(retry,bounds)
                finite=bool(returned_valid and _finite(cpu_value) and _finite(primary_value) and
                    np.isfinite(cpu_gradient).all() and np.isfinite(primary_gradient).all())
                agreement=bool(finite and abs(primary_value-cpu_value)<AGREEMENT_TOLERANCE and abs(retry.fun-cpu_value)<AGREEMENT_TOLERANCE)
                nonworsening=bool(finite and reference is not None and max(cpu_value,float(retry.fun))<=reference+NONWORSENING_TOLERANCE)
                attempt['acceptance']=dict(finite=finite,feasible=feasible,returned_result_valid=returned_valid,cpu_nll=_number(cpu_value),primary_nll=_number(primary_value),
                    primary_minus_cpu_nll=_number(primary_value-cpu_value),nonworsening=nonworsening,
                    cpu_primary_agreement=agreement,agreement_tolerance_strict=AGREEMENT_TOLERANCE,
                    nonworsening_tolerance=NONWORSENING_TOLERANCE,cpu_gradient=_vector(cpu_gradient),
                    primary_gradient=_vector(primary_gradient))
                if feasible and agreement and nonworsening:
                    lineage['accepted_tier']=f'original_cpu_maxls{maxls}'
                    return retry
            except (FloatingPointError,ValueError,OverflowError) as exc:
                if attempt is None:lineage['attempts'].append(_snapshot(None,'original_cpu_fp64',retry_options,time.perf_counter()-began,exc))
                else:attempt['acceptance_error']=f'{type(exc).__name__}: {exc}'
        if original_error is not None:raise original_error
        if not _valid_result(primary,kwargs['bounds']):
            raise FloatingPointError('Invalid primary optimizer result; numerical recovery failed; raw lineage retained')
        return primary


def fit_with_recovery(history,maxiter=100,ratios=(.4,.7,.8),deadline=None,include_omori=True):
    """Original fit signature; production uses maxiter100 and include_omori=False.

    A failed retry remains failed. Selected independent CPU-order32 gates remain
    the responsibility of the unchanged completion worker's validation stage.
    """
    source=Path(ep.__file__)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=EXPECTED_EMPIRICAL_SHA256:
        raise RuntimeError('Original scientific source differs from reviewed frozen code')
    if maxiter!=100:raise ValueError('Recovery keeps original maxiter100')
    controller=RecoveryController(history)
    original=ep.fit_candidates
    cloned=types.FunctionType(original.__code__,dict(original.__globals__,minimize=controller),
        name=original.__name__,argdefs=original.__defaults__,closure=original.__closure__)
    cloned.__kwdefaults__=original.__kwdefaults__
    result=cloned(history,maxiter=maxiter,ratios=ratios,deadline=deadline,include_omori=include_omori)
    optimizer_records=[(i,r) for i,r in enumerate(result['subfits']) if r['origin']=='optimizer']
    if len(optimizer_records)!=controller.calls:raise RuntimeError('Optimizer lineage count differs')
    for lineage_index,line in enumerate(controller.lineages):
        index,record=optimizer_records[line['optimizer_call_index']]
        line['canonical_subfit_index']=index
        record['numerical_recovery_lineage_index']=lineage_index
    result['numerical_recovery']=dict(policy=POLICY,primary_optimizer_calls=controller.calls,
        failed_primary_starts=len(controller.lineages),accepted_recoveries=sum(line['accepted_tier'] is not None for line in controller.lineages),
        agreement_tolerance_strict=AGREEMENT_TOLERANCE,nonworsening_tolerance=NONWORSENING_TOLERANCE,
        original_source_sha256=EXPECTED_EMPIRICAL_SHA256,lineages=controller.lineages,
        success_rule='Primary success unchanged; retry requires SciPy success, finite feasible values/gradients, original-primary/CPU agreement and non-worsening; original selected CPU32 gate remains required')
    return result
