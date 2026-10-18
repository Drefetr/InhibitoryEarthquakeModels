"""Restricted horizon feasibility pilot. No ranks, critical values or power estimates."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,copy,importlib.util,json,math,sys,time
from decimal import Decimal
from dataclasses import asdict
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'iteration_6/code'))
from checkpoint_io import read_json,sha256,sealed_json,atomic_bytes,canonical_hash,utc_now
import numpy as np
from scipy.optimize import minimize


def module(relative,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value


def transformed_background(a,d,H,T0):return a+d*(H/T0-1),d*H/T0


def prefix(arrays,H):
    exact=np.asarray(arrays['exact_time_strings']);count=sum(Decimal(str(t))<Decimal.from_float(float(H)) for t in exact)
    return {k:np.asarray(v)[:count].copy() for k,v in arrays.items()}


def fit_pair(arrays,law,config,context,deadline):
    sys.path.insert(0,str(ROOT/'iteration_20/code'))
    from restricted_history import RestrictedHistory
    kernel=law.trigger;ratio=law.ratio;records=[];h=fine=None
    try:
        search_deadline=deadline-config['validation_reserve_seconds']
        h=RestrictedHistory(arrays,law,clock_mode='paired_clock',order=16,backend='cuda',context=context,deadline=search_deadline)
        rate=float(np.clip(math.log(max(len(arrays['times']),1)/law.horizon*.5),-11.5,-.1))
        bounds=[(-12.,0.),(-2.,2.),(0.,.98),(0.,3.)]
        def start(point,indices,label):
            best=None;attempts=[];x0=np.asarray(point,dtype=float)
            for maxls in config['maxls_tiers']:
                then=time.perf_counter()
                def fun(x):
                    p=x0.copy();p[indices]=x;return h.objective(p,kernel,ratio,True,indices)
                try:
                    r=minimize(fun,x0[indices].copy(),jac=True,method='L-BFGS-B',bounds=[bounds[i] for i in indices],
                               options=dict(maxiter=config['maxiter'],maxls=maxls,ftol=1e-10,gtol=2e-6))
                    p=x0.copy();p[indices]=r.x;value=h.objective(p,kernel,ratio,False)
                    ok=bool(r.success and np.isfinite(p).all() and np.isfinite(value) and abs(value-r.fun)<1e-4 and value<=h.objective(x0,kernel,ratio,False)+1e-8)
                    attempts.append(dict(success=ok,optimizer_success=bool(r.success),message=str(r.message),pars=p.tolist(),nll=value,maxls=maxls,nit=int(r.nit),nfev=int(r.nfev),elapsed_seconds=time.perf_counter()-then))
                    if ok:best=dict(pars=p.tolist(),nll=value,trigger=asdict(kernel),ratio=ratio);break
                except Exception as exc:
                    attempts.append(dict(success=False,maxls=maxls,error=type(exc).__name__+': '+str(exc),elapsed_seconds=time.perf_counter()-then))
                    if time.perf_counter()>=search_deadline:break
            records.append(dict(label=label,start=x0.tolist(),free=indices,attempts=attempts,complete=best is not None))
            if best is None:raise RuntimeError('Unresolved prescribed solver '+label)
            return best
        nulls=[start([rate,0.,.35,0.],[0,1,2],'null0'),start([max(-12.,rate-.5),.1,.7,0.],[0,1,2],'null1')]
        null=min(nulls,key=lambda p:p['nll']);alts=[copy.deepcopy(null)]
        for eta in [0.,.2]:
            point=list(null['pars']);point[3]=eta;alts.append(start(point,[0,1,2,3],'alternative_eta_'+str(eta)))
        alternative=min(alts,key=lambda p:p['nll']);h.deadline=deadline
        c16=h.contrast(null,alternative);gpu_values=[h.objective(p['pars'],kernel,ratio,False) for p in [null,alternative]]
        mark=h.mark_nll_constant;h.close();h=None;context.torch.cuda.empty_cache()
        fine=RestrictedHistory(arrays,law,clock_mode='paired_clock',order=32,backend='cpu',deadline=deadline)
        cpu_values=[fine.objective(p['pars'],kernel,ratio,False) for p in [null,alternative]];c32=fine.contrast(null,alternative)
        deltas=[abs(a-b) for a,b in zip(gpu_values,cpu_values)]
        complete=max(deltas)<1e-4 and abs(c16['statistic']-c32['statistic'])<1e-4 and all(abs(c['direct_minus_subtraction'])<1e-4 and c['statistic']>=-1e-4 for c in [c16,c32])
        for p in [null,alternative]:p['full_marked_nll']=p['nll']+mark
        return dict(complete=complete,status='complete' if complete else 'numerical_gate_failed',null=null,alternative=alternative,
                    contrast16=c16,contrast32=c32,selected_NLL16=gpu_values,selected_NLL32=cpu_values,deltas=deltas,records=records,
                    fixed_beta=law.beta,coordinate_horizon=law.coordinate,clock_mode='paired_clock',mark_nll_constant=mark)
    except Exception as exc:
        return dict(complete=False,status='fit_unresolved',error=type(exc).__name__+': '+str(exc),records=records)
    finally:
        for obj in [h,fine]:
            if obj is not None:obj.close()


def run(config,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);started=time.perf_counter();deadline=started+config['wall_cap_seconds']
    for p,h in config['source_hashes'].items():
        if sha256(ROOT/p)!=h:raise ValueError('Frozen source changed: '+p)
    if config['status']!='frozen_before_horizon_pilot_outcomes':raise ValueError('Unfrozen horizon pilot')
    sealed_json(folder/'protocol.json',config);context=None;states={i:'pending' for i in config['planned_ids']};costs=[]
    source12=module('iteration_12/code/clock_simulator.py','_horizon27_precise_clock')
    sys.path.insert(0,str(ROOT/'iteration_18/code'));sys.path.insert(0,str(ROOT/'iteration_17/code'))
    from cuda_oracle import CudaOracleContext
    from oracle_likelihood import law_from_setting
    c=config['generating_law'];T0=c['T0'];last_error=None
    try:
        context=CudaOracleContext(max_allocated_bytes=2**30);warm=context.warmup()
        for rep in range(len(config['seeds']['null'])):
            long_paths={}
            for arm,eta in [('null',0.),('inhibited',c['eta'])]:
                seed=config['seeds'][arm][rep];key=f'{arm}_r{rep:02d}';H=4*T0;fit=copy.deepcopy(c['fit']);a,d=transformed_background(fit['pars'][0],fit['pars'][1],H,T0)
                fit['pars']=[a,d,fit['pars'][2],eta];fit['stress']=copy.deepcopy(fit['trigger'])
                remaining=deadline-time.perf_counter()
                if remaining<=config['cleanup_reserve_seconds']:raise TimeoutError('global pilot budget exhausted')
                sim=source12.simulate_fitted(seed,H,c['beta'],fit,max_events=config['max_events'],max_proposals=config['max_proposals'],seconds=min(config['simulation_seconds'],remaining-config['cleanup_reserve_seconds']))
                arrays={k:sim[k] for k in source12.ARRAY_KEYS};path=folder/(key+'.npz');np.savez_compressed(path,**arrays)
                meta={k:v for k,v in sim.items() if k not in arrays and k!='checkpoint_state'}
                sealed_json(folder/(key+'_simulation.json'),dict(seed=seed,arm=arm,complete=sim['complete'],metadata=meta,checkpoint_state=sim['checkpoint_state'],arrays_sha256=sha256(path),background_reparameterization=dict(aH=a,dH=d,H=H,T0=T0),original_generating_fit=c['fit']))
                if not sim['complete']:
                    for mult in [1,2,4]:states[f'{key}_h{mult}']='simulation_censored'
                    raise RuntimeError('Simulation unresolved: '+sim['reason'])
                long_paths[arm]=(arrays,path,seed,key)
            for mult in [1,2,4]:
                for arm in ['null','inhibited']:
                    arrays,path,seed,key=long_paths[arm];id=f'{key}_h{mult}';began=time.perf_counter();H=mult*T0
                    if deadline-began<=config['cleanup_reserve_seconds']+config['validation_reserve_seconds']:
                        states[id]='resource_limited';raise TimeoutError('Insufficient remaining validation reserve')
                    cut=prefix(arrays,H)
                    law=law_from_setting(dict(beta=c['beta'],horizon=H,coordinate_horizon=T0,generating_fit=dict(c['fit'],stress=c['fit']['trigger'])))
                    payload=fit_pair(cut,law,config,context,min(deadline-config['cleanup_reserve_seconds'],began+config['fit_seconds']))
                    status='complete' if payload['complete'] else 'fit_unresolved';states[id]=status;elapsed=time.perf_counter()-began;costs.append(elapsed)
                    sealed_json(folder/(id+'_fit.json'),dict(id=id,status=status,seed=seed,arm=arm,horizon=H,n_events=len(cut['times']),payload=payload,elapsed_seconds=elapsed,parent_catalogue_sha256=sha256(path)))
                    sealed_json(folder/'checkpoint.json',dict(updated_utc=utc_now(),states=states,elapsed_seconds=time.perf_counter()-started))
                    print(id,status,len(cut['times']),round(elapsed,3),flush=True)
                    if not payload['complete']:raise RuntimeError('First unresolved fit: '+id)
    except Exception as exc:last_error=type(exc).__name__+': '+str(exc)
    finally:
        if context is not None:context.close()
        summary=dict(completed_utc=utc_now(),states=states,counts={s:list(states.values()).count(s) for s in set(states.values())},error=last_error,
                     all_complete=all(s=='complete' for s in states.values()),elapsed_seconds=time.perf_counter()-started,
                     max_planned_fits=len(states),new_precise_long_paths_max=10,no_power_estimate=True,no_null_calibration=True,
                     scope='Restricted fixed-kernel/mark-law horizon feasibility only; not a sample-size guarantee or full-nuisance power study')
        sealed_json(folder/'summary.json',summary)
    return 0 if summary['all_complete'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if sha256(args.protocol)!=args.protocol_sha256:raise ValueError('Protocol checksum differs')
    raise SystemExit(run(read_json(args.protocol),args.output))
