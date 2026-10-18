"""The prospectively restricted 150-start nuisance-ablation estimator.

No file writes, saved-data execution or CUDA initialization on import. All
primary, retry and witness records remain available to the caller.
"""
from dataclasses import asdict
import argparse,json,math,sys,time
import numpy as np
from scipy.optimize import minimize,OptimizeResult
from restricted_history import RestrictedHistory,KernelSpec,law_from_setting,oracle,cpu
CudaOracleContext=oracle.CudaOracleContext
PROCEDURES=('known_known','background_estimated','trigger_estimated','both_estimated')
BOUNDS=np.array([[-12.,0.],[-2.,2.],[0.,.98],[0.,3.]])
LBFGS=dict(maxiter=100,maxls=40,ftol=1e-10,gtol=2e-6)
POWELL=dict(maxiter=100,maxfev=2000,ftol=1e-10,xtol=1e-8)


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,(np.floating,float)):return float(value) if math.isfinite(value) else None
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value


def snapshot(result,elapsed,tier):
    if result is None:return dict(tier=tier,elapsed_seconds=elapsed,success=False,error='No result')
    return clean(dict(tier=tier,elapsed_seconds=elapsed,pars=np.asarray(result.x),nll=float(result.fun),success=bool(result.success),status=int(result.status),returned_finite=bool(np.isfinite(result.x).all() and np.isfinite(result.fun)),
                      message=str(result.message),nit=int(getattr(result,'nit',0)),nfev=int(getattr(result,'nfev',0)),jac=getattr(result,'jac',None)))


def feasible(result,bounds):
    x=np.asarray(result.x,dtype=float)
    return bool(result.success and x.shape==(len(bounds),) and np.isfinite(x).all() and np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1]) and
                np.isfinite(result.fun) and (getattr(result,'jac',None) is None or np.isfinite(result.jac).all()))


def fit_start(history,cpu_factory,start,free,kernel,ratio,metadata,deadline):
    """One prescribed start, with uniform failure-only CPU and value tiers."""
    began=time.perf_counter();start=np.asarray(start,dtype=float);free=list(free);bounds=BOUNDS[free];x0=start[free];attempts=[];selected=None;selected_tier=None;cpu_history=None
    record=dict(metadata,start=start.tolist(),free_indices=free,bounds=bounds.tolist(),trigger=asdict(kernel),ratio=float(ratio),origin='optimizer')
    def full(x):p=start.copy();p[free]=x;return p
    def fun(h,x,gradient=True):cpu._deadline(deadline);return h.objective(full(x),kernel,ratio,gradient,free)
    def reference_value(h,x):return float(fun(h,x,False))
    reference=reference_value(history,x0)
    def run_solver(h,tier,maxls):
        tick=time.perf_counter()
        try:
            result=minimize(lambda x:fun(h,x,True),x0.copy(),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(LBFGS,maxls=maxls))
            attempts.append(snapshot(result,time.perf_counter()-tick,tier));return result
        except (FloatingPointError,ValueError,OverflowError) as exc:
            attempts.append(dict(tier=tier,success=False,error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-tick));return None
    primary=run_solver(history,'primary_cuda40',40)
    if primary is not None and feasible(primary,bounds):
        value=reference_value(history,primary.x)
        if abs(value-primary.fun)<1e-4 and max(value,float(primary.fun))<=reference+1e-8:selected=primary;selected_tier='primary_cuda40'
    if selected is None:
        cpu_history=cpu_factory();reference=min(reference,reference_value(cpu_history,x0))
        def accept(result):
            nonlocal reference
            if result is None or not feasible(result,bounds):return False
            cv=reference_value(cpu_history,result.x);gv=reference_value(history,result.x)
            return bool(abs(cv-gv)<1e-4 and abs(cv-result.fun)<1e-4 and max(cv,gv,float(result.fun))<=reference+1e-8)
        # Only independently validated finite failed candidates enter the bound.
        if primary is not None and np.isfinite(primary.fun) and np.isfinite(primary.x).all() and np.all(primary.x>=bounds[:,0]) and np.all(primary.x<=bounds[:,1]):
            pv=reference_value(cpu_history,primary.x);pg=reference_value(history,primary.x)
            if abs(pv-pg)<1e-4 and abs(pv-primary.fun)<1e-4:reference=min(reference,pv,pg,float(primary.fun))
        for tier,maxls in (('retry_cpu40',40),('retry_cpu200',200)):
            retry=run_solver(cpu_history,tier,maxls)
            if accept(retry):selected=retry;selected_tier=tier;break
            if retry is not None and np.isfinite(retry.fun) and np.isfinite(retry.x).all() and np.all(retry.x>=bounds[:,0]) and np.all(retry.x<=bounds[:,1]):
                cv=reference_value(cpu_history,retry.x);gv=reference_value(history,retry.x)
                if abs(cv-gv)<1e-4 and abs(cv-retry.fun)<1e-4:reference=min(reference,cv,gv,float(retry.fun))
        if selected is None:
            tick=time.perf_counter()
            try:
                value=lambda x:fun(cpu_history,x,False)
                result=minimize(value,x0.copy(),method='Powell',bounds=bounds,options=dict(POWELL));line=snapshot(result,time.perf_counter()-tick,'retry_cpu_Powell');attempts.append(line)
                full_ok=accept(result);face_ok=True;face_result=None
                if 2 in free:
                    npos=free.index(2);remaining=[i for i in range(len(free)) if i!=npos];face_start=x0.copy();face_start[npos]=0.;face_reference=min(value(face_start),reference_value(history,face_start))
                    def reconstruct_face(x):p=face_start.copy();p[remaining]=x;return p
                    def face(x):
                        cpu._deadline(deadline);p=full(reconstruct_face(x));return cpu_history.objective(p,kernel,ratio,True,[free[i] for i in remaining])
                    if remaining:
                        face_result=minimize(face,face_start[remaining],jac=True,method='L-BFGS-B',bounds=bounds[remaining],options=dict(LBFGS,maxls=200))
                        face_point=reconstruct_face(face_result.x);fv=value(face_point);fg=reference_value(history,face_point)
                        face_ok=feasible(face_result,bounds[remaining]) and abs(fv-fg)<1e-4 and abs(fv-face_result.fun)<1e-4 and max(fv,fg,float(face_result.fun))<=face_reference+1e-8
                    else:
                        fv=value(face_start);fg=reference_value(history,face_start)
                        face_ok=math.isfinite(fv) and math.isfinite(fg) and abs(fv-fg)<1e-4 and max(fv,fg)<=face_reference+1e-8
                        face_result=OptimizeResult(x=np.empty(0),fun=fv,success=True,status=0,message='Exact zero-dimensional n0 face',nit=0,nfev=1)
                    attempts.append(snapshot(face_result,time.perf_counter()-tick,'retry_exact_n0_face'))
                    line['face_own_start_nll']=face_reference;line['face_passed']=bool(face_ok)
                    if face_ok:
                        face_result=OptimizeResult(x=reconstruct_face(face_result.x),fun=float(face_result.fun),success=True,status=0,message='Exact n0 face converged',nit=face_result.nit,nfev=face_result.nfev)
                # Both declared solvers must succeed even if one candidate wins.
                if full_ok and face_ok:
                    chosen=result if face_result is None or result.fun<=face_result.fun else face_result
                    if accept(chosen):selected=chosen;selected_tier='retry_cpu_Powell' if chosen is result else 'retry_exact_n0_face'
            except (FloatingPointError,ValueError,OverflowError) as exc:attempts.append(dict(tier='retry_value_tier',success=False,error=f'{type(exc).__name__}: {exc}'))
    record.update(success=selected is not None,attempts=attempts,elapsed_seconds=time.perf_counter()-began,nonworsening_reference_nll=reference)
    if selected is None:record.update(pars=start.tolist(),nll=None,message='Unresolved prescribed start; partial search is not accepted')
    else:
        pars=full(selected.x);nll=history.objective(pars,kernel,ratio,False);record.update(pars=pars.tolist(),nll=float(nll),status=int(selected.status),message=str(selected.message),nit=int(getattr(selected,'nit',0)),nfev=int(getattr(selected,'nfev',0)),accepted_tier=selected_tier)
        record['boundary_hits']=[['a','d','n','eta'][i] for i in free if np.isclose(pars[i],BOUNDS[i,0],atol=1e-7,rtol=0) or np.isclose(pars[i],BOUNDS[i,1],atol=1e-7,rtol=0)]
        try:
            _,g=history.objective(pars,kernel,ratio,True,free);projected=g.copy()
            for j,i in enumerate(free):
                if pars[i]<=BOUNDS[i,0]+1e-8 and g[j]>0 or pars[i]>=BOUNDS[i,1]-1e-8 and g[j]<0:projected[j]=0.
            record.update(gradient=g.tolist(),gradient_infinity_norm=float(np.max(np.abs(g))),projected_gradient_infinity_norm=float(np.max(np.abs(projected))))
        except FloatingPointError as exc:record.update(gradient=None,gradient_unrepresentable=str(exc))
    return clean(record)


def validate_contract(contract):
    p=contract['parameters'];s=contract['solver']
    if p['trigger']['estimated_r_grid']!=[.4,.7,.8] or p['trigger']['estimated_alpha_grid']!=[.5,.7] or p['trigger']['estimated_tau_days_grid']!=[.1,1.,10.]:raise ValueError('Exact18 trigger cells required')
    if s['primary']['maxiter']!=100 or s['primary']['maxls']!=40 or contract['bookkeeping']['per_history_prescribed_minimizations']!=150:raise ValueError('Exact150-start recipe required')
    if s['primary']['ftol']!=1e-10 or s['primary']['gtol']!=2e-6 or s['powell']!=POWELL or p['background']['estimated_bounds']!=BOUNDS[:2].tolist() or p['trigger']['estimated_n_bounds']!=BOUNDS[2].tolist() or p['eta']['bounds']!=BOUNDS[3].tolist():raise ValueError('Solver or parameter bounds differ')
    if [p['id'] for p in contract['procedures']]!=list(PROCEDURES) or [p['prescribed_minimizations'] for p in contract['procedures']]!=[2,4,72,72]:raise ValueError('Four procedure counts differ')


def fit_case(arrays,law,*,clock_mode,contract,deadline,context,progress_callback=None):
    """All four procedures, 150 prescribed calls, then independent CPU32 gates.

    A deadline/exception retains returned partials and callback start records.
    The caller owns persistence and hard process/resource enforcement.
    """
    validate_contract(contract);began=time.perf_counter();methods=[];records=[];history=None;cpu16=None;cpu32=None;failure=None
    result=dict(complete=False,status='partial',methods=methods,records=records,planned_optimizer_calls=150,prescribed_optimizer_calls=0)
    def emit(event):
        if progress_callback is not None:progress_callback(clean(event))
    try:
        context.begin_measurement();history=RestrictedHistory(arrays,law,clock_mode=clock_mode,order=16,backend='cuda',context=context,deadline=deadline)
        def cpu_factory():
            nonlocal cpu16
            if cpu16 is None:cpu16=RestrictedHistory(arrays,law,clock_mode=clock_mode,order=16,backend='cpu',deadline=deadline)
            return cpu16
        a0=float(np.clip(math.log(max(len(history.times)/law.horizon,1e-8)),-12,0));known=np.array([law.a,law.d,law.n,0.]);global_cells=[(KernelSpec('ml',tau,alpha=alpha),ratio) for ratio in (.4,.7,.8) for alpha in (.5,.7) for tau in (.1,1.,10.)]
        def point(pars,kernel,ratio,origin,method,phase):return dict(pars=np.asarray(pars).tolist(),nll=history.objective(pars,kernel,ratio,False),trigger=asdict(kernel),ratio=float(ratio),origin=origin,success=True,method=method,phase=phase)
        for mid in PROCEDURES:
            bg=mid in ('background_estimated','both_estimated');tr=mid in ('trigger_estimated','both_estimated');cells=global_cells if tr else [(law.trigger,law.ratio)]
            method=dict(id=mid,complete=False,subfits=[],null=None,alternative=None,cells=[]);methods.append(method)
            free0=([0,1] if bg else [])+([2] if tr else []);free1=free0+[3]
            for cell_index,(kernel,ratio) in enumerate(cells):
                cpu._deadline(deadline);null_candidates=[]
                if not free0:null_candidates=[point(known,kernel,ratio,'fixed_null',mid,'null')]
                else:
                    for si in range(2):
                        start=known.copy()
                        if bg:start[:2]=[a0,0.] if si==0 else [float(np.clip(a0-.5,-12,0)),.1]
                        if tr:start[2]=(.35,.7)[si]
                        rec=fit_start(history,cpu_factory,start,free0,kernel,ratio,dict(method=mid,phase='null',cell_index=cell_index,start_index=si),deadline)
                        records.append(rec);method['subfits'].append(rec);result['prescribed_optimizer_calls']+=1;emit(dict(event='optimizer_completed',method=mid,index=len(records)-1,record=rec))
                        if not rec['success']:raise RuntimeError('Unresolved prescribed null start')
                        null_candidates.append(rec)
                null=min(null_candidates,key=lambda r:r['nll']);alternative_candidates=[point(np.r_[null['pars'][:3],0.],kernel,ratio,'exact_matching_null_subset',mid,'alternative')]
                for si,eta in enumerate((0.,.2)):
                    start=np.asarray(null['pars']).copy();start[3]=eta
                    rec=fit_start(history,cpu_factory,start,free1,kernel,ratio,dict(method=mid,phase='alternative',cell_index=cell_index,start_index=si),deadline)
                    records.append(rec);method['subfits'].append(rec);result['prescribed_optimizer_calls']+=1;emit(dict(event='optimizer_completed',method=mid,index=len(records)-1,record=rec))
                    if not rec['success']:raise RuntimeError('Unresolved prescribed alternative start')
                    alternative_candidates.append(rec)
                alternative=min(alternative_candidates,key=lambda r:r['nll']);method['cells'].append(dict(index=cell_index,trigger=asdict(kernel),ratio=ratio,null=null,alternative=alternative,exact_null_subset=alternative_candidates[0]))
                if method['null'] is None or null['nll']<method['null']['nll']:method['null']=null
                if method['alternative'] is None or alternative['nll']<method['alternative']['nll']:method['alternative']=alternative
            method['contrast16']=history.contrast(method['null'],method['alternative']);method['optimization_complete']=True
            emit(dict(event='procedure_optimized',method=mid,optimizer_calls=len(method['subfits']),partial_method=method))
        if len(records)!=150:raise RuntimeError('Prescribed call count differs')
        # Witnesses cannot alter any start or selected maximum.
        witnesses=[];kk=methods[0];witness=point([law.a,law.d,law.n,law.eta if hasattr(law,'eta') else contract['law']['alternative_eta']],law.trigger,law.ratio,'fixed_eta_witness','known_known','alternative')
        fixed_contrast=history.contrast(kk['null'],witness)
        witnesses.append(dict(kind='fixed_eta_star',method='known_known',witness_nll=witness['nll'],selected_nll=kk['alternative']['nll'],difference=kk['alternative']['nll']-witness['nll'],
                              contrast16=fixed_contrast,selected_GLR16=kk['contrast16']['statistic'],direct_GLR_shortfall=fixed_contrast['statistic']-kk['contrast16']['statistic']))
        for broad,narrow in ((1,0),(2,0),(3,1),(3,2)):
            for nc in methods[narrow]['cells']:
                bc=next(c for c in methods[broad]['cells'] if c['trigger']==nc['trigger'] and c['ratio']==nc['ratio'])
                for phase in ('null','alternative'):
                    rec=nc[phase];witness_value=history.objective(rec['pars'],KernelSpec(**rec['trigger']),rec['ratio'],False)
                    witnesses.append(dict(kind='cross_procedure_subset',broader=methods[broad]['id'],narrower=methods[narrow]['id'],broader_cell=bc['index'],narrower_cell=nc['index'],phase=phase,
                                          witness_nll=witness_value,selected_nll=bc[phase]['nll'],difference=bc[phase]['nll']-witness_value))
        result['witnesses']=witnesses
        result['cuda_metrics']=context.cuda_metrics(32768);result['feature_metadata']=dict(nodes16=history.nodes,panels=history.panels,stress_modes=len(law.stress.mixture[0]),preparation_seconds=history.preparation_seconds,clock_mode=history.mode,mark_nll_constant=history.mark_nll_constant)
        history.close();context.torch.cuda.empty_cache()
        cpu32=RestrictedHistory(arrays,law,clock_mode=clock_mode,order=32,backend='cpu',deadline=deadline)
        gates=[]
        for method in methods:
            for phase in ('null','alternative'):
                rec=method[phase];value=cpu32.objective(rec['pars'],KernelSpec(**rec['trigger']),rec['ratio'],False);delta=abs(value-rec['nll'])
                gates.append(dict(method=method['id'],phase=phase,cuda16_nll=rec['nll'],cpu32_nll=value,absolute_difference=delta,passed=delta<1e-4))
            method['contrast32']=cpu32.contrast(method['null'],method['alternative']);delta=abs(method['contrast32']['statistic']-method['contrast16']['statistic'])
            method['gates']=dict(statistic_16_32_difference=delta,nll_subtraction_control_gate=1e-4,direct_nesting_passed=method['contrast16']['statistic']>=-1e-4 and method['contrast32']['statistic']>=-1e-4,
                                 passed=delta<1e-4 and abs(method['contrast16']['direct_minus_subtraction'])<1e-4 and abs(method['contrast32']['direct_minus_subtraction'])<1e-4 and method['alternative']['nll']<=method['null']['nll']+1e-8 and method['contrast16']['statistic']>=-1e-4 and method['contrast32']['statistic']>=-1e-4)
        fixed32=cpu32.contrast(kk['null'],witness);witnesses[0]['contrast32']=fixed32;witnesses[0]['selected_GLR32']=kk['contrast32']['statistic'];witnesses[0]['direct_GLR32_shortfall']=fixed32['statistic']-kk['contrast32']['statistic']
        for w in witnesses:w['passed']=w['difference']<=1e-4 and w.get('direct_GLR_shortfall',0.)<=1e-4 and w.get('direct_GLR32_shortfall',0.)<=1e-4
        for method in methods:
            method['complete']=method['gates']['passed'] and all(g['passed'] for g in gates if g['method']==method['id']) and all(w['passed'] for w in witnesses if w.get('broader',w.get('method'))==method['id'])
        result['validation']=dict(selected_objectives=gates,cpu32_feature_seconds=cpu32.preparation_seconds,nodes32=cpu32.nodes,witnesses_passed=all(w['passed'] for w in witnesses))
        complete=all(m['complete'] for m in methods) and result['validation']['witnesses_passed']
        result.update(complete=bool(complete),status='complete' if complete else 'unresolved_validation')
    except Exception as exc:
        result.update(complete=False,status='time_cap' if isinstance(exc,TimeoutError) else 'unresolved',error=f'{type(exc).__name__}: {exc}')
    finally:
        for h in (history,cpu16,cpu32):
            if h is not None:h.close()
        result['elapsed_seconds']=time.perf_counter()-began
    return clean(result)


def main():
    p=argparse.ArgumentParser(description='Restricted150-start evaluator library; no saved-data fit CLI. Use reviewed root benchmark harness.');p.parse_args();print('fit_case(arrays,law,clock_mode=...,contract=...,deadline=...,context=...,progress_callback=...)')


if __name__=='__main__':main()
