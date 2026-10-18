"""Independent value-only amplitude-profile search; no saved-data work on import.

This is a new diagnostic algorithm, not the frozen iteration-21 estimator.
It imports that estimator's immutable history/features only for exact-law
geometry and independent pointwise controls. No old optimizer is called.
"""
from dataclasses import asdict
from pathlib import Path
import copy,hashlib,json,math,sys,time
import numpy as np
from scipy.optimize import brentq,minimize,minimize_scalar
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[2]
STAGED_SHA='338145457dd54d90ac929fda32224adcb7c02870e884724341f6222a4fce6f44'
path=ROOT/'iteration_21/code/staged_fit.py'
if hashlib.sha256(path.read_bytes()).hexdigest()!=STAGED_SHA:raise ValueError('Frozen staged numerical authority changed')
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'iteration_21/code'))
import staged_fit as authority
RestrictedHistory=authority.RestrictedHistory
KernelSpec=authority.KernelSpec
CudaOracleContext=authority.CudaOracleContext
law_from_setting=authority.law_from_setting
cpu=authority.cpu
clean=authority.clean
background_integrals=authority.original.RestrictedHistory.objective.__globals__['background_integrals']

SEARCH_SPEC=dict(a_bounds=[-12.,0.],d_bounds=[-2.,2.],eta_bounds=[0.,3.],amplitude_xtol=2e-12,
                 amplitude_rtol=4*np.finfo(float).eps,amplitude_maxiter=100,KK_eta_grid_points=17,null_d_grid_points=17,
                 alternative_d_grid_points=7,alternative_eta_grid_points=7,powell_grid_seeds=3,include_old_alternative_and_null_seeds=True,
                 powell_options=dict(maxiter=2000,maxfev=100000,xtol=1e-6,ftol=1e-10),scalar_xatol=1e-7,scalar_maxiter=100,
                 eta_profile_points=11,scalar_reduction_chunk_nodes=2**20,modal_query_chunk_nodes=32768,
                 local_bracket_rule='center<=both_neighbors_and_strictly_less_than_at_least_one;skip_fully_flat',
                 failed_solver_return_eligible=False,all_mandated_solvers_must_succeed=True)
TOL=1e-4
METHODS=('known_known','background_estimated')


def _arrays(logq,logc):
    q=np.asarray(logq,dtype=float);c=np.asarray(logc,dtype=float)
    if q.ndim!=1 or q.shape!=c.shape or np.isnan(q).any() or np.isnan(c).any() or np.isposinf(q).any() or np.isposinf(c).any() or np.any(np.isneginf(q)&np.isneginf(c)):
        raise ValueError('Invalid/unrepresentable event log components')
    return q,c


def event_probabilities(a,logq,logc):
    """Stable background share, including exact c=0 without forming q or c."""
    q,c=_arrays(logq,logc);p=np.empty(len(q));zero=np.isneginf(c)
    p[zero]=1.;p[~zero]=expit(a+q[~zero]-c[~zero])
    return p


def amplitude_value(a,logq,logc,J,trigger_compensator=0.):
    q,c=_arrays(logq,logc)
    if not math.isfinite(a) or not math.isfinite(J) or J<0 or not math.isfinite(trigger_compensator):raise ValueError('Nonfinite amplitude objective inputs')
    logs=np.logaddexp(a+q,c)
    if not np.isfinite(logs).all():raise FloatingPointError('Event log intensity not representable')
    value=math.fsum((math.exp(a)*J,trigger_compensator,-math.fsum(logs)))
    if not math.isfinite(value):raise FloatingPointError('Ground likelihood not representable')
    return float(value)


def amplitude_profile(logq,logc,J,*,a_bounds=(-12.,0.),xtol=2e-12,rtol=4*np.finfo(float).eps,maxiter=100):
    """Solve the bounded convex problem in A=exp(a), using monotone F_A.

    F_A(a)=J-exp(-a)*sum(pi). The a-score A*F_A need not be
    monotone; it is not used as a monotonicity assertion or Newton derivative.
    Strict convexity needs at least one nonzero q; flat/empty cases retain a
    canonical bounded endpoint. Event count is never filtered or conditioned.
    """
    q,c=_arrays(logq,logc);lo,hi=map(float,a_bounds)
    if not lo<hi or not math.isfinite(J) or J<0:raise ValueError('Invalid amplitude bounds/integral')
    evaluations=[]
    def derivative(a):
        value=J-math.exp(-a)*math.fsum(event_probabilities(a,q,c));evaluations.append(dict(a=float(a),F_A=float(value)))
        if not math.isfinite(value):raise FloatingPointError('Amplitude derivative not representable')
        return value
    left=derivative(lo);right=derivative(hi)
    if left>=0:a=lo;boundary='lower';iterations=0
    elif right<=0:a=hi;boundary='upper';iterations=0
    else:
        a,result=brentq(derivative,lo,hi,xtol=xtol,rtol=rtol,maxiter=maxiter,full_output=True,disp=True)
        if not result.converged:raise RuntimeError('Amplitude root did not converge')
        boundary=None;iterations=int(result.iterations)
    residual=derivative(a)
    return dict(a=float(a),A=math.exp(a),boundary=boundary,root_converged=True,iterations=iterations,root_evaluations=evaluations,
                F_A_lower=left,F_A_upper=right,F_A_at_solution=residual,n_events=len(q),objective_without_trigger=amplitude_value(a,q,c,J))


class ProfileObjective:
    """One fixed-stress/trigger history. Cache scalar results, never event matrices."""
    def __init__(self,history,*,context=None,deadline=None,scalar_chunk_nodes=2**20):
        self.history=history;self.law=history.law;self.context=context;self.deadline=deadline
        self.scalar_chunk_nodes=int(scalar_chunk_nodes)
        if not 1<=self.scalar_chunk_nodes<=2**20:raise ValueError('Scalar reduction chunk exceeds declared policy')
        self.logh,self.K,_=history.trigger_features(self.law.trigger,self.law.ratio)
        self.logc=self.logh+math.log(self.law.n) if self.law.n>0 else np.full_like(self.logh,-np.inf)
        self.trigger_compensator=self.law.n*self.K;self.integrals={};self.cache={};self.evaluations=[];self.fixed_evaluations=[];self.errors=[];self.J_evaluations=0;self.profile_requests=0;self.cache_hits=0

    def integral(self,d,eta):
        cpu._deadline(self.deadline);key=(float(d),float(eta))
        if key in self.integrals:return self.integrals[key]
        h=self.history;parts=[]
        if eta==0:J=background_integrals(0.,d,self.law.horizon,self.law.coordinate)[0]
        else:
            quiet=background_integrals(0.,d,h.quiet_duration,self.law.coordinate)[0]
            if h.backend=='cuda':
                torch=self.context.torch
                with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
                    for j in range(0,h.nodes,self.scalar_chunk_nodes):
                        cpu._deadline(self.deadline);sl=slice(j,j+self.scalar_chunk_nodes)
                        x=h.device_arrays['xq'][sl];g=h.device_arrays['gq'][sl];w=h.device_arrays['wq'][sl]
                        value=w*torch.exp(d*x-eta*g);parts.append(float(torch.sum(value).item()));del value
                        self.context.memory_metrics()
                torch.cuda.synchronize(self.context.device)
            else:
                for j in range(0,h.nodes,self.scalar_chunk_nodes):
                    cpu._deadline(self.deadline);sl=slice(j,j+self.scalar_chunk_nodes)
                    parts.append(math.fsum(h.wq[sl]*np.exp(d*h.xq[sl]-eta*h.gq[sl])))
            J=math.fsum((quiet,math.fsum(parts)))
        cpu._deadline(self.deadline)
        if not math.isfinite(J) or J<=0:raise FloatingPointError('Background integral not finite/positive; no numeric-zero substitution')
        self.integrals[key]=float(J);self.J_evaluations+=1;return float(J)

    def logq(self,d,eta):
        with np.errstate(over='raise',invalid='raise'):q=d*self.history.xe-eta*self.history.ge
        if not np.isfinite(q).all():raise FloatingPointError('Background event exponent not representable')
        return q

    def fixed(self,a,d,eta,*,origin='fixed_point'):
        cpu._deadline(self.deadline);J=self.integral(d,eta);q=self.logq(d,eta);value=amplitude_value(a,q,self.logc,J,self.trigger_compensator)
        point=dict(pars=[float(a),float(d),self.law.n,float(eta)],nll=value,trigger=asdict(self.law.trigger),ratio=self.law.ratio,origin=origin,success=True)
        self.fixed_evaluations.append(point);return copy.deepcopy(point)

    def evaluate(self,d,eta):
        cpu._deadline(self.deadline);d=float(d);eta=float(eta);key=(d,eta)
        self.profile_requests+=1
        if not -2<=d<=2 or not 0<=eta<=3:raise ValueError('Search point outside original full domain')
        if key in self.cache:self.cache_hits+=1;return copy.deepcopy(self.cache[key])
        began=time.perf_counter()
        try:
            J=self.integral(d,eta);q=self.logq(d,eta)
            a=amplitude_profile(q,self.logc,J);value=amplitude_value(a['a'],q,self.logc,J,self.trigger_compensator)
            point=dict(pars=[a['a'],d,self.law.n,eta],nll=value,trigger=asdict(self.law.trigger),ratio=self.law.ratio,origin='amplitude_profile',success=True,
                       amplitude=a,J=J,evaluation_id=len(self.evaluations),elapsed_seconds=time.perf_counter()-began)
            self.cache[key]=point;self.evaluations.append(point);return copy.deepcopy(point)
        except Exception as exc:
            self.errors.append(dict(d=d,eta=eta,error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-began));raise


def _best(points):
    if not points:raise RuntimeError('No eligible finite candidate')
    return min(points,key=lambda p:p['nll'])


def scalar_search(value,grid,extra,*,label,deadline,records):
    """Grid + all strict observed local brackets, retaining every scalar status."""
    points=[value(float(x)) for x in grid];eligible=list(points)
    for x in extra:eligible.append(value(float(x)))
    brackets=[]
    for i in range(1,len(grid)-1):
        left,mid,right=[points[k]['nll'] for k in (i-1,i,i+1)]
        if mid<=left and mid<=right and (mid<left or mid<right):brackets.append((float(grid[i-1]),float(grid[i+1])))
    for index,bounds in enumerate(brackets):
        cpu._deadline(deadline);began=time.perf_counter()
        try:
            result=minimize_scalar(lambda x:value(float(x))['nll'],method='bounded',bounds=bounds,options=dict(xatol=1e-7,maxiter=100))
            rec=dict(label=label,index=index,bounds=list(bounds),success=bool(result.success),x=float(result.x),nll=float(result.fun),status=int(result.status),message=str(result.message),nit=int(result.nit),nfev=int(result.nfev),elapsed_seconds=time.perf_counter()-began)
            candidate=value(float(result.x));rec['candidate']=candidate
            if result.success and math.isfinite(result.fun) and abs(result.fun-candidate['nll'])<TOL:eligible.append(candidate)
            else:rec['success']=False
        except TimeoutError:raise
        except Exception as exc:rec=dict(label=label,index=index,bounds=list(bounds),success=False,error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-began)
        records.append(clean(rec))
    return _best(eligible),eligible,dict(label=label,grid_points=len(grid),extra_points=len(extra),local_brackets=[list(x) for x in brackets],solver_runs=len(brackets))


def _old_methods(oldcase,law):
    p=oldcase.get('payload',oldcase)
    if p.get('complete') is not True or [m['id'] for m in p['methods']]!=list(METHODS):raise ValueError('Complete two-method old case required')
    out={m['id']:m for m in p['methods']}
    for m in out.values():
        for phase in ('null','alternative'):
            r=m[phase]
            if r['pars'][2]!=law.n or r['trigger']!=asdict(law.trigger) or r['ratio']!=law.ratio:raise ValueError('Old fixed trigger/law differs')
            p=np.asarray(r['pars'],dtype=float)
            if p.shape!=(4,) or not np.isfinite(p).all() or not -12<=p[0]<=0 or not -2<=p[1]<=2 or not 0<=p[3]<=3 or not math.isfinite(r['nll']):raise ValueError('Old point outside original finite box')
        if m['null']['pars'][3]!=0:raise ValueError('Old null must be eta0')
    for phase in ('null','alternative'):
        if out['known_known'][phase]['pars'][:2]!=[law.a,law.d]:raise ValueError('Old known background was not supplied')
    return out


def fit_profile_case(arrays,law,*,clock_mode,protocol,oldcase,context,deadline,progress_callback=None):
    """New finite diagnostic search, preserving every old selected point floor.

    Failed optimizer returns remain reported feasible witnesses only. They are
    excluded from completed winner selection, and any mandated failure prevents
    complete status. No old critical value, rank, or power is computed here.
    """
    if protocol.get('search')!=clean(SEARCH_SPEC):raise ValueError('Exact prospectively frozen search specification required')
    began=time.perf_counter();history=None;independent=None;objective=None;search_records=[];stages=[];methods=[];controls=[];old=_old_methods(oldcase,law)
    ident=oldcase.get('id');reserve=protocol['resources']['validation_reserve_seconds'].get(ident,45.)
    search_deadline=deadline-reserve
    out=dict(complete=False,status='partial',algorithm='value_only_amplitude_profile',methods=methods,search_records=search_records,stages=stages,controls=controls)
    def emit(event):
        if progress_callback is not None:progress_callback(clean(event))
    try:
        cpu._deadline(search_deadline);context.begin_measurement();history=RestrictedHistory(arrays,law,clock_mode=clock_mode,order=16,backend='cuda',context=context,deadline=search_deadline)
        objective=ProfileObjective(history,context=context,deadline=search_deadline,scalar_chunk_nodes=SEARCH_SPEC['scalar_reduction_chunk_nodes'])
        # Rescore all old eligible floors in the independent value expression.
        old_points={mid:{phase:objective.fixed(r['pars'][0],r['pars'][1],r['pars'][3],origin='old_selected_floor') for phase,r in om.items() if phase in ('null','alternative')} for mid,om in old.items()}
        old_controls=[]
        for mid in METHODS:
            for phase in ('null','alternative'):
                point=old_points[mid][phase];saved=old[mid][phase]['nll'];delta=abs(point['nll']-saved)
                old_controls.append(dict(method=mid,phase=phase,saved_primary16_nll=saved,profile16_nll=point['nll'],absolute_difference=delta,passed=delta<TOL))
        out['old_primary_controls']=old_controls
        # K/K: fixed a,d; independent eta grid and every observed local bracket.
        kk_runs=[];eta_grid=np.linspace(0.,3.,17)
        kvalue=lambda eta:objective.fixed(law.a,law.d,eta,origin='known_known_eta_search')
        kk_alt,kk_candidates,kk_stage=scalar_search(kvalue,eta_grid,[old['known_known']['alternative']['pars'][3],0.],label='known_known_eta',deadline=search_deadline,records=kk_runs)
        kk_candidates.insert(0,old_points['known_known']['alternative']);kk_candidates.insert(0,old_points['known_known']['null']);kk_alt=_best(kk_candidates);search_records.extend(kk_runs)
        kk=dict(id='known_known',null=old_points['known_known']['null'],alternative=kk_alt,candidates=kk_candidates,search_runs=kk_runs,complete=False)
        methods.append(kk);stages.append(kk_stage);emit(dict(event='search_stage_completed',stage='known_known_eta',record=kk_stage,selected=kk_alt))
        # BG null: scalar d after exact bounded amplitude profiling.
        d_grid=np.linspace(-2.,2.,17);null_runs=[]
        null,nu_candidates,null_stage=scalar_search(lambda d:objective.evaluate(d,0.),d_grid,[old['background_estimated']['null']['pars'][1],old['background_estimated']['alternative']['pars'][1]],label='background_null',deadline=search_deadline,records=null_runs)
        nu_candidates.insert(0,old_points['background_estimated']['null']);null=_best(nu_candidates);search_records.extend(null_runs);stages.append(null_stage)
        emit(dict(event='search_stage_completed',stage='background_null',record=null_stage,selected=null))
        # Explicit 7x7 grid and canonical top-three grid starts plus both old starts.
        grid=[]
        for d in np.linspace(-2.,2.,7):
            for eta in np.linspace(0.,3.,7):grid.append(objective.evaluate(d,eta))
        alt_candidates=[old_points['background_estimated']['alternative'],old_points['background_estimated']['null']]+grid
        old_alt=old['background_estimated']['alternative']['pars'];old_null=old['background_estimated']['null']['pars']
        seeds=[p['pars'][1::2] for p in sorted(grid,key=lambda p:p['nll'])[:3]]+[[old_alt[1],old_alt[3]],[old_null[1],0.]]
        unique=[]
        for seed in seeds:
            seed=list(map(float,seed))
            if seed not in unique:unique.append(seed)
            alt_candidates.append(objective.evaluate(*seed))
        for index,start in enumerate(unique):
            cpu._deadline(search_deadline);tick=time.perf_counter()
            try:
                result=minimize(lambda x:objective.evaluate(float(x[0]),float(x[1]))['nll'],np.array(start),method='Powell',bounds=[(-2.,2.),(0.,3.)],options=dict(SEARCH_SPEC['powell_options']))
                finite=np.isfinite(result.x).all() and np.isfinite(result.fun) and -2<=result.x[0]<=2 and 0<=result.x[1]<=3
                point=objective.evaluate(*result.x) if finite else None
                success=bool(result.success and finite and abs(result.fun-point['nll'])<TOL)
                rec=dict(label='background_alternative_Powell',index=index,start=start,bounds=[[-2.,2.],[0.,3.]],success=success,returned_success=bool(result.success),status=int(result.status),message=str(result.message),pars=np.asarray(result.x).tolist(),nll=float(result.fun),nit=int(result.nit),nfev=int(result.nfev),candidate=point,elapsed_seconds=time.perf_counter()-tick)
                if success:alt_candidates.append(point)
            except TimeoutError:raise
            except Exception as exc:rec=dict(label='background_alternative_Powell',index=index,start=start,success=False,error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-tick)
            search_records.append(clean(rec));emit(dict(event='search_stage_completed',stage='background_alternative_Powell',record=rec))
        # Explicit eta-profile searches. Their successful eligible points are
        # also eligible full alternative candidates, not hidden extra winners.
        curves=[]
        for index,eta in enumerate(np.linspace(0.,3.,11)):
            if eta==0:point=null;eligible=nu_candidates;stage=dict(label='eta_profile',eta=0.,reused_null=True);runs=[]
            else:
                runs=[];point,eligible,stage=scalar_search(lambda d,eta=float(eta):objective.evaluate(d,eta),d_grid,[old_null[1],old_alt[1]],label='eta_profile_'+str(index),deadline=search_deadline,records=runs)
                search_records.extend(runs)
            alt_candidates.extend(eligible);curve=dict(eta=float(eta),selected=point,search=stage,solver_runs=runs);curves.append(curve)
            emit(dict(event='eta_profile_completed',index=index,record=curve))
        alt=_best(alt_candidates);bg=dict(id='background_estimated',null=null,alternative=alt,candidates=alt_candidates,null_candidates=nu_candidates,eta_profile=curves,complete=False)
        methods.append(bg)
        # Pointwise controls use the old backend only at four old, four new,
        # and four profiled corner points, never at every search evaluation.
        points=[]
        for mid in METHODS:
            for phase in ('null','alternative'):points.append(('old_'+mid+'_'+phase,old_points[mid][phase]));points.append(('new_'+mid+'_'+phase,next(m for m in methods if m['id']==mid)[phase]))
        for d in (-2.,2.):
            for eta in (0.,3.):points.append(('profile_corner_'+str(d)+'_'+str(eta),objective.evaluate(d,eta)))
        for label,p in points:
            value=history.objective(p['pars'],law.trigger,law.ratio,False);delta=abs(value-p['nll']);controls.append(dict(label=label,pars=p['pars'],profile_value=p['nll'],old_backend_value=value,absolute_difference=delta,passed=delta<TOL))
        for m in methods:
            m['old_selected']=old_points[m['id']]
            m['contrast16']=history.contrast(m['null'],m['alternative']);m['old_contrast16']=history.contrast(old_points[m['id']]['null'],old_points[m['id']]['alternative'])
        out['cuda_metrics']=context.cuda_metrics(32768);out['scalar_reduction_chunk_nodes']=SEARCH_SPEC['scalar_reduction_chunk_nodes']
        out['feature_metadata']=dict(nodes16=history.nodes,stress_modes=len(law.stress.mixture[0]),clock_mode=history.mode,preparation_seconds=history.preparation_seconds)
        history.close();context.torch.cuda.empty_cache()
        independent=RestrictedHistory(arrays,law,clock_mode=clock_mode,order=32,backend='cpu',deadline=deadline)
        validation=[];old32controls=[]
        for m in methods:
            mid=m['id'];old32={}
            for phase in ('null','alternative'):
                point=m[phase];v=independent.objective(point['pars'],law.trigger,law.ratio,False);delta=abs(v-point['nll'])
                validation.append(dict(method=mid,phase=phase,profile16_nll=point['nll'],independent32_nll=v,absolute_difference=delta,passed=delta<TOL))
                op=old_points[mid][phase];old32[phase]=independent.objective(op['pars'],law.trigger,law.ratio,False)
                old32controls.append(dict(method=mid,phase=phase,profile16_nll=op['nll'],independent32_nll=old32[phase],saved_primary16_nll=old[mid][phase]['nll'],absolute_difference=abs(old32[phase]-op['nll']),saved_absolute_difference=abs(old32[phase]-old[mid][phase]['nll']),passed=abs(old32[phase]-op['nll'])<TOL and abs(old32[phase]-old[mid][phase]['nll'])<TOL))
            m['contrast32']=independent.contrast(m['null'],m['alternative']);m['old_contrast32']=independent.contrast(old_points[mid]['null'],old_points[mid]['alternative'])
            m['gains']=dict(null_NLL=old_points[mid]['null']['nll']-m['null']['nll'],alternative_NLL=old_points[mid]['alternative']['nll']-m['alternative']['nll'],direct_GLR_shift16=m['contrast16']['statistic']-m['old_contrast16']['statistic'],direct_GLR_shift32=m['contrast32']['statistic']-m['old_contrast32']['statistic'])
            m['old_rescored32_nll']=old32
            own_runs=kk_runs if mid=='known_known' else [r for r in search_records if r['label']!='known_known_eta']
            m['gates']=dict(selected_values_passed=all(g['passed'] for g in validation if g['method']==mid),old_values_passed=all(g['passed'] for g in old32controls if g['method']==mid),
                            direct_order_delta=abs(m['contrast32']['statistic']-m['contrast16']['statistic']),old_direct_order_delta=abs(m['old_contrast32']['statistic']-m['old_contrast16']['statistic']),old_direct_saved_delta=abs(m['old_contrast16']['statistic']-old[mid]['contrast16']['statistic']),
                            old_points_not_worsened=m['gains']['null_NLL']>=-1e-8 and m['gains']['alternative_NLL']>=-1e-8,
                            direct_controls_passed=all(abs(c['direct_minus_subtraction'])<TOL for c in (m['contrast16'],m['contrast32'],m['old_contrast16'],m['old_contrast32'])),
                            null_nested=m['contrast16']['statistic']>=-TOL and m['contrast32']['statistic']>=-TOL,all_mandated_solvers_successful=all(r['success'] for r in own_runs))
            m['complete']=bool(m['gates']['selected_values_passed'] and m['gates']['old_values_passed'] and m['gates']['direct_order_delta']<TOL and m['gates']['old_direct_order_delta']<TOL and m['gates']['old_direct_saved_delta']<TOL and m['gates']['old_points_not_worsened'] and m['gates']['direct_controls_passed'] and m['gates']['null_nested'] and m['gates']['all_mandated_solvers_successful'])
        searches_ok=all(r['success'] for r in search_records)
        complete=all(m['complete'] for m in methods) and searches_ok and all(c['passed'] for c in controls+old_controls+old32controls)
        out.update(complete=bool(complete),status='complete' if complete else 'unresolved_search_or_validation',all_mandated_solvers_successful=searches_ok,validation=validation,
                   old32_controls=old32controls,independent32_feature_seconds=independent.preparation_seconds,nodes32=independent.nodes,validation_reserve_seconds=reserve,search_deadline=search_deadline)
        emit(dict(event='validation_completed',complete=out['complete'],validation=validation))
    except Exception as exc:out.update(complete=False,status='time_cap' if isinstance(exc,TimeoutError) else 'unresolved',error=f'{type(exc).__name__}: {exc}')
    finally:
        if objective is not None:out.update(evaluations=objective.evaluations,fixed_evaluations=objective.fixed_evaluations,evaluation_errors=objective.errors,background_integral_evaluations=objective.J_evaluations,profile_requests=objective.profile_requests,profile_cache_hits=objective.cache_hits)
        for h in (history,independent):
            if h is not None:h.close()
        out['elapsed_seconds']=time.perf_counter()-began
    return clean(out)


def validate_result(payload,protocol):
    """Pure complete-payload guard. No history, objective or solver is called."""
    def require(ok,message):
        if not ok:raise ValueError(message)
    def close(a,b,tol=1e-10):return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol
    def point(p):
        x=p['pars'];require(len(x)==4 and all(math.isfinite(v) for v in x) and -12<=x[0]<=0 and -2<=x[1]<=2 and 0<=x[3]<=3 and math.isfinite(p['nll']),'Nonfinite/out-of-box point')
        require(x[2]==n and p['ratio']==fit['ratio'] and p['trigger']==fit['trigger'],'Fixed trigger changed')
    def same_point(a,b):return all(a[k]==b[k] for k in ('pars','nll','trigger','ratio','origin'))
    def contrast(c):
        require(all(math.isfinite(c[k]) for k in ('statistic','nll_subtraction_control','direct_minus_subtraction')),'Nonfinite contrast')
        require(close(c['direct_minus_subtraction'],c['statistic']-c['nll_subtraction_control'],1e-12) and abs(c['direct_minus_subtraction'])<TOL,'Direct/subtraction control failed')
    require(protocol['search']==clean(SEARCH_SPEC),'Search protocol differs')
    require(payload.get('complete') is True and payload.get('status')=='complete','Not a complete diagnostic')
    require([m['id'] for m in payload['methods']]==list(METHODS),'Two procedures required')
    fit=protocol['law']['alternative_generating_fit'];n=fit['pars'][2]
    runs=payload['search_records'];stages=payload['stages'];methods={m['id']:m for m in payload['methods']}
    require(len(stages)==2 and [x['label'] for x in stages]==['known_known_eta','background_null'],'Mandatory grid stages missing')
    expected_runs=0
    for stage,size in zip(stages,(17,17)):
        require(stage['grid_points']==size and stage['solver_runs']==len(stage['local_brackets']),'Scalar grid/brackets missing')
        own=[r for r in runs if r['label']==stage['label']]
        require(len(own)==stage['solver_runs'] and [r['bounds'] for r in own]==stage['local_brackets'],'Scalar run coverage differs')
        expected_runs+=stage['solver_runs']
    bg=methods['background_estimated'];kk=methods['known_known']
    require(len(kk['candidates'])>=21 and len(bg['candidates'])>=51 and len(bg['null_candidates'])>=19,'Required candidate grids missing')
    kkgrid=kk['candidates'][2:19]
    require([p['pars'][3] for p in kkgrid]==list(np.linspace(0.,3.,17)),'KK grid differs')
    grid=bg['candidates'][2:51]
    require([[p['pars'][1],p['pars'][3]] for p in grid]==[[float(d),float(e)] for d in np.linspace(-2.,2.,7) for e in np.linspace(0.,3.,7)],'BG 7x7 grid differs')
    oldalt=bg['old_selected']['alternative']['pars'];oldnull=bg['old_selected']['null']['pars']
    seeds=[p['pars'][1::2] for p in sorted(grid,key=lambda p:p['nll'])[:3]]+[[oldalt[1],oldalt[3]],[oldnull[1],0.]];unique=[]
    for seed in seeds:
        if seed not in unique:unique.append(seed)
    powell=[r for r in runs if r['label']=='background_alternative_Powell']
    require([r['start'] for r in powell]==unique,'Prescribed Powell starts missing/changed');expected_runs+=len(unique)
    curves=bg['eta_profile'];require(len(curves)==11 and [c['eta'] for c in curves]==list(np.linspace(0.,3.,11)),'All eleven eta profile points required')
    for i,curve in enumerate(curves):
        point(curve['selected']);require(curve['selected']['pars'][3]==curve['eta'],'Eta profile coordinate differs')
        if i==0:require(curve['search'].get('reused_null') is True and not curve['solver_runs'],'Eta0 must reuse null')
        else:
            st=curve['search'];own=curve['solver_runs'];require(st['grid_points']==17 and len(own)==st['solver_runs']==len(st['local_brackets']),'Profile d grid/runs missing')
            require([r['bounds'] for r in own]==st['local_brackets'],'Profile brackets differ')
            flat=[r for r in runs if r['label']=='eta_profile_'+str(i)];require(flat==own,'Profile runs differ from global trace');expected_runs+=len(own)
    require(len(runs)==expected_runs and payload['all_mandated_solvers_successful'] is True and all(r.get('success') is True for r in runs),'Mandatory solver trace incomplete')
    for r in runs:
        point(r['candidate']);require(close(r['nll'],r['candidate']['nll'],TOL) and r['nfev']>0,'Returned candidate differs')
        if r['label']=='background_alternative_Powell':require(r['nfev']<=100000 and r['nit']<=2000 and r['bounds']==[[-2.,2.],[0.,3.]],'Powell budget/box differs')
        else:require(r['nit']<=100,'Scalar budget differs')
    require(len(payload['controls'])==12 and len(payload['old_primary_controls'])==4 and len(payload['old32_controls'])==4 and len(payload['validation'])==4,'Required 12+4+4+4 value controls missing')
    labels=['old_'+m+'_'+p for m in METHODS for p in ('null','alternative')]+['new_'+m+'_'+p for m in METHODS for p in ('null','alternative')]+['profile_corner_'+str(d)+'_'+str(e) for d in (-2.,2.) for e in (0.,3.)]
    controlmap={c['label']:c for c in payload['controls']}
    require(len(controlmap)==12 and set(controlmap)==set(labels),'Pointwise control labels missing/duplicated')
    for c in payload['controls']:
        delta=abs(c['profile_value']-c['old_backend_value']);require(c['passed'] is True and delta<TOL and close(delta,c['absolute_difference'],1e-12),'Pointwise old backend control failed')
        require(len(c['pars'])==4 and all(math.isfinite(v) for v in c['pars']) and -12<=c['pars'][0]<=0 and -2<=c['pars'][1]<=2 and c['pars'][2]==n and 0<=c['pars'][3]<=3,'Control point outside law/box')
    for mid,m in methods.items():
        for phase in ('null','alternative'):
            for tag,target in (('old',m['old_selected'][phase]),('new',m[phase])):
                c=controlmap[tag+'_'+mid+'_'+phase];require(c['pars']==target['pars'] and c['profile_value']==target['nll'],'Control detached from selected point')
    for d in (-2.,2.):
        for e in (0.,3.):require(controlmap['profile_corner_'+str(d)+'_'+str(e)]['pars'][1::2]==[d,e],'Profile corner coordinate differs')
    oldmap={}
    for c in payload['old_primary_controls']:
        delta=abs(c['profile16_nll']-c['saved_primary16_nll']);require(c['passed'] is True and delta<TOL and close(delta,c['absolute_difference'],1e-12),'Saved old point reproduction failed');oldmap[(c['method'],c['phase'])]=c
        require(c['profile16_nll']==methods[c['method']]['old_selected'][c['phase']]['nll'],'Old primary control detached from point')
    require(set(oldmap)=={(m,p) for m in METHODS for p in ('null','alternative')},'Old phase coverage differs')
    for name in ('validation','old32_controls'):
        rows=payload[name];require({(r['method'],r['phase']) for r in rows}==set(oldmap),'CPU32 phase coverage differs')
        for c in rows:
            delta=abs(c['profile16_nll']-c['independent32_nll']);require(c['passed'] is True and delta<TOL and close(delta,c['absolute_difference'],1e-12),'CPU32 point control failed')
            target=methods[c['method']]['old_selected'][c['phase']] if name=='old32_controls' else methods[c['method']][c['phase']]
            require(c['profile16_nll']==target['nll'],'CPU32 control detached from point')
            if name=='old32_controls':require(abs(c['independent32_nll']-c['saved_primary16_nll'])<TOL,'Old CPU32 saved-point reproduction failed')
    for mid,m in methods.items():
        require(m['complete'] is True,'Method incomplete')
        for phase in ('null','alternative'):
            point(m[phase]);point(m['old_selected'][phase]);require(m[phase]['nll']<=m['old_selected'][phase]['nll']+1e-8,'Old eligible point worsened')
        require(m['null']['pars'][3]==0. and m['old_selected']['null']['pars'][3]==0.,'Null eta changed')
        for p in m['candidates']:
            point(p)
            if mid=='known_known':require(p['pars'][:2]==fit['pars'][:2],'Known background changed in candidate')
        if mid=='known_known':
            require(m['null']['pars'][:2]==fit['pars'][:2] and m['alternative']['pars'][:2]==fit['pars'][:2],'Known background was estimated')
        else:
            for p in m['null_candidates']:point(p);require(p['pars'][3]==0.,'Background null candidate eta changed')
            require(same_point(m['null'],_best(m['null_candidates'])),'Null winner/tie selection differs')
        require(same_point(m['alternative'],_best(m['candidates'])),'Alternative winner/tie selection differs')
        for key in ('contrast16','contrast32','old_contrast16','old_contrast32'):contrast(m[key])
        for tag,key16,key32,rowname in (('new','contrast16','contrast32','validation'),('old','old_contrast16','old_contrast32','old32_controls')):
            cn=controlmap[tag+'_'+mid+'_null'];ca=controlmap[tag+'_'+mid+'_alternative'];rows={c['phase']:c for c in payload[rowname] if c['method']==mid}
            require(close(m[key16]['nll_subtraction_control'],2*(cn['old_backend_value']-ca['old_backend_value']),1e-8),'Order16 contrast detached from original-backend point values')
            require(close(m[key32]['nll_subtraction_control'],2*(rows['null']['independent32_nll']-rows['alternative']['independent32_nll']),1e-8),'Order32 contrast detached from independent point values')
        require(abs(m['contrast16']['statistic']-m['contrast32']['statistic'])<TOL and abs(m['old_contrast16']['statistic']-m['old_contrast32']['statistic'])<TOL,'Direct order gate failed')
        require(min(m['contrast16']['statistic'],m['contrast32']['statistic'])>=-TOL,'Matching null nesting failed')
        for phase in ('null','alternative'):require(close(m['gains'][phase+'_NLL'],m['old_selected'][phase]['nll']-m[phase]['nll']),'NLL gain arithmetic differs')
        for order in (16,32):require(close(m['gains']['direct_GLR_shift'+str(order)],m['contrast'+str(order)]['statistic']-m['old_contrast'+str(order)]['statistic']),'Direct shift arithmetic differs')
        require(m['gates']['old_direct_saved_delta']<TOL,'Saved original GLR reproduction failed')
    return True
