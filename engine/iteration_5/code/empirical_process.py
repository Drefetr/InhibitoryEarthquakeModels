"""Predictable diagnostic likelihood and bounded grid estimator.

Independent reconstruction; iteration_4 supplied numerical lessons, not code
or data reproduction. All histories exclude the current event. Time is days.
"""
from __future__ import annotations
import time
from dataclasses import asdict
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize
from kernels import KernelSpec

GAMMA=.75*np.log(10.)
G6=float(np.exp(GAMMA))


def stress_marks(excess):
    return np.exp(GAMMA*np.asarray(excess))


def unit_productivity(excess,beta,ratio):
    if beta<=0 or not 0<=ratio<1:
        raise ValueError('beta>0 and productivity ratio in [0,1) required')
    return (1-ratio)*np.exp(beta*ratio*np.asarray(excess))


def event_states(times,amplitudes,kernel):
    """Pre-event modal state; tied times are rejected by History."""
    weights,rates=kernel.mixture
    result=np.empty((len(times),len(rates)))
    state=np.zeros(len(rates));old=0.
    for j,(t,amplitude) in enumerate(zip(times,amplitudes)):
        state*=np.exp(-rates*(t-old));result[j]=state
        state+=amplitude;old=t
    return result


def direct_history(times,amplitudes,query,kernel,derivative=False):
    """Independent literal O(NQ) histories, used only for validation."""
    result=[]
    for t in query:
        keep=np.asarray(times)<t
        value=kernel.h(t-np.asarray(times)[keep]) if derivative else kernel.phi(t-np.asarray(times)[keep])
        result.append(np.dot(np.asarray(amplitudes)[keep],value))
    return np.asarray(result)


class EmpiricalHistory:
    def __init__(self,times,excess,beta,end,coordinate_horizon=24394.,start=0.,order=16,
                 memory_limit_mb=800.):
        times=np.asarray(times,dtype=float);excess=np.asarray(excess,dtype=float)
        if times.shape!=excess.shape or np.any(np.diff(times)<=0) or np.any(times<0):
            raise ValueError('strictly ordered nonnegative event times required')
        if np.any(excess<0) or not 0<=start<end or coordinate_horizon<=0:
            raise ValueError('invalid marks/window/horizon')
        keep=times<end
        self.times=times[keep];self.excess=excess[keep]
        self.end=float(end);self.start=float(start);self.beta=float(beta)
        self.coordinate_horizon=float(coordinate_horizon);self.order=int(order)
        self.event_mask=self.times>=start
        self.xe=2*self.times[self.event_mask]/coordinate_horizon-1
        self.stress={};self.trigger={}
        self.memory_limit_mb=memory_limit_mb
        z,w=leggauss(order)
        # Geometric offsets plus geometric midpoints, including fast postshock
        # recovery and gaps extending over several decades. No trigger numerical
        # integral is required: its compensator is analytic.
        offsets=np.geomspace(1e-6,1e5,23)
        offsets=np.sort(np.r_[offsets,np.sqrt(offsets[:-1]*offsets[1:])])
        left=np.r_[0.,self.times]
        boundaries=np.r_[start,self.times[(self.times>start)&(self.times<end)],end]
        qa=[];qb=[];qi=[]
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            edges=np.r_[a,a+offsets[offsets<b-a],b]
            qa.extend(edges[:-1]);qb.extend(edges[1:])
            qi.extend([np.searchsorted(self.times,a,side='right')]*(len(edges)-1))
        qa=np.asarray(qa);qb=np.asarray(qb)
        self.tq=((qa[:,None]+qb[:,None])/2+(qb-qa)[:,None]*z/2).ravel()
        self.wq=((qb-qa)[:,None]*w/2).ravel()
        self.qindex=np.repeat(np.asarray(qi,dtype=int),order)
        self.qlag=self.tq-left[self.qindex]
        self.xq=2*self.tq/coordinate_horizon-1
        estimated=(24*len(self.times)**2+len(self.tq)*8*35)/(1024**2)
        if estimated>memory_limit_mb:
            raise MemoryError(f'estimated peak {estimated:.1f}MiB exceeds {memory_limit_mb}MiB')

    def stress_features(self,kernel):
        if kernel.key in self.stress:return self.stress[kernel.key]
        gm=stress_marks(self.excess)/G6
        if kernel.kind=='omori':
            ge=direct_history(self.times,gm,self.times,kernel)[self.event_mask]
            gq=direct_history(self.times,gm,self.tq,kernel)
        else:
            weights,rates=kernel.mixture
            pre=event_states(self.times,gm,kernel)
            post=np.vstack([np.zeros(len(rates)),pre+gm[:,None]])
            ge=(pre@weights)[self.event_mask]
            gq=np.empty(len(self.tq))
            for a in range(0,len(gq),2048):
                q=slice(a,a+2048)
                gq[q]=(post[self.qindex[q]]*np.exp(-self.qlag[q,None]*rates))@weights
        self.stress[kernel.key]=(ge,gq)
        return ge,gq

    def trigger_features(self,kernel,ratio):
        key=(kernel.key,float(ratio))
        if key in self.trigger:return self.trigger[key]
        kp=unit_productivity(self.excess,self.beta,ratio)
        if kernel.kind=='omori':
            he=direct_history(self.times,kp,self.times,kernel,True)[self.event_mask]
        else:
            weights,rates=kernel.mixture
            he=(event_states(self.times,kp,kernel)@(weights*rates))[self.event_mask]
        upper=kernel.cdf(self.end-self.times)
        lower=np.where(self.times<self.start,kernel.cdf(np.maximum(self.start-self.times,0.)),0.)
        integral=float(np.dot(kp,upper-lower))
        self.trigger[key]=(he,integral)
        return he,integral

    def objective(self,pars,trigger,ratio,stress=None,gradient=False):
        a,d,n=np.asarray(pars)[:3]
        eta=float(pars[3]) if stress is not None else 0.
        he,integral=self.trigger_features(trigger,ratio)
        if stress is None:
            ge=0.;gq=0.
        else:ge,gq=self.stress_features(stress)
        logbe=a+d*self.xe-eta*ge
        logbq=a+d*self.xq-eta*gq
        bq=np.exp(logbq)
        loghe=np.full_like(he,-np.inf)
        np.log(he,out=loghe,where=he>0)
        logtrigger=loghe+np.log(n) if n>0 else np.full_like(he,-np.inf)
        loglam=np.logaddexp(logbe,logtrigger)
        value=float(-loglam.sum()+np.dot(self.wq,bq)+n*integral)
        if not np.isfinite(value):raise FloatingPointError('nonfinite likelihood')
        if not gradient:return value
        eb=np.exp(logbe-loglam)
        with np.errstate(over='ignore',invalid='ignore'):
            trigger_ratio=np.exp(loghe-loglam)
        if not np.all(np.isfinite(trigger_ratio)):
            raise FloatingPointError('unrepresentable n-boundary derivative')
        wb=self.wq*bq
        grad=[-eb.sum()+wb.sum(),-np.dot(eb,self.xe)+np.dot(wb,self.xq),-trigger_ratio.sum()+integral]
        if stress is not None:grad.append(np.dot(eb,ge)-np.dot(wb,gq))
        return value,np.asarray(grad)


def default_candidates():
    exp=[KernelSpec('exp',t) for t in (.1,1.,10.)]
    ml=[KernelSpec('ml',t,alpha=a) for a in (.5,.7) for t in (.1,1.,10.)]
    om=[KernelSpec('omori',t,p=p) for p in (1.1,1.3,1.7) for t in (.1,1.,10.)]
    return exp,ml,om


def fit_candidates(history,maxiter=100,ratios=(.4,.7,.8),deadline=None,include_omori=True):
    """Complete finite grid with two prescribed starts and explicit boundaries.

    Every optimization is retained. A resource cap returns the unfinished search
    with `complete=False`; no partial grid is silently called its full estimator.
    """
    started=time.perf_counter();records=[];best={};nulls={};complete=True;skipped=[]
    exp,ml,om=default_candidates()
    a0=float(np.clip(np.log(max(history.event_mask.sum(),1)/(history.end-history.start)*.6),-11.5,-.1))
    bounds=[(-12.,0.),(-2.,2.),(0.,.98)]
    def one(model,kernel,ratio,stress=None,base=None,origin='optimizer'):
        nonlocal complete
        if deadline is not None and time.perf_counter()>deadline:
            complete=False;return None
        starts=[np.array([a0,0.,.35]),np.array([a0-.5,.1,.7])] if base is None else [np.asarray(base[:3])]
        if stress is not None:starts=[np.r_[starts[0],0.],np.r_[starts[0],.2]]
        chosen=None
        for start in starts:
            st=time.perf_counter()
            rec={'model':model,'trigger':asdict(kernel),'ratio':float(ratio),'stress':asdict(stress) if stress else None,
                 'start':start.tolist(),'origin':origin}
            try:
                fit=minimize(lambda q:history.objective(q,kernel,ratio,stress,True),start,jac=True,
                    method='L-BFGS-B',bounds=bounds+([(0.,3.)] if stress else []),
                    options={'maxiter':maxiter,'ftol':1e-10,'gtol':2e-6,'maxls':40})
                rec.update(pars=fit.x.tolist(),nll=float(fit.fun),success=bool(fit.success),message=str(fit.message),
                           nit=int(fit.nit),nfev=int(fit.nfev),status=int(fit.status))
            except (FloatingPointError,ValueError,OverflowError) as exc:
                rec.update(pars=start.tolist(),nll=None,success=False,message=f'{type(exc).__name__}: {exc}',nit=0,nfev=0,status=-1)
            rec['elapsed_seconds']=time.perf_counter()-st;records.append(rec)
            if rec['success'] and (chosen is None or rec['nll']<chosen['nll']):chosen=rec
        if chosen is not None and (model not in best or chosen['nll']<best[model]['nll']):best[model]=chosen
        return chosen
    for kind,kernels in [('exp',exp),('ml_null',ml),('omori',om if include_omori else [])]:
        for k in kernels:
            for ratio in ratios:
                rec=one(kind,k,ratio)
                if kind=='ml_null' and rec is not None:nulls[(k.key,ratio)]=rec
    for k in ml:
        for ratio in ratios:
            base=nulls.get((k.key,ratio))
            if base is None:
                skipped.append({'trigger':asdict(k),'ratio':ratio,'reason':'no_converged_matching_null'})
                continue
            for multiplier in (1.,.1,10.):
                stress=KernelSpec('ml',k.tau*multiplier,alpha=k.alpha)
                model='ml_shared' if multiplier==1 else 'ml_separate_scale'
                rec=one(model,k,ratio,stress,base['pars'])
                # Exact eta=0 candidate is retained independently of optimizer.
                boundary=dict(base,model=model,stress=asdict(stress),pars=base['pars']+[0.],origin='null_boundary',start=None)
                records.append(boundary)
                if model not in best or boundary['nll']<best[model]['nll']:best[model]=boundary
                if multiplier==1:
                    shared=dict(best['ml_shared'],model='ml_separate_scale',origin='shared_subset')
                    records.append(shared)
                    if 'ml_separate_scale' not in best or shared['nll']<best['ml_separate_scale']['nll']:best['ml_separate_scale']=shared
    return {'best':best,'subfits':records,'complete':complete and not skipped,
            'resource_grid_complete':complete,'skipped_alternative_grids':skipped,
            'all_optimizations_converged':all(r['success'] for r in records if r['origin']=='optimizer'),
            'elapsed_seconds':time.perf_counter()-started,
            'grid':{'tau':[.1,1.,10.],'alpha':[.5,.7],'omori_p':[1.1,1.3,1.7],
                    'productivity_ratio':list(ratios),'stress_scale_multiplier':[.1,1.,10.]},
            'bounds':{'a':[-12,0],'d':[-2,2],'n':[0,.98],'eta':[0,3]},'maxiter':maxiter}


def score_fit(history,record):
    trigger=KernelSpec(**record['trigger'])
    stress=KernelSpec(**record['stress']) if record.get('stress') else None
    return -history.objective(record['pars'],trigger,record['ratio'],stress)


def simulate_null(seed,horizon,beta,pars,trigger,ratio,coordinate_horizon=24394.,max_events=20000,
                  max_proposals=1000000,seconds=30.):
    """Finite-mixture dominating thinning, fresh marks after acceptance.

    Independent implementation of the iteration_4 dominating-envelope argument.
    Supports fitted ML/exp null export. Resource-censored prefixes are retained.
    """
    if trigger.kind=='omori':raise ValueError('null simulator supports finite mixture exp/ML only')
    rng=np.random.default_rng(seed);a,d,n=pars[:3]
    weights,rates=trigger.mixture;state=np.zeros(len(rates))
    upper_background=np.exp(max(a-d,a+d*(2*horizon/coordinate_horizon-1)))
    times=[];marks=[];t=0.;proposals=0;started=time.perf_counter();reason=''
    while t<horizon:
        if len(times)>=max_events or proposals>=max_proposals or time.perf_counter()-started>seconds:
            reason='event_cap' if len(times)>=max_events else 'proposal_cap' if proposals>=max_proposals else 'time_cap';break
        upper=upper_background+state@weights
        dt=rng.exponential(1/upper)
        if t+dt>=horizon:t=horizon;break
        t+=dt;proposals+=1;state*=np.exp(-rates*dt)
        intensity=np.exp(a+d*(2*t/coordinate_horizon-1))+state@weights
        if intensity>upper*(1+1e-12):raise AssertionError('invalid thinning envelope')
        if rng.uniform()*upper<intensity:
            q=rng.exponential(1/beta);times.append(t);marks.append(q)
            state+=rates*n*unit_productivity(q,beta,ratio)
    return {'times':np.asarray(times),'excess':np.asarray(marks),'complete':not reason,
            'reason':reason,'proposals':proposals,'elapsed_seconds':time.perf_counter()-started}
