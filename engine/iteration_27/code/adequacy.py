"""Observed predictive diagnostics: independent geometry and likelihood assembly.

Only the pinned iteration-5 finite-mixture kernel implementation is reused.
The observed catalogue was previously inspected: rolling windows are exploratory.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
KERNEL_PATH = ROOT / 'iteration_5/code/kernels.py'
KERNEL_SHA = '3561ef3471adf349caebebc6c2ca12881f2a3d9c7c7415d15453b5be20b978ae'
if hashlib.sha256(KERNEL_PATH.read_bytes()).hexdigest() != KERNEL_SHA:
    raise RuntimeError('pinned kernel source changed')
_spec = importlib.util.spec_from_file_location('_adequacy27_kernels', KERNEL_PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
KernelSpec = _module.KernelSpec
GAMMA = .75 * np.log(10.)
FAMILIES = ('poisson_constant', 'poisson_loglinear', 'etas_omori', 'ml_trigger', 'ml_shared')
BOUNDS = [(-12., 0.), (-2., 2.), (0., .98)]


class BudgetExceeded(RuntimeError):
    pass


def check_deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise BudgetExceeded('declared wall deadline reached')


def _bounded_solve(target, start, bounds, options, deadline, verify):
    """Uniform prospective failure-only recovery; failed returns never compete.

    A slot is complete only when one solver return reports success and passes
    finite/bounds/nonworsening/fresh-value checks. All preceding failures remain
    in its lineage. Recovery restarts from the original prescribed start.
    """
    attempts=[];initial=None
    recipes=[('L-BFGS-B',dict(options)),('L-BFGS-B',dict(options,maxls=200)),
             ('Powell',{'maxiter':2000,'maxfev':100000,'xtol':1e-6,'ftol':1e-10})]
    for index,(method,opts) in enumerate(recipes):
        began=time.perf_counter()
        row={'tier':index,'method':method,'options':opts,'start':list(start)}
        try:
            check_deadline(deadline)
            if initial is None:initial=float(target(np.asarray(start))[0])
            if method=='Powell':
                opt=minimize(lambda x:target(x)[0],start,method=method,bounds=bounds,options=opts)
            else:opt=minimize(target,start,jac=True,method=method,bounds=bounds,options=opts)
            row.update(pars=opt.x.tolist(),reported_nll=float(opt.fun),success=bool(opt.success),
                       solver_return_success=bool(opt.success),solver_message=str(opt.message),
                       solver_status=int(opt.status),message=str(opt.message),status=int(opt.status),nit=int(opt.nit),nfev=int(opt.nfev))
            feasible=bool(np.isfinite(opt.x).all() and all(lo<=x<=hi for x,(lo,hi) in zip(opt.x,bounds)))
            row['finite_feasible']=feasible
            fresh=float(target(opt.x)[0]) if feasible else None
            independent=float(verify(opt.x,index>0)) if feasible and opt.success else None
            row.update(fresh_nll=fresh,control_nll=independent,
                       control_scope='fresh order32 geometry for recovery; separate fixed-kernel route for continuous Omori',
                       initial_nll=initial)
            accepted=bool(opt.success and feasible and np.isfinite(opt.fun) and np.isfinite(fresh)
                and independent is not None and np.isfinite(independent)
                and abs(float(opt.fun)-fresh)<1e-4 and abs(fresh-independent)<1e-4
                and fresh<=initial+1e-8)
            row['accepted']=accepted
            row['elapsed_seconds']=time.perf_counter()-began;attempts.append(row)
            if accepted:return opt,attempts,initial,False
        except BudgetExceeded as exc:
            row.setdefault('success',False)
            row.update(accepted=False,message=str(exc),evaluation_error=str(exc),status=-2,
                       elapsed_seconds=time.perf_counter()-began);attempts.append(row)
            return None,attempts,initial,True
        except (FloatingPointError,ValueError,OverflowError) as exc:
            row.setdefault('success',False)
            row.update(accepted=False,message=f'{type(exc).__name__}: {exc}',evaluation_error=f'{type(exc).__name__}: {exc}',status=-1,
                       elapsed_seconds=time.perf_counter()-began);attempts.append(row)
    return None,attempts,initial,False


def background_integral(a, d, start, end, T0):
    """Analytic loglinear integral and derivative w.r.t. d, including d=0."""
    center = (start + end) / T0 - 1.
    half = (end - start) / T0
    z = d * half
    if abs(z) < 1e-4:
        sinhc = 1 + z*z/6 + z**4/120 + z**6/5040
        correction = half * (z/3-z**3/45+2*z**5/945)
    else:
        sinhc = np.sinh(z)/z
        correction = half * (1/np.tanh(z)-1/z)
    total = float(np.exp(a+d*center)*(end-start)*sinhc)
    return total, total*(center+correction)


def _modal_states(times, amplitude, rates):
    """Pre-event states; current event never contributes to its own intensity."""
    state = np.zeros(len(rates)); previous = 0.
    before = np.empty((len(times), len(rates)))
    for i, t in enumerate(times):
        state *= np.exp(-(t-previous)*rates)
        before[i] = state
        state += amplitude[i]
        previous = t
    return before


class PredictiveHistory:
    """Window [start,end), retaining all catalogue parents in [0,end).

    Nodes are constructed independently of EmpiricalHistory. Each event-free
    interval is subdivided at geometric recovery offsets. Trigger integrals use
    exact kernel CDF differences; the final event-free tail is always retained.
    """
    def __init__(self, times, excess, beta, start, end, T0=24394., order=16,
                 memory_limit_mb=512.):
        times = np.asarray(times, dtype=float); excess = np.asarray(excess, dtype=float)
        if times.ndim != 1 or times.shape != excess.shape or not np.isfinite(times).all() or not np.isfinite(excess).all():
            raise ValueError('finite matching vectors required')
        if np.any(np.diff(times) <= 0) or np.any(times < 0) or np.any(excess < 0):
            raise ValueError('strictly ordered nonnegative events/marks required')
        if not (np.isfinite(beta) and beta > 0 and 0 <= start < end and T0 > 0 and order in (16, 32)):
            raise ValueError('invalid window/beta/order')
        self.times = times[times < end]; self.excess = excess[times < end]
        self.beta, self.start, self.end, self.T0, self.order = float(beta), float(start), float(end), float(T0), int(order)
        self.memory_limit_mb = float(memory_limit_mb)
        self.mask = self.times >= start
        self.events = self.times[self.mask]
        self.xe = 2*self.events/T0-1
        self.edges = np.r_[start, self.events[self.events > start], end]
        # If an event lies exactly at start its residual is zero and is recorded;
        # geometrical intervals still have strictly positive lengths.
        self.event_interval = np.searchsorted(self.edges[1:], self.events, side='left')
        offsets = np.geomspace(1e-6, 1e5, 23)
        offsets = np.sort(np.r_[offsets, np.sqrt(offsets[:-1]*offsets[1:])])
        z, w = leggauss(order)
        nodes, weights, groups = [], [], []
        for j, (left, right) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            cuts = np.r_[left, left+offsets[offsets < right-left], right]
            half = np.diff(cuts)/2
            nodes.append(((cuts[:-1]+cuts[1:])[:, None]/2+half[:, None]*z).ravel())
            weights.append((half[:, None]*w).ravel())
            groups.extend([j] * (len(half)*order))
        self.tq = np.concatenate(nodes); self.wq = np.concatenate(weights)
        self.group = np.asarray(groups); self.xq = 2*self.tq/T0-1
        estimated = 8*(12*len(self.tq)+4*len(self.times)**2)/(1024**2)
        if estimated > memory_limit_mb:
            raise MemoryError('independent geometry memory limit')
        self._stress = {}; self._trigger = {}

    def stress(self, kernel):
        if kernel.key in self._stress:
            return self._stress[kernel.key]
        amplitude = np.exp(GAMMA*(self.excess-1.))
        weights, rates = kernel.mixture
        if (8*len(rates)*len(self.times)+8*2048*len(rates))/(1024**2) > self.memory_limit_mb:
            raise MemoryError('modal history memory limit')
        before = _modal_states(self.times, amplitude, rates)
        ge = (before @ weights)[self.mask]
        after = np.vstack([np.zeros(len(rates)), before+amplitude[:, None]])
        parent = np.searchsorted(self.times, self.tq, side='right')
        origin = np.r_[0., self.times][parent]
        gq = np.empty(len(self.tq))
        for j in range(0, len(gq), 2048):
            sl = slice(j, j+2048)
            gq[sl] = np.sum(after[parent[sl]]*np.exp(-(self.tq[sl]-origin[sl])[:, None]*rates)*weights, axis=1)
        self._stress[kernel.key] = ge, gq
        return ge, gq

    def trigger(self, kernel, ratio):
        key = kernel.key, float(ratio)
        if key in self._trigger:
            return self._trigger[key]
        if not 0 <= ratio < 1:
            raise ValueError('productivity ratio outside [0,1)')
        kp = (1-ratio)*np.exp(self.beta*ratio*self.excess)
        if kernel.kind == 'omori':
            h = np.zeros(len(self.times))
            for i, t in enumerate(self.times):
                h[i] = kp[:i] @ kernel.h(t-self.times[:i])
            he = h[self.mask]
        else:
            weights, rates = kernel.mixture
            he = (_modal_states(self.times, kp, rates) @ (weights*rates))[self.mask]
        # CDF at every endpoint includes exactly the previously observed parents.
        cumulative = np.array([kp[self.times < t] @ kernel.cdf(t-self.times[self.times < t]) for t in self.edges])
        pieces = np.diff(cumulative)
        self._trigger[key] = he, pieces
        return he, pieces

    def components(self, pars, trigger=None, ratio=0., stress=None):
        a, d = np.asarray(pars)[:2]
        n = float(pars[2]) if trigger is not None else 0.
        eta = float(pars[3]) if stress is not None else 0.
        ge, gq = self.stress(stress) if stress is not None else (np.zeros(len(self.events)), np.zeros(len(self.tq)))
        he, hi = self.trigger(trigger, ratio) if trigger is not None else (np.zeros(len(self.events)), np.zeros(len(self.edges)-1))
        logb = a+d*self.xe-eta*ge
        logh = np.full_like(he, -np.inf)
        np.log(he, out=logh, where=he > 0)
        loglam = np.logaddexp(logb, logh+np.log(n) if n > 0 else np.full_like(he, -np.inf))
        bq = np.exp(a+d*self.xq-eta*gq)
        if stress is None:
            pieces = np.array([background_integral(a, d, l, r, self.T0)[0] for l, r in zip(self.edges[:-1], self.edges[1:])])
        else:
            pieces = np.bincount(self.group, weights=self.wq*bq, minlength=len(self.edges)-1)
        pieces = pieces+n*hi
        return loglam, pieces, (ge, gq, he, hi, bq, logb, logh)

    def objective(self, pars, trigger=None, ratio=0., stress=None, gradient=False):
        loglam, pieces, aux = self.components(pars, trigger, ratio, stress)
        value = float(pieces.sum()-loglam.sum())
        if not math.isfinite(value):
            raise FloatingPointError('nonfinite likelihood')
        if not gradient:
            return value
        ge, gq, he, hi, bq, logb, logh = aux
        eb = np.exp(logb-loglam)
        if stress is None:
            B, Bd = background_integral(pars[0], pars[1], self.start, self.end, self.T0)
        else:
            B = self.wq @ bq; Bd = self.wq @ (bq*self.xq)
        grad = [-eb.sum()+B, -eb @ self.xe+Bd]
        if trigger is not None:
            grad.append(-np.exp(logh-loglam).sum()+hi.sum())
        if stress is not None:
            grad.append(eb @ ge-self.wq @ (bq*gq))
        if not np.isfinite(grad).all():
            raise FloatingPointError('nonfinite gradient')
        return value, np.asarray(grad)


def _record_components(record):
    trigger = KernelSpec(**record['trigger']) if record.get('trigger') else None
    stress = KernelSpec(**record['stress']) if record.get('stress') else None
    return trigger, record.get('ratio', 0.), stress


def omori_continuous_objective(history, pars, ratio, gradient=True):
    """Independent normalized Omori likelihood with c,p derivatives.

    Parameter vector is (a,d,n,log(c),p). No inhibitory quadrature is needed.
    """
    a,d,n,logc,p=pars; c=np.exp(logc)
    parents=history.times
    kp=(1-ratio)*np.exp(history.beta*ratio*history.excess)
    lag=history.events[:,None]-parents[None,:]
    prior=lag>0; positive=np.maximum(lag,0.)
    logu=np.log1p(positive/c)
    density=np.where(prior,(p-1)/c*np.exp(-p*logu),0.)
    he=density@kp
    hc=(density*(-1+p*positive/(positive+c)))@kp
    hp=(density*(1/(p-1)-logu))@kp
    def cumulative(t):
        u=np.maximum(t-parents,0.); v=np.log1p(u/c); tail=np.exp((1-p)*v)
        return (-np.expm1((1-p)*v),-tail*(p-1)*u/(u+c),tail*v)
    top=cumulative(history.end); bot=cumulative(history.start)
    I,Ic,Ip=[float(kp@(x-y)) for x,y in zip(top,bot)]
    logb=a+d*history.xe
    logh=np.full_like(he,-np.inf);np.log(he,out=logh,where=he>0)
    loglam=np.logaddexp(logb,logh+np.log(n) if n>0 else np.full_like(he,-np.inf))
    B,Bd=background_integral(a,d,history.start,history.end,history.T0)
    value=float(B+n*I-loglam.sum())
    if not gradient:return value
    inv=np.exp(-loglam);eb=np.exp(logb-loglam)
    deriv=np.array([B-eb.sum(),Bd-eb@history.xe,I-he@inv,n*(Ic-hc@inv),n*(Ip-hp@inv)])
    if not np.isfinite(value) or not np.isfinite(deriv).all():raise FloatingPointError('nonfinite continuous Omori')
    return value,deriv


def evaluate_window(times, excess, record, start, end, T0, beta, order=16):
    history = PredictiveHistory(times, excess, beta, start, end, T0, order)
    loglam, pieces, _ = history.components(record['pars'], *_record_components(record))
    n = len(history.events)
    mark = float(n*np.log(beta)-beta*history.excess[history.mask].sum())
    ground = float(loglam.sum()-pieces.sum())
    # Each event compensator is the integral since the previous observed event
    # or the window origin. An event exactly at the origin has zero increment.
    residuals = [0. if t == start else float(pieces[j]) for t, j in zip(history.events, history.event_interval)]
    tail = float(pieces[-1])
    return {'start':float(start), 'end':float(end), 'order':order, 'n_events':n,
            'ground_loglik':ground, 'common_mark_loglik':mark, 'full_marked_loglik':ground+mark,
            'event_log_intensities':loglam.tolist(), 'integrated_intensity':float(pieces.sum()),
            'event_compensator_increments':residuals, 'final_no_event_compensator':tail,
            'count_minus_compensator':float(n-pieces.sum()), 'final_tail_right_censored':True,
            'rescaled_uniform':(-np.expm1(-np.asarray(residuals))).tolist(),
            'quadrature_nodes':len(history.tq),
            'scope':'predictable conditional score; test events update history after occurrence; final tail retained'}


def selected_controls(times, excess, record, start, end, T0, beta, tolerance=1e-4):
    coarse = evaluate_window(times, excess, record, start, end, T0, beta, 16)
    fine = evaluate_window(times, excess, record, start, end, T0, beta, 32)
    refined = dict(record)
    for key in ('trigger', 'stress'):
        if record.get(key) and record[key]['kind'] == 'ml':
            refined[key] = asdict(replace(KernelSpec(**record[key]), spectral_step=.2, spectral_tail_bound=1e-10))
    spectral = evaluate_window(times, excess, refined, start, end, T0, beta, 32)
    delta = fine['ground_loglik']-coarse['ground_loglik']
    spectral_delta = spectral['ground_loglik']-fine['ground_loglik']
    return {'order16':coarse, 'order32':fine, 'spectral_refined32_ground_loglik':spectral['ground_loglik'],
            'quadrature_delta':delta, 'spectral_delta':spectral_delta,
            'pass':bool(abs(delta) < tolerance and abs(spectral_delta) < tolerance)}


def fit_training(times, excess, train_end, T0, protocol, deadline, progress_callback=None):
    """Finite prescribed search; every required optimization must succeed.

    No saved real-fit points are used. Failed/censored starts remain in records;
    selected fits never turn an unfinished grid into a complete estimator.
    """
    began = time.perf_counter(); times=np.asarray(times); excess=np.asarray(excess)
    train = times < train_end
    if not train.any() or excess[train].sum() <= 0:
        raise ValueError('training mark MLE requires positive total excess and events')
    beta = float(train.sum()/excess[train].sum())
    history = PredictiveHistory(times, excess, beta, 0., train_end, T0, 16,
                                protocol.get('memory_limit_mb',512.))
    fine_history=None
    def fine():
        nonlocal fine_history
        if fine_history is None:
            fine_history=PredictiveHistory(times,excess,beta,0.,train_end,T0,32,protocol.get('memory_limit_mb',512.))
        return fine_history
    records=[]; best={}; censored=False
    a0=float(np.clip(np.log(max(train.sum(),1)/train_end*.6),-11.5,-.1))
    options=protocol.get('optimizer', {'maxiter':150,'maxfun':20000,'ftol':1e-10,'gtol':2e-6,'maxls':40})
    def one(family, trigger, ratio, stress, start, origin='optimizer'):
        nonlocal censored
        rec={'id':len(records),'model':family,'trigger':asdict(trigger) if trigger else None,
             'stress':asdict(stress) if stress else None,'ratio':float(ratio),
             'start':list(start),'origin':origin}
        st=time.perf_counter()
        def target(x):
            check_deadline(deadline)
            return history.objective(x,trigger,ratio,stress,True)
        bounds=BOUNDS[:2] if trigger is None else BOUNDS+([(0.,3.)] if stress else [])
        try:
            def verify(x,refined):
                check_deadline(deadline)
                return (fine() if refined else history).objective(x,trigger,ratio,stress)
            opt,lineage,initial,hitcap=_bounded_solve(target,start,bounds,options,deadline,verify)
            rec.update(solver_attempts=lineage,initial_nll=initial,accepted_tier=next((r['tier'] for r in lineage if r['accepted']),None))
            if hitcap:censored=True
            if opt is None:
                last=lineage[-1]
                rec.update(pars=last.get('pars',list(start)),nll=last.get('reported_nll'),success=False,
                           message=last['message'],status=last['status'],nit=last.get('nit',0),nfev=last.get('nfev',0))
            else:
                rec.update(pars=opt.x.tolist(),nll=float(lineage[-1]['fresh_nll']),success=True,
                           message=str(opt.message),status=int(opt.status),nit=int(opt.nit),nfev=int(opt.nfev))
        except BudgetExceeded as exc:
            censored=True
            rec.update(pars=list(start),nll=None,success=False,message=str(exc),status=-2,nit=0,nfev=0)
        except (FloatingPointError,ValueError,OverflowError) as exc:
            rec.update(pars=list(start),nll=None,success=False,message=f'{type(exc).__name__}: {exc}',status=-1,nit=0,nfev=0)
        rec['elapsed_seconds']=time.perf_counter()-st;records.append(rec)
        if rec['success'] and (family not in best or rec['nll']<best[family]['nll']):best[family]=rec
        if progress_callback:progress_callback(rec)
        return rec
    # Constant Poisson is exact within the same declared a bounds, d fixed zero.
    a=float(np.clip(np.log(train.sum()/train_end),-12.,0.))
    const={'id':len(records),'model':'poisson_constant','trigger':None,'stress':None,'ratio':0.,
           'start':None,'origin':'analytic','pars':[a,0.], 'nll':history.objective([a,0.]),
           'success':True,'message':'bounded exact constant-rate MLE','status':0,'nit':0,'nfev':1,'elapsed_seconds':0.}
    records.append(const);best['poisson_constant']=const
    for start in ([a0,0.],[a0-.5,.1]):
        one('poisson_loglinear',None,0.,None,start)
        if censored:break
    ratios=protocol.get('ratios',[.4,.7,.8])
    ml=[KernelSpec('ml',tau,alpha=alpha) for alpha in protocol.get('ml_alpha',[.5,.7]) for tau in protocol.get('ml_tau',[.1,1.,10.])]
    omori=[KernelSpec('omori',tau,p=p) for p in protocol.get('omori_p',[1.1,1.3,1.7]) for tau in protocol.get('omori_tau',[.1,1.,10.])]
    matched=[]; coarse_omori={}
    for family,kernels in [('etas_omori',omori),('ml_trigger',ml)]:
        for kernel in kernels:
            for ratio in ratios:
                chosen=None
                for start in ([a0,0.,.35],[a0-.5,.1,.7]):
                    rec=one(family,kernel,ratio,None,start)
                    if rec['success'] and (chosen is None or rec['nll']<chosen['nll']):chosen=rec
                    if censored:break
                if family=='ml_trigger' and chosen is not None:matched.append((kernel,ratio,chosen))
                if family=='etas_omori' and chosen is not None and (ratio not in coarse_omori or chosen['nll']<coarse_omori[ratio]['nll']):coarse_omori[ratio]=chosen
                if censored:break
            if censored:break
        if censored:break
    if protocol.get('continuous_omori',True):
        for ratio in ratios:
            if censored:break
            if ratio not in coarse_omori:continue
            coarse=coarse_omori[ratio]
            floor=dict(coarse,id=len(records),origin='coarse_omori_floor',start=None,elapsed_seconds=0.)
            records.append(floor)
            c0=coarse['trigger']['tau'];p0=coarse['trigger']['p']
            starts=[coarse['pars']+[float(np.log(c0)),p0],
                    [a0,0.,.35,float(np.log(.01)),1.1],
                    [a0-.5,.1,.7,float(np.log(1.)),2.5]]
            for start in starts:
                rec={'id':len(records),'model':'etas_omori','trigger':None,'stress':None,'ratio':float(ratio),
                     'start':start,'origin':'continuous_omori_optimizer'}
                st=time.perf_counter()
                def target(x):
                    check_deadline(deadline)
                    return omori_continuous_objective(history,x,ratio)
                try:
                    def verify(x,refined):
                        check_deadline(deadline)
                        return (fine() if refined else history).objective(x[:3],KernelSpec('omori',float(np.exp(x[3])),p=float(x[4])),ratio)
                    opt,lineage,initial,hitcap=_bounded_solve(target,start,
                        BOUNDS+[(np.log(1e-5),np.log(10.)),(1.01,3.)],options,deadline,verify)
                    rec.update(solver_attempts=lineage,initial_nll=initial,accepted_tier=next((r['tier'] for r in lineage if r['accepted']),None))
                    if hitcap:censored=True
                    if opt is None:
                        last=lineage[-1];q=last.get('pars',start)
                        rec.update(pars=q[:3],optimizer_pars=q,nll=last.get('reported_nll'),success=False,
                            message=last['message'],status=last['status'],nit=last.get('nit',0),nfev=last.get('nfev',0))
                    else:
                        rec.update(pars=opt.x[:3].tolist(),optimizer_pars=opt.x.tolist(),
                            trigger=asdict(KernelSpec('omori',float(np.exp(opt.x[3])),p=float(opt.x[4]))),
                            nll=float(lineage[-1]['fresh_nll']),success=True,message=str(opt.message),
                            status=int(opt.status),nit=int(opt.nit),nfev=int(opt.nfev))
                except BudgetExceeded as exc:
                    censored=True
                    rec.update(pars=start[:3],optimizer_pars=start,nll=None,success=False,message=str(exc),status=-2,nit=0,nfev=0)
                except (FloatingPointError,ValueError,OverflowError) as exc:
                    rec.update(pars=start[:3],optimizer_pars=start,nll=None,success=False,message=f'{type(exc).__name__}: {exc}',status=-1,nit=0,nfev=0)
                rec['elapsed_seconds']=time.perf_counter()-st;records.append(rec)
                if rec['success'] and rec['nll']<best['etas_omori']['nll']:best['etas_omori']=rec
                if progress_callback:progress_callback(rec)
                if censored:break
    for kernel,ratio,null in matched:
        if censored:break
        floor=dict(null,id=len(records),model='ml_shared',stress=asdict(kernel),pars=null['pars']+[0.],
                   origin='matching_null_boundary',start=None,elapsed_seconds=0.)
        records.append(floor)
        if 'ml_shared' not in best or floor['nll']<best['ml_shared']['nll']:best['ml_shared']=floor
        for eta in (0.,.2):
            one('ml_shared',kernel,ratio,kernel,null['pars']+[eta])
            if censored:break
    expected=3+2*len(ratios)*(len(omori)+len(ml))+3*len(ml)*len(ratios)+(4*len(ratios) if protocol.get('continuous_omori',True) else 0)
    complete=(not censored and len(records)==expected and set(best)==set(FAMILIES)
              and all(r['success'] for r in records))
    return {'complete':bool(complete),'censored':censored,'status':'complete' if complete else 'incomplete',
            'beta_train':beta,'n_train':int(train.sum()),'train_end':float(train_end),'T0':float(T0),
            'best':best,'records':records,'search':protocol,'expected_records':expected,'elapsed_seconds':time.perf_counter()-began,
            'strict_required_start_acceptance':all(r['success'] for r in records),
            'all_required_start_slots_accepted':all(r['success'] for r in records),
            'all_primary_solver_calls_successful':all(r['solver_attempts'][0]['success'] for r in records if r['origin'] in ('optimizer','continuous_omori_optimizer')),
            'search_scope':'finite kernel/productivity grid and local optimization; no global optimum claim'}


def validate_fit(payload, search=None):
    """Pure operational guard; no likelihood calls or optimizer execution."""
    if payload.get('complete') is not True or payload.get('censored') is not False or payload.get('status')!='complete':
        raise ValueError('incomplete fit')
    if search is None:
        import json
        search=json.loads((ROOT/'iteration_27/configs/D_interface.json').read_text())['search']
    if payload.get('search')!=search:raise ValueError('search binding')
    ratios=search.get('ratios',[.4,.7,.8])
    ml=[KernelSpec('ml',t,alpha=v) for v in search.get('ml_alpha',[.5,.7]) for t in search.get('ml_tau',[.1,1.,10.])]
    omori=[KernelSpec('omori',t,p=v) for v in search.get('omori_p',[1.1,1.3,1.7]) for t in search.get('omori_tau',[.1,1.,10.])]
    expected=3+2*len(ratios)*(len(omori)+len(ml))+3*len(ml)*len(ratios)+(4*len(ratios) if search.get('continuous_omori',True) else 0)
    records=payload['records']
    if len(records)!=expected or payload['expected_records']!=expected or set(payload['best'])!=set(FAMILIES):
        raise ValueError('missing grid/model records')
    if [r['id'] for r in records]!=list(range(len(records))):raise ValueError('record identity')
    for r in records:
        if r.get('success') is not True or not np.isfinite(r['nll']) or not np.isfinite(r['pars']).all():raise ValueError('failed/nonfinite fit')
        bounds=BOUNDS[:2] if r['trigger'] is None else BOUNDS+([(0.,3.)] if r['stress'] else [])
        if len(r['pars'])!=len(bounds) or any(not lo<=x<=hi for x,(lo,hi) in zip(r['pars'],bounds)):raise ValueError('bounds')
        if r['model']=='poisson_constant' and r['pars'][1]!=0.:raise ValueError('constant trend changed')
        if not np.isfinite(r['elapsed_seconds']) or r['elapsed_seconds']<0 or r['status']!=0 or r['nit']<0 or r['nfev']<0:raise ValueError('solver metadata')
        if r['origin'] in ('optimizer','continuous_omori_optimizer'):
            lineage=r['solver_attempts']
            if not 1<=len(lineage)<=3 or r['accepted_tier']!=len(lineage)-1:raise ValueError('solver lineage')
            for tier,attempt in enumerate(lineage):
                expected_method='Powell' if tier==2 else 'L-BFGS-B'
                expected_options=({'maxiter':2000,'maxfev':100000,'xtol':1e-6,'ftol':1e-10} if tier==2
                    else dict(search.get('optimizer',{'maxiter':150,'maxfun':20000,'ftol':1e-10,'gtol':2e-6,'maxls':40}),**({'maxls':200} if tier==1 else {})))
                if attempt['tier']!=tier or attempt['method']!=expected_method or attempt['options']!=expected_options or attempt['start']!=r['start']:raise ValueError('recovery recipe')
                if attempt['accepted']!=(tier==len(lineage)-1):raise ValueError('failed return competing')
            accepted=lineage[-1]
            q=r.get('optimizer_pars',r['pars'])
            if not accepted['success'] or not accepted['finite_feasible'] or accepted['pars']!=q or accepted['fresh_nll']!=r['nll'] or accepted['initial_nll']!=r['initial_nll']:raise ValueError('accepted lineage join')
            if not np.isfinite([accepted['reported_nll'],accepted['fresh_nll'],accepted['control_nll'],accepted['initial_nll']]).all():raise ValueError('accepted value finite')
            if abs(accepted['reported_nll']-r['nll'])>=1e-4 or abs(accepted['control_nll']-r['nll'])>=1e-4 or r['nll']>r['initial_nll']+1e-8:raise ValueError('accepted value gates')
        if r['origin']=='continuous_omori_optimizer':
            q=r['optimizer_pars']
            if len(q)!=5 or q[:3]!=r['pars'] or not np.isfinite(q).all():raise ValueError('continuous coordinates')
            if not np.log(1e-5)<=q[3]<=np.log(10.) or not 1.01<=q[4]<=3.:raise ValueError('continuous bounds')
            if r['trigger']!=asdict(KernelSpec('omori',float(np.exp(q[3])),p=float(q[4]))):raise ValueError('continuous kernel join')
    # Reconstruct the complete prescribed record sequence without evaluating a
    # likelihood. This prevents a self-reported count from hiding skipped cells.
    if not (payload['n_train']>0 and payload['beta_train']>0 and payload['train_end']>0 and payload['T0']>0):raise ValueError('training metadata')
    a0=float(np.clip(np.log(payload['n_train']/payload['train_end']*.6),-11.5,-.1))
    def require(i,family,origin,kernel,ratio,stress,start):
        r=records[i]
        if (r['model'],r['origin'],r['trigger'],r['ratio'],r['stress'],r['start'])!=(family,origin,asdict(kernel) if kernel else None,float(ratio),asdict(stress) if stress else None,start):raise ValueError('prescribed search coverage/order')
        return r
    require(0,'poisson_constant','analytic',None,0.,None,None)
    for i,start in enumerate(([a0,0.],[a0-.5,.1]),1):require(i,'poisson_loglinear','optimizer',None,0.,None,start)
    i=3;matched=[];coarse={}
    for family,kernels in [('etas_omori',omori),('ml_trigger',ml)]:
        for k in kernels:
            for ratio in ratios:
                pair=[]
                for start in ([a0,0.,.35],[a0-.5,.1,.7]):
                    pair.append(require(i,family,'optimizer',k,ratio,None,start));i+=1
                winner=min(pair,key=lambda r:r['nll'])
                if family=='ml_trigger':matched.append((k,ratio,winner))
                elif ratio not in coarse or winner['nll']<coarse[ratio]['nll']:coarse[ratio]=winner
    if search.get('continuous_omori',True):
        for ratio in ratios:
            old=coarse[ratio];k=KernelSpec(**old['trigger'])
            floor=require(i,'etas_omori','coarse_omori_floor',k,ratio,None,None);i+=1
            if floor['pars']!=old['pars'] or floor['nll']!=old['nll']:raise ValueError('coarse floor')
            starts=[old['pars']+[float(np.log(k.tau)),k.p],[a0,0.,.35,float(np.log(.01)),1.1],[a0-.5,.1,.7,float(np.log(1.)),2.5]]
            for start in starts:
                r=records[i]
                if r['model']!='etas_omori' or r['origin']!='continuous_omori_optimizer' or r['ratio']!=ratio or r['stress'] is not None or r['start']!=start:raise ValueError('continuous start coverage')
                i+=1
    for k,ratio,old in matched:
        floor=require(i,'ml_shared','matching_null_boundary',k,ratio,k,None);i+=1
        if floor['pars']!=old['pars']+[0.] or floor['nll']!=old['nll']:raise ValueError('matching null floor')
        for eta in (0.,.2):require(i,'ml_shared','optimizer',k,ratio,k,old['pars']+[eta]);i+=1
    if i!=expected:raise ValueError('search terminal count')
    if payload.get('all_required_start_slots_accepted') is not True or payload.get('strict_required_start_acceptance') is not True:raise ValueError('slot acceptance metadata')
    if payload.get('all_primary_solver_calls_successful')!=all(r['solver_attempts'][0]['success'] for r in records if r['origin'] in ('optimizer','continuous_omori_optimizer')):raise ValueError('primary solver metadata')
    for family,r in payload['best'].items():
        if r!=records[r['id']] or r['model']!=family:raise ValueError('selected provenance')
        if r['nll']!=min(x['nll'] for x in records if x['model']==family):raise ValueError('selected minimum')
    if payload['best']['ml_shared']['nll']>payload['best']['ml_trigger']['nll']+1e-8:raise ValueError('nesting')
    return True
