"""Streaming, fixed-nuisance finite-mixture point-process log likelihood ratio.

No catalogue is simulated and no parameter is estimated here. Projected and
paired clocks are explicit observation/scoring conventions; neither is upgraded.
"""
from dataclasses import dataclass
from pathlib import Path
import math, sys, time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'iteration_5/code'))
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import exprel, logsumexp
from kernels import KernelSpec

PRIMARY_ETA=0.4885010300470942
GAMMA=.75*math.log(10.)
OFFSETS=np.geomspace(1e-6,1e5,23)
OFFSETS=np.sort(np.r_[OFFSETS,np.sqrt(OFFSETS[:-1]*OFFSETS[1:])])


class OracleDeadlineExceeded(TimeoutError):pass


@dataclass(frozen=True)
class KnownLaw:
    a:float
    d:float
    n:float
    beta:float
    ratio:float
    horizon:float
    trigger:KernelSpec
    stress:KernelSpec
    coordinate_horizon:float|None=None

    def __post_init__(self):
        values=(self.a,self.d,self.n,self.beta,self.ratio,self.horizon)
        if not all(math.isfinite(float(x)) for x in values) or not 0<=self.n<1 or not 0<=self.ratio<1 or self.beta<=0 or self.horizon<=0:
            raise ValueError('Invalid fixed generating law')
        if self.coordinate_horizon is not None and (not math.isfinite(self.coordinate_horizon) or self.coordinate_horizon<=0):raise ValueError('Invalid coordinate horizon')
        if self.trigger.kind not in ('exp','ml') or self.stress.kind not in ('exp','ml'):raise ValueError('Positive finite-mixture kernels required')

    @property
    def coordinate(self):return self.horizon if self.coordinate_horizon is None else self.coordinate_horizon


def law_from_setting(setting):
    f=setting['generating_fit'];a,d,n=f['pars'][:3]
    return KnownLaw(a,d,n,setting['beta'],f['ratio'],setting['horizon'],KernelSpec(**f['trigger']),KernelSpec(**f['stress']),setting.get('coordinate_horizon'))


def _difference(hi,lo,other_hi,other_lo):return math.fsum((float(hi),-float(other_hi),float(lo),-float(other_lo)))


def _clock(arrays,law,mode):
    t=np.asarray(arrays['times'],dtype=float);q=np.asarray(arrays['excess'],dtype=float)
    if t.ndim!=1 or t.shape!=q.shape or not np.isfinite(t).all() or not np.isfinite(q).all() or np.any(t<0) or np.any(q<0):raise ValueError('Invalid catalogue arrays')
    if mode in ('projected','original_projected_clock'):
        if np.any(np.diff(t)<=0) or len(t) and t[-1]>=law.horizon:raise ValueError('Projected clock must be strictly increasing below horizon')
        return t,q,None,np.diff(np.r_[0.,t,law.horizon]),'original_projected_clock'
    if mode not in ('paired','paired_clock'):raise ValueError('Explicit recognized clock_mode required')
    if 'times_lo' not in arrays or 'event_gaps' not in arrays:raise ValueError('Paired clock requires saved lows and authoritative gaps')
    lo=np.asarray(arrays['times_lo'],dtype=float);g=np.asarray(arrays['event_gaps'],dtype=float)
    if lo.shape!=t.shape or g.shape!=t.shape or not np.isfinite(lo).all() or not np.isfinite(g).all() or np.any(g<=0):raise ValueError('Invalid saved paired clock arrays')
    if not len(t):return t,q,lo,np.array([law.horizon]),'paired_clock'
    represented=np.array([_difference(t[i],lo[i],t[i-1] if i else 0.,lo[i-1] if i else 0.) for i in range(len(t))])
    if np.any(represented<=0) or any(abs(float(v))>4*math.ulp(float(h)) for h,v in zip(t,lo)):raise ValueError('Noncanonical paired timestamp or nonpositive represented gap')
    allowance=np.array([4*(math.ulp(float(v))+math.ulp(float(lo[i-1] if i else 0.))+math.ulp(float(represented[i]))+math.ulp(float(g[i]))) for i,v in enumerate(lo)])
    if np.any(abs(represented-g)>allowance):raise ValueError('Authoritative gaps inconsistent with paired clock')
    remaining=_difference(law.horizon,0.,t[-1],lo[-1])
    if remaining<=0:raise ValueError('Paired catalogue must end strictly before horizon')
    intervals=np.r_[g,remaining]
    if abs(math.fsum(intervals)-law.horizon)>math.fsum(allowance)+4*math.ulp(law.horizon):raise ValueError('Paired intervals do not cover horizon')
    return t,q,lo,intervals,'paired_clock'


def _deadline(deadline):
    if deadline is not None and time.perf_counter()>deadline:raise OracleDeadlineExceeded('Oracle evaluation deadline')


def evaluate_oracle(arrays,law,*,clock_mode,eta=PRIMARY_ETA,order=32,method='gauss',deadline=None,chunk_nodes=512,
                    adaptive_epsabs=1e-10,adaptive_epsrel=1e-10):
    """Return signed ell(eta)-ell(0), with identical fixed beta and nuisances.

    Auxiliary NLL0/NLL1 include analytic trigger and baseline compensators. The
    primary LR is accumulated directly and never formed by subtracting NLLs.
    Working arrays are at most chunk_nodes x stress_modes, never N x N.
    """
    began=time.perf_counter();_deadline(deadline)
    if not math.isfinite(eta) or eta<0 or int(order)!=order or order<=0 or int(chunk_nodes)!=chunk_nodes or chunk_nodes<=0:raise ValueError('Invalid eta/order/chunk')
    if method not in ('gauss','adaptive'):raise ValueError('Unknown quadrature method')
    t,q,lo,intervals,mode=_clock(arrays,law,clock_mode);paired=mode=='paired_clock'
    sw,sr=law.stress.mixture;ew,er=law.trigger.mixture
    z,w=leggauss(int(order));s=np.zeros(len(sr));e=np.zeros(len(er))
    logwr=np.log(ew)+np.log(er)
    event0=[];event_ratio=[];reductions=[];trigger_integrals=[];adaptive_errors=[]
    nodes=0;panels=0;max_working_nodes=0

    def interval_integral(j,gap):
        nonlocal nodes,panels,max_working_nodes
        if gap<=0 or eta==0 or not np.any(s):return 0.
        a=float(t[j-1]) if j else 0.;low=float(lo[j-1]) if paired and j else 0.
        offsets=OFFSETS[OFFSETS<gap]
        edges=np.r_[0.,offsets,gap] if paired else np.r_[a,a+offsets,a+gap]
        panels+=len(edges)-1
        if method=='adaptive':
            terms=[]
            def value(rel):
                _deadline(deadline)
                g=float(np.dot(s*np.exp(-sr*rel),sw))
                absolute=a+(low+rel) if paired else a+rel
                return math.exp(law.a+law.d*(2*absolute/law.coordinate-1))*(-math.expm1(-eta*g))
            rel_edges=edges if paired else edges-a
            for left,right in zip(rel_edges[:-1],rel_edges[1:]):
                v,err=quad(value,float(left),float(right),epsabs=adaptive_epsabs/max(1,len(intervals)*(len(edges)-1)),epsrel=adaptive_epsrel,limit=100)
                terms.append(v);adaptive_errors.append(err)
            return math.fsum(terms)
        # Use the original projected absolute-panel arithmetic, or the paired
        # implementation's relative nodes/weights and display-only background.
        query=((edges[:-1,None]+edges[1:,None])/2+(edges[1:,None]-edges[:-1,None])*z/2).ravel()
        weight=((edges[1:,None]-edges[:-1,None])*w/2).ravel()
        relative=query if paired else query-a
        absolute=a+(low+query) if paired else query
        terms=[];nodes+=len(query);max_working_nodes=max(max_working_nodes,min(len(query),chunk_nodes))
        for first in range(0,len(query),chunk_nodes):
            _deadline(deadline);sl=slice(first,first+chunk_nodes)
            g=(np.exp(-relative[sl,None]*sr)*s)@sw
            with np.errstate(over='ignore'):
                reduction=-np.expm1(-eta*g)
            terms.append(float(np.dot(weight[sl],np.exp(law.a+law.d*(2*absolute[sl]/law.coordinate-1))*reduction)))
        return math.fsum(terms)

    for j,gap in enumerate(intervals):
        _deadline(deadline);reductions.append(interval_integral(j,float(gap)))
        if j==len(t):break
        s*=np.exp(-sr*gap);e*=np.exp(-er*gap)
        ge=float(np.dot(s,sw))
        logtrigger=-math.inf
        if law.n>0 and np.any(e):
            loge=np.full_like(e,-np.inf);np.log(e,out=loge,where=e>0)
            logtrigger=math.log(law.n)+float(logsumexp(loge+logwr))
        absolute=(t[j]+lo[j]) if paired else t[j]
        logb=law.a+law.d*(2*absolute/law.coordinate-1)
        l0=float(np.logaddexp(logb,logtrigger));event0.append(l0)
        if eta==0 or ge==0:lr=0.
        else:
            # Normalized log components avoid subtracting two large LL totals;
            # if triggering vanishes this gives exactly -eta*G, even deeply.
            with np.errstate(over='ignore'):
                penalty=float(eta*ge)
            lr=float(np.logaddexp((logb-l0)-penalty,logtrigger-l0))
        if not math.isfinite(lr):raise FloatingPointError('Unrepresentable event log ratio')
        event_ratio.append(lr)
        try:gm=math.exp(GAMMA*float(q[j]))/math.exp(GAMMA);kp=(1-law.ratio)*math.exp(law.beta*law.ratio*float(q[j]))
        except OverflowError as exc:raise FloatingPointError('Unrepresentable transformed mark; no clipping') from exc
        s+=gm;e+=kp
        if not np.isfinite(s).all() or not np.isfinite(e).all():raise FloatingPointError('Unrepresentable modal state; no clipping')
        duration=_difference(law.horizon,0.,t[j],lo[j]) if paired else law.horizon-t[j]
        trigger_integrals.append(float(kp*np.dot(ew,-np.expm1(-er*duration))))
    baseline=math.exp(law.a-law.d)*law.horizon*float(exprel(2*law.d*law.horizon/law.coordinate))
    integral_trigger=math.fsum(trigger_integrals);c=math.fsum(reductions);events=math.fsum(event_ratio)
    log_lr=math.fsum((events,c));ll0=math.fsum(event0)-baseline-law.n*integral_trigger
    ll1=math.fsum((math.fsum(event0),events,-baseline,c,-law.n*integral_trigger))
    if eta==0:log_lr=events=c=0.;ll1=ll0
    values=(log_lr,events,c,ll0,ll1,baseline,integral_trigger)
    if not all(math.isfinite(x) for x in values):raise FloatingPointError('Nonfinite known-law likelihood result')
    return dict(log_lr=log_lr,event_log_ratio_sum=events,compensator_reduction=c,nll0=-ll0,nll1=-ll1,
        analytic_baseline_compensator=baseline,unit_trigger_compensator=integral_trigger,n_events=len(t),eta=float(eta),
        beta_fixed=law.beta,clock_mode=mode,order=int(order),method=method,quadrature_nodes=nodes,panels=panels,
        adaptive_error_sum=math.fsum(adaptive_errors),elapsed_seconds=time.perf_counter()-began,
        maximum_working_query_nodes=max_working_nodes,stress_modes=len(sr),trigger_modes=len(er),
        mark_density_cancellation='Same fixed iid exponential beta and support under both simple laws; no beta refit',
        statistic='Signed full-window ground log likelihood ratio, including compensator and random count')


def score_refined(arrays,law,*,clock_mode,eta=PRIMARY_ETA,tolerance=1e-6,deadline=None,chunk_nodes=512):
    """Frozen 16/32 gate; 64 only on failure, otherwise retain unresolved status."""
    if tolerance<=0 or not math.isfinite(tolerance):raise ValueError('Positive finite refinement tolerance')
    values={}
    for order in (16,32):values[str(order)]=evaluate_oracle(arrays,law,clock_mode=clock_mode,eta=eta,order=order,deadline=deadline,chunk_nodes=chunk_nodes)
    delta=abs(values['32']['log_lr']-values['16']['log_lr']);selected=32;initial_delta=delta
    if delta>tolerance:
        values['64']=evaluate_oracle(arrays,law,clock_mode=clock_mode,eta=eta,order=64,deadline=deadline,chunk_nodes=chunk_nodes)
        delta=abs(values['64']['log_lr']-values['32']['log_lr']);selected=64
    return dict(complete=bool(delta<=tolerance),refinement_passed=bool(delta<=tolerance),tolerance=tolerance,
        initial_16_32_difference=initial_delta,selected_refinement_difference=delta,selected_order=selected,
        result=values[str(selected)],evaluations=values)
