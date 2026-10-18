"""Same thinning/RNG law with an explicit precise accepted-event clock.

Legacy FP64 Kahan projection/background and modal decay arithmetic are retained.
Exact Decimal sums of the sampled binary64 waiting times supply ordering/gaps;
accepted display ties are never jittered or rejected just for display precision.
Float-only likelihood code must not consume tied projected times.
"""
from __future__ import annotations
from decimal import Decimal, localcontext
import argparse,hashlib,json,math,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'iteration_5/code'))
from empirical_process import G6,GAMMA,unit_productivity
from kernels import KernelSpec

ARRAY_KEYS=('times','times_lo','event_gaps','excess','exact_time_strings')
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def prefix_digest(arrays):
    value=hashlib.sha256()
    for key in ARRAY_KEYS:
        a=np.asarray(arrays[key]);value.update(key.encode());value.update(str(a.dtype).encode());value.update(str(a.shape).encode());value.update(a.tobytes())
    return value.hexdigest()
def precise_lag(high_a,low_a,high_b,low_b):return math.fsum([float(high_a),-float(high_b),float(low_a),-float(low_b)])
def _pack(times,lows,gaps,marks,exact):
    return dict(times=np.asarray(times,dtype=np.float64),times_lo=np.asarray(lows,dtype=np.float64),
        event_gaps=np.asarray(gaps,dtype=np.float64),excess=np.asarray(marks,dtype=np.float64),exact_time_strings=np.asarray(exact,dtype=str))

def simulate_fitted(seed,horizon,beta,fit,max_events=20000,max_proposals=1000000,seconds=30.,
                    resume=None,stop_on_float_tie=False,rng_factory=None):
    """Return arrays + exact restart state; resume={'state':..., 'arrays':...}.

    Caps count total events/proposals, including the saved prefix. A diagnostic
    stop_on_float_tie ends before drawing the pending event's fresh mark; restart
    finishes that same accepted proposal using the saved RNG state, without a
    new exponential/uniform. Uniform repaired production uses False.
    """
    if not np.all(np.isfinite([horizon,beta,seconds])) or min(horizon,beta,seconds)<=0:raise ValueError('Finite positive horizon/beta/budget required')
    if any(not isinstance(x,(int,np.integer)) or isinstance(x,bool) or x<=0 for x in [max_events,max_proposals]):raise ValueError('Positive integer caps required')
    trigger=KernelSpec(**fit['trigger']);stress=KernelSpec(**fit['stress']) if fit.get('stress') else None
    if trigger.kind not in ('exp','ml') or stress and stress.kind not in ('exp','ml'):raise ValueError('Finite positive mixtures required')
    pars=fit['pars'];a,d,n=pars[:3];eta=pars[3] if stress is not None else 0.
    if not np.all(np.isfinite(pars)) or not np.isfinite(fit['ratio']) or not 0<=fit['ratio']<1 or not 0<=n<1 or eta<0:raise ValueError('Invalid generating parameters')
    fingerprint=canonical(dict(seed=int(seed),horizon=float(horizon),beta=float(beta),fit=fit))
    ew,er=trigger.mixture;sw,sr=stress.mixture if stress else (np.array([1.]),np.array([1.]))
    excitation=np.zeros(len(er));depletion=np.zeros(len(sr));upper_background=np.exp(a+abs(d))
    rng=(rng_factory or np.random.default_rng)(seed)
    times=[];lows=[];gaps=[];marks=[];exact=[];proposals=0;t=0.;correction=0.;clock=Decimal(0);last_event=Decimal(0)
    min_slack=float('inf');pending=None;ties=[];tie_count=0
    if resume is not None:
        state=resume['state'];arrays=resume['arrays']
        payload=dict(state);seal=payload.pop('state_sha256',None)
        if canonical(payload)!=seal:raise ValueError('Restart state seal differs')
        if state['source_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise ValueError('Restart simulator source differs')
        if state['schema_version']!=1 or state['input_sha256']!=fingerprint or state['complete']:raise ValueError('Wrong/non-resumable simulation state')
        if prefix_digest(arrays)!=state['prefix_sha256']:raise ValueError('Restart prefix differs')
        times=np.asarray(arrays['times'],dtype=float).tolist();lows=np.asarray(arrays['times_lo'],dtype=float).tolist()
        gaps=np.asarray(arrays['event_gaps'],dtype=float).tolist();marks=np.asarray(arrays['excess'],dtype=float).tolist();exact=np.asarray(arrays['exact_time_strings'],dtype=str).tolist()
        if any(len(z)!=state['event_count'] for z in [times,lows,gaps,marks,exact]):raise ValueError('Restart event counts differ')
        t=state['legacy_t'];correction=state['legacy_correction'];clock=Decimal(state['exact_clock']);last_event=Decimal(state['last_event_clock'])
        proposals=state['proposals'];excitation=np.asarray(state['excitation'],dtype=float);depletion=np.asarray(state['depletion'],dtype=float)
        if excitation.shape!=er.shape or depletion.shape!=sr.shape:raise ValueError('Restart mode shape differs')
        rng.bit_generator.state=state['rng_state'];pending=state['pending_accept']
        if pending and 'consumed_mark' in pending:raise ValueError('Mark-transform overflow requires an explicit arithmetic repair, not a fresh mark')
        min_slack=state['min_envelope_slack'] if state['min_envelope_slack'] is not None else float('inf')
        ties=state['coarse_tie_examples'];tie_count=state['coarse_tie_count']
    started=time.perf_counter();reason=''
    with localcontext() as context:
        # Every finite binary64 number has at most1074 fractional decimal
        # places and309 integer digits.1600 digits exactly accommodate sums
        # in a finite binary64 horizon; this is not platform longdouble.
        context.prec=1600
        while t<horizon or pending is not None:
            if len(times)>=max_events or proposals>=max_proposals or time.perf_counter()-started>seconds:
                reason='event_cap' if len(times)>=max_events else 'proposal_cap' if proposals>=max_proposals else 'time_cap';break
            if pending is None:
                upper=upper_background+float(excitation@ew)
                if not np.isfinite(upper) or upper<=0:reason='nonfinite_or_nonpositive_envelope';break
                dt=float(rng.exponential(1/upper))
                if not np.isfinite(dt) or dt<=0:reason='unrepresentable_waiting_time';break
                adjusted=dt-correction;new=t+adjusted;correction=(new-t)-adjusted
                clock+=Decimal.from_float(dt)
                # Preserve the original horizon decision/arithmetic exactly.
                if new>=horizon:t=float(horizon);break
                t=new;proposals+=1
                excitation*=np.exp(-er*dt);depletion*=np.exp(-sr*dt)
                intensity=np.exp(a+d*(2*t/horizon-1)-eta*float(depletion@sw))+float(excitation@ew)
                slack=upper-intensity;min_slack=min(min_slack,slack)
                if intensity>upper*(1+1e-12):raise AssertionError('Thinning envelope violation')
                if rng.uniform()*upper>=intensity:continue
                pending=dict(dt=dt,upper=upper,intensity=float(intensity),proposal=proposals,
                    legacy_t=t,legacy_correction=correction,legacy_ulp=float(np.spacing(t)),
                    previous_legacy_time=times[-1] if times else None)
            low=float(clock-Decimal.from_float(t));gap=float(clock-last_event)
            if gap<=0 or not np.isfinite(gap):reason='precise_clock_not_increasing';break
            if times and precise_lag(t,low,times[-1],lows[-1])<=0:reason='two_part_clock_resolution_exhausted';break
            coarse_tie=bool(times and t<=times[-1])
            if coarse_tie and not pending.get('tie_recorded'):
                tie_count+=1;pending['tie_recorded']=True
                if len(ties)<32:ties.append(dict(pending,exact_clock=str(clock),previous_exact_clock=str(last_event),event_gap=gap,times_lo=low))
            if coarse_tie and stop_on_float_tie:reason='unrepresentable_distinct_event_time';break
            q=float(rng.exponential(1/beta))
            gm=float(np.exp(GAMMA*q)/G6);kp=float(n*unit_productivity(q,beta,fit['ratio']))
            if not np.isfinite(gm+kp):
                pending['consumed_mark']=q;reason='mark_transform_overflow';break
            times.append(t);lows.append(low);gaps.append(gap);marks.append(q);exact.append(str(clock));last_event=clock
            excitation+=er*kp;depletion+=gm;pending=None
    arrays=_pack(times,lows,gaps,marks,exact)
    state=dict(schema_version=1,input_sha256=fingerprint,seed=int(seed),horizon=float(horizon),beta=float(beta),
        event_count=len(times),proposals=proposals,legacy_t=t,legacy_correction=correction,exact_clock=str(clock),last_event_clock=str(last_event),
        excitation=excitation.tolist(),depletion=depletion.tolist(),rng_state=rng.bit_generator.state,pending_accept=pending,
        prefix_sha256=prefix_digest(arrays),complete=not reason,reason=reason,min_envelope_slack=None if not np.isfinite(min_slack) else min_slack,
        coarse_tie_count=tie_count,coarse_tie_examples=ties,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numeric_contract='Legacy FP64 Kahan projection/background/modal dt arithmetic; exact Decimal sum of binary64 waits, with relative gaps and residuals; no jitter/resampling')
    state['state_sha256']=canonical(state)
    return dict(**arrays,checkpoint_state=state,complete=not reason,reason=reason,proposals=proposals,
        elapsed_seconds=time.perf_counter()-started,min_envelope_slack=state['min_envelope_slack'],coarse_tie_count=tie_count,
        clock='legacy projected times + exact Decimal accumulated binary64 waits; times_lo residual and positive relative event_gaps',
        requires_precise_likelihood=bool(np.any(np.diff(arrays['times'])<=0)))

def _save(folder,label,result):
    arrays={k:result[k] for k in ARRAY_KEYS};path=folder/(label+'.npz');np.savez_compressed(path,**arrays)
    (folder/(label+'_state.json')).write_text(json.dumps(result['checkpoint_state'],indent=2,allow_nan=False)+'\n',encoding='utf-8')
    metadata={k:v for k,v in result.items() if k not in ARRAY_KEYS and k!='checkpoint_state'}
    metadata.update(events=len(result['times']),arrays_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        minimum_event_gap=float(np.min(result['event_gaps'])) if len(result['event_gaps']) else None,
        legacy_timestamp_ties=int(np.count_nonzero(np.diff(result['times'])<=0)))
    (folder/(label+'_summary.json')).write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n',encoding='utf-8');return metadata

def probe():
    began=time.perf_counter();deadline=began+105.
    folder=ROOT/'iteration_12/results/simulation_probe'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True,exist_ok=False)
    base=ROOT/'iteration_11/results/full_bootstrap';plan=json.loads((base/'plan.json').read_text(encoding='utf-8'));config=plan['config']
    outputs={};source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name,diagnostic,limit in [('null_0841',True,20000),('null_1292',False,20000)]:
        case=next(c for c in config['draws'] if c['id']==name)
        first=simulate_fitted(case['seed'],config['horizon'],config['beta'],config['generating_fits']['null'],
            max_events=limit,max_proposals=5000000,seconds=min(20.,max(.01,deadline-time.perf_counter())),stop_on_float_tie=diagnostic)
        with np.load(base/'runs'/name/'catalogue.npz',allow_pickle=False) as old:
            matches=np.array_equal(first['times'],old['times']) and np.array_equal(first['excess'],old['excess'])
        if not matches:raise AssertionError('Saved original prefix differs: '+name)
        outputs[name+'_replay']=_save(folder,name+'_replay',first)
        extended=simulate_fitted(case['seed'],config['horizon'],config['beta'],config['generating_fits']['null'],
            max_events=100000,max_proposals=5000000,seconds=min(60. if name=='null_1292' else 25.,max(.01,deadline-time.perf_counter())),
            resume={'state':first['checkpoint_state'],'arrays':{k:first[k] for k in ARRAY_KEYS}})
        original_count=len(first['times'])
        assert np.array_equal(extended['times'][:original_count],first['times']) and np.array_equal(extended['excess'][:original_count],first['excess'])
        outputs[name+'_extended']=_save(folder,name+'_extended',extended)
        outputs[name+'_extended']['original_prefix_times_and_marks_exact']=True
        print(json.dumps({name:outputs[name+'_extended']}),flush=True)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='One bounded replay/continuation probe only; no fits; old files unchanged',
        code_sha256=source,plan_id=plan['plan_id'],outputs=outputs,elapsed_seconds=time.perf_counter()-began,
        sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [base/'plan.json',ROOT/'iteration_5/code/calibration_simulator.py',ROOT/'iteration_5/code/empirical_process.py',ROOT/'iteration_5/code/kernels.py']},
        original_prefix_sha256={name:hashlib.sha256((base/'runs'/name/'catalogue.npz').read_bytes()).hexdigest() for name in ['null_0841','null_1292']})
    (folder/'probe_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(str(folder),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');args=parser.parse_args()
    if args.probe:probe()
