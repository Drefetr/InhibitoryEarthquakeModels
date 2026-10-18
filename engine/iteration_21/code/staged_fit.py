"""Six-call, method-validated staging of the frozen iteration-20 estimator.

The numerical implementation is imported unchanged. This module controls only
the two retained methods, their validation order, and reusable stage payloads.
It performs no file writes, CUDA initialization, or saved-data work on import.
"""
from dataclasses import asdict
from pathlib import Path
import argparse, copy, hashlib, importlib.util, json, math, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PINNED_SOURCES = {
    'iteration_20/code/restricted_fit.py': '29663cb80701a90332b0dfe0876f5b2a64a6d0841b6c9de66f80aade8beba418',
    'iteration_20/code/restricted_history.py': '0d3e2b538a2757343424ab13cb891d605225451e5db728739d0723624b0691f5',
}
for _path, _sha in PINNED_SOURCES.items():
    if hashlib.sha256((ROOT / _path).read_bytes()).hexdigest() != _sha:
        raise ValueError('Frozen numerical source changed: ' + _path)
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'iteration_20/code'))
import restricted_fit as original
if Path(original.__file__).resolve() != (ROOT / 'iteration_20/code/restricted_fit.py').resolve():
    raise ValueError('Unexpected restricted_fit module authority')
RestrictedHistory = original.RestrictedHistory
KernelSpec = original.KernelSpec
CudaOracleContext = original.CudaOracleContext
law_from_setting = original.law_from_setting
fit_start = original.fit_start
cpu = original.cpu
clean = original.clean
METHODS = ('known_known', 'background_estimated')
COUNTS = dict(known_known=2, background_estimated=4)
SCHEMA = 1
TOLERANCE = 1e-4


def canonical_hash(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_pins():
    return dict(PINNED_SOURCES, **{'iteration_21/code/staged_fit.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


def input_binding(arrays, law, clock_mode, contract):
    """Bind every provided array, including precision arrays unused in projected mode."""
    array_pins = {}
    for name in sorted(arrays):
        a = np.asarray(arrays[name])
        if a.dtype.hasobject:
            raise ValueError('Object arrays are not a checkpoint authority')
        array_pins[name] = dict(dtype=a.dtype.str, shape=list(a.shape), sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    return dict(schema_version=SCHEMA, arrays=array_pins, law=asdict(law), clock_mode=clock_mode,
                contract_sha256=canonical_hash(contract), sources=source_pins())


def seal(payload):
    out = clean(copy.deepcopy(payload))
    out.pop('checkpoint_sha256', None)
    out['checkpoint_sha256'] = canonical_hash(out)
    return out


def verify_checkpoint(payload, binding, kind, method):
    if not isinstance(payload, dict):
        raise ValueError('Checkpoint must be a mapping')
    p = copy.deepcopy(payload)
    digest = p.pop('checkpoint_sha256', None)
    if digest is None or canonical_hash(p) != digest:
        raise ValueError('Checkpoint content digest differs')
    if p.get('binding_sha256') != canonical_hash(binding) or p.get('checkpoint_kind') != kind or p.get('id') != method:
        raise ValueError('Checkpoint source/input/stage identity differs')
    return copy.deepcopy(payload)


def _finite_contrast(c):
    return (all(isinstance(c.get(k),(int,float)) and math.isfinite(c[k]) for k in ('statistic','nll_subtraction_control','direct_minus_subtraction')) and
            abs(c['direct_minus_subtraction']-(c['statistic']-c['nll_subtraction_control']))<=1e-12)


def validate_optimized_record(record, contract, binding=None):
    """Metadata-only deep cache guard; no history construction or numerical calls."""
    binding=record.get('input_binding') if binding is None else binding
    if binding is None or binding.get('contract_sha256')!=canonical_hash(contract) or binding.get('sources')!=source_pins():raise ValueError('Optimization authority differs')
    method=record.get('id')
    if method not in METHODS:raise ValueError('Unknown checkpoint method')
    p=verify_checkpoint(record,binding,'optimized_method',method); l=binding['law']; known=[l['a'],l['d'],l['n'],0.]
    if p.get('input_binding')!=binding or p.get('complete') is not False or p.get('optimization_complete') is not True or p.get('status')!='optimized_unvalidated':raise ValueError('Optimization artifact is not pending validation')
    sub=p.get('subfits',[])
    if len(sub)!=COUNTS[method]:raise ValueError('Incomplete prescribed start set')
    bg=method==METHODS[1]; a0=float(np.clip(math.log(max(binding['arrays']['times']['shape'][0]/l['horizon'],1e-8)),-12,0))
    phases=['null','null','alternative','alternative'] if bg else ['alternative','alternative']
    si_sequence=[0,1,0,1] if bg else [0,1]
    null=min(sub[:2],key=lambda r:r['nll']) if bg else p['null']
    if not bg and (null.get('pars')!=known or null.get('origin')!='fixed_null' or null.get('phase')!='null'):raise ValueError('Fixed null differs')
    for r,phase,si in zip(sub,phases,si_sequence):
        free=([0,1] if bg else [])+([3] if phase=='alternative' else [])
        start=known.copy()
        if phase=='null':start[:2]=[a0,0.] if si==0 else [float(np.clip(a0-.5,-12,0)),.1]
        else:start=list(null['pars']);start[3]=(0.,.2)[si]
        if r.get('success') is not True or r.get('origin')!='optimizer' or r.get('method')!=method or r.get('phase')!=phase or r.get('cell_index')!=0 or r.get('start_index')!=si or r.get('free_indices')!=free or r.get('bounds')!=original.BOUNDS[free].tolist() or r.get('start')!=start:raise ValueError('Prescribed start/box/lineage identity differs')
        if r.get('trigger')!=l['trigger'] or r.get('ratio')!=l['ratio']:raise ValueError('Trigger was changed')
        pars=np.asarray(r.get('pars'),dtype=float)
        if pars.shape!=(4,) or not np.isfinite(pars).all() or np.any(pars<original.BOUNDS[:,0]) or np.any(pars>original.BOUNDS[:,1]) or not isinstance(r.get('nll'),(int,float)) or not math.isfinite(r['nll']):raise ValueError('Optimizer return is not finite/feasible')
        if any(pars[i]!=start[i] for i in range(4) if i not in free):raise ValueError('Fixed coordinate was estimated')
        tier=r.get('accepted_tier')
        if tier not in ('primary_cuda40','retry_cpu40','retry_cpu200','retry_cpu_Powell') or not any(a.get('tier')==tier and a.get('success') is True for a in r.get('attempts',[])):raise ValueError('Successful accepted solver lineage missing')
    if canonical_hash(null)!=canonical_hash(p['null']):raise ValueError('Selected null/tie rule differs')
    if len(p.get('cells',[]))!=1:raise ValueError('Exactly one trigger cell required')
    cell=p['cells'][0];subset=cell['exact_null_subset'];expected_pars=list(null['pars']);expected_pars[3]=0.
    if subset.get('pars')!=expected_pars or subset.get('origin')!='exact_matching_null_subset' or subset.get('nll')!=null['nll']:raise ValueError('Exact eligible null subset differs')
    alt=min([subset]+sub[-2:],key=lambda r:r['nll'])
    if canonical_hash(alt)!=canonical_hash(p['alternative']) or canonical_hash(cell['null'])!=canonical_hash(null) or canonical_hash(cell['alternative'])!=canonical_hash(alt):raise ValueError('Selected alternative/tie rule differs')
    if cell.get('index')!=0 or cell.get('trigger')!=l['trigger'] or cell.get('ratio')!=l['ratio'] or not _finite_contrast(p.get('contrast16',{})):raise ValueError('Cell/contrast metadata differs')
    if abs(p['contrast16']['nll_subtraction_control']-2*(null['nll']-alt['nll']))>1e-8:raise ValueError('NLL contrast control differs')
    ws=p.get('witnesses',[])
    if len(ws)!=(2 if bg else 1) or any(not isinstance(w.get('difference'),(int,float)) or not math.isfinite(w['difference']) for w in ws):raise ValueError('Witnesses missing/nonfinite')
    for w,phase in zip(ws,('null','alternative') if bg else ('alternative',)):
        selected=p[phase]
        if w.get('selected_nll')!=selected['nll'] or not isinstance(w.get('witness_nll'),(int,float)) or not math.isfinite(w['witness_nll']) or w['difference']!=selected['nll']-w['witness_nll']:raise ValueError('Witness arithmetic differs')
        if bg:
            if w.get('kind')!='cross_procedure_subset' or w.get('phase')!=phase or w.get('broader')!=method or w.get('narrower')!=METHODS[0] or w.get('broader_cell')!=0 or w.get('narrower_cell')!=0:raise ValueError('Cross-method witness identity differs')
        else:
            wp=w.get('witness_point',{});fc=w.get('contrast16',{})
            if w.get('kind')!='fixed_eta_star' or w.get('method')!=method or wp.get('pars')!=known[:3]+[contract['law']['alternative_eta']] or wp.get('nll')!=w['witness_nll'] or wp.get('trigger')!=l['trigger'] or wp.get('ratio')!=l['ratio'] or not _finite_contrast(fc):raise ValueError('Fixed eta witness differs')
            if w.get('selected_GLR16')!=p['contrast16']['statistic'] or w.get('direct_GLR_shortfall')!=fc['statistic']-p['contrast16']['statistic'] or abs(fc['nll_subtraction_control']-2*(null['nll']-w['witness_nll']))>1e-8:raise ValueError('Fixed eta direct witness arithmetic differs')
    return copy.deepcopy(p)


def validate_method_record(record, contract, binding=None, known_known=None):
    """Metadata-only completed-method guard; independent numerical work is not replayed."""
    binding=record.get('input_binding') if binding is None else binding; method=record.get('id')
    if binding is None or method not in METHODS:raise ValueError('Invalid completed method authority')
    p=verify_checkpoint(record,binding,'validated_method',method)
    opt=validate_optimized_record(p['optimized_checkpoint'],contract,binding)
    if p.get('input_binding')!=binding or p.get('complete') is not True or p.get('status')!='complete' or p.get('optimized_checkpoint_sha256')!=opt['checkpoint_sha256']:raise ValueError('Validated method identity differs')
    for key in ('subfits','cells','null','alternative','contrast16'):
        if canonical_hash(p.get(key))!=canonical_hash(opt[key]):raise ValueError('Validated selection differs from optimizer checkpoint')
    gs=p.get('selected_objectives',[])
    if len(gs)!=2 or [g.get('phase') for g in gs]!=['null','alternative']:raise ValueError('Two selected CPU32 objectives required')
    for g,phase in zip(gs,('null','alternative')):
        if g.get('method')!=method or g.get('cuda16_nll')!=opt[phase]['nll'] or not all(isinstance(g.get(k),(int,float)) and math.isfinite(g[k]) for k in ('cuda16_nll','cpu32_nll','absolute_difference')):raise ValueError('Invalid CPU32 validation evidence')
        delta=abs(g['cuda16_nll']-g['cpu32_nll'])
        if g.get('passed') is not True or delta>=TOLERANCE or g['absolute_difference']!=delta:raise ValueError('Selected CPU32 gate failed')
    c16=opt['contrast16'];c32=p.get('contrast32',{})
    if not _finite_contrast(c32) or abs(c32['nll_subtraction_control']-2*(gs[0]['cpu32_nll']-gs[1]['cpu32_nll']))>1e-8:raise ValueError('CPU32 direct contrast control differs')
    delta=abs(c16['statistic']-c32['statistic']);gates=p.get('gates',{})
    if gates.get('passed') is not True or gates.get('statistic_16_32_difference')!=delta or delta>=TOLERANCE or abs(c16['direct_minus_subtraction'])>=TOLERANCE or abs(c32['direct_minus_subtraction'])>=TOLERANCE or min(c16['statistic'],c32['statistic']) < -TOLERANCE:raise ValueError('Direct GLR validation failed')
    ws=p.get('witnesses',[])
    if len(ws)!=(1 if method==METHODS[0] else 2):raise ValueError('Completed witness count differs')
    for w in ws:
        if w.get('passed') is not True or not all(isinstance(w.get(k,0.),(int,float)) and math.isfinite(w.get(k,0.)) and w.get(k,0.)<=TOLERANCE for k in ('difference','direct_GLR_shortfall','direct_GLR32_shortfall')):raise ValueError('Witness gate failed')
    for w,ow in zip(ws,opt['witnesses']):
        for k in ow:
            if canonical_hash(w.get(k))!=canonical_hash(ow[k]):raise ValueError('Validated witness differs from optimization')
    if method==METHODS[0]:
        w=ws[0]
        if not _finite_contrast(w.get('contrast32',{})) or w['kind']!='fixed_eta_star' or w.get('selected_GLR32')!=c32['statistic'] or w.get('direct_GLR32_shortfall')!=w['contrast32']['statistic']-c32['statistic']:raise ValueError('Independent fixed eta witness missing/different')
        if not isinstance(w.get('cpu32_witness_nll'),(int,float)) or not math.isfinite(w['cpu32_witness_nll']) or abs(w['contrast32']['nll_subtraction_control']-2*(gs[0]['cpu32_nll']-w['cpu32_witness_nll']))>1e-8:raise ValueError('Independent fixed eta witness NLL control differs')
    if method==METHODS[1]:
        if known_known is None:raise ValueError('Known-known validated dependency required')
        kk=validate_method_record(known_known,contract,binding)
        if p.get('known_known_checkpoint_sha256')!=kk['checkpoint_sha256']:raise ValueError('Known-known validated dependency differs')
        if opt.get('known_known_optimized_checkpoint_sha256')!=kk['optimized_checkpoint_sha256']:raise ValueError('Known-known optimization dependency differs')
        for w in ws:
            if abs(w['witness_nll']-kk[w['phase']]['nll'])>1e-8:raise ValueError('Known-known witness objective differs')
    return copy.deepcopy(p)


def validate_contract(contract, law):
    """Accept only the six-call subset, with exactly the original numeric policy."""
    p = contract['parameters']; s = contract['solver']; v = contract['validation']
    if [m['id'] for m in contract['procedures']] != list(METHODS) or [m['prescribed_minimizations'] for m in contract['procedures']] != [2, 4]:
        raise ValueError('Exact KK2/BG4 procedure subset required')
    if contract['bookkeeping']['per_history_prescribed_minimizations'] != 6:
        raise ValueError('Exactly six prescribed starts required')
    if p['background']['estimated_bounds'] != original.BOUNDS[:2].tolist() or p['eta']['bounds'] != original.BOUNDS[3].tolist():
        raise ValueError('Original parameter bounds required')
    if {k: s['primary'][k] for k in original.LBFGS} != original.LBFGS or s['powell'] != original.POWELL:
        raise ValueError('Original solver options required')
    if p['beta']['value'] != law.beta or not p['beta']['fixed'] or p['background']['known'] != [law.a, law.d]:
        raise ValueError('Fixed supplied law differs')
    if contract['law']['horizon_days']!=law.horizon or contract['law']['coordinate_horizon_days']!=law.coordinate:
        raise ValueError('Fixed horizon/background coordinate differs')
    known = p['trigger']['known']
    if known != dict(n=law.n, r_kappa=law.ratio, alpha_e=law.trigger.alpha, tau_e=law.trigger.tau) or p['stress']['kernel'] != asdict(law.stress):
        raise ValueError('Fixed trigger/stress law differs')
    if v['selected_objectives'] != 4 or v['selected_CPU16_CPU32_absolute_NLL_gate'] != TOLERANCE or v['each_procedure_D16_D32_absolute_gate'] != TOLERANCE:
        raise ValueError('Original numerical validation gates required')
    if v['full_NLL_subtraction_control_absolute_gate'] != TOLERANCE or v['direct_GLR_nonnegative_tolerance'] != TOLERANCE:
        raise ValueError('Original GLR control gates required')


class StagedCase:
    """One case's reusable GPU16, retry CPU16, and independent CPU32 caches.

    The caller owns deadlines, process containment and atomic persistence. A
    method_validated callback is the earliest eligible completed-method commit.
    A newly computed failed method is never automatically retried by this class.
    """
    def __init__(self, arrays, law, *, clock_mode, contract, deadline, context=None, progress_callback=None):
        validate_contract(contract, law)
        self.arrays=arrays; self.law=law; self.clock_mode=clock_mode; self.contract=contract
        self.deadline=deadline; self.context=context; self.callback=progress_callback
        self.binding=input_binding(arrays,law,clock_mode,contract); self.binding_sha256=canonical_hash(self.binding)
        self.gpu=None; self.cpu16=None; self.cpu32=None; self.optimized={}; self.validated={}
        self.records=[]; self.events=[]; self.began=time.perf_counter(); self.closed=False
        self.measurement_started=False

    def emit(self, event):
        event=clean(event); self.events.append(dict(event=event['event'], method=event.get('method')))
        if self.callback is not None:self.callback(event)

    def gpu_history(self):
        cpu._deadline(self.deadline)
        if self.gpu is None:
            if self.context is None:raise ValueError('CUDA context required only for new GPU optimization')
            if not self.measurement_started:self.context.begin_measurement(); self.measurement_started=True
            self.gpu=RestrictedHistory(self.arrays,self.law,clock_mode=self.clock_mode,order=16,backend='cuda',context=self.context,deadline=self.deadline)
        return self.gpu

    def cpu16_history(self):
        cpu._deadline(self.deadline)
        if self.cpu16 is None:self.cpu16=RestrictedHistory(self.arrays,self.law,clock_mode=self.clock_mode,order=16,backend='cpu',deadline=self.deadline)
        return self.cpu16

    def cpu32_history(self):
        cpu._deadline(self.deadline)
        if self.cpu32 is None:self.cpu32=RestrictedHistory(self.arrays,self.law,clock_mode=self.clock_mode,order=32,backend='cpu',deadline=self.deadline)
        return self.cpu32

    def point(self, history, pars, method, phase, origin):
        return clean(dict(pars=list(pars), nll=history.objective(pars,self.law.trigger,self.law.ratio,False),
                          trigger=asdict(self.law.trigger), ratio=self.law.ratio, origin=origin, success=True, method=method, phase=phase))

    def _check_optimized(self, checkpoint, method):
        if checkpoint.get('id')!=method:raise ValueError('Checkpoint method differs')
        return validate_optimized_record(checkpoint,self.contract,self.binding)

    def adopt_optimized(self, checkpoint):
        mid=checkpoint.get('id'); p=self._check_optimized(checkpoint,mid)
        if mid==METHODS[1] and (METHODS[0] not in self.optimized or p.get('known_known_optimized_checkpoint_sha256')!=self.optimized[METHODS[0]]['checkpoint_sha256']):raise ValueError('Known-known optimized witness dependency differs')
        if mid in self.optimized and self.optimized[mid]['checkpoint_sha256']!=p['checkpoint_sha256']:raise ValueError('Conflicting optimization checkpoint')
        self.optimized[mid]=p
        return p

    def adopt_validated(self, checkpoint):
        mid=checkpoint.get('id'); p=validate_method_record(checkpoint,self.contract,self.binding,self.validated.get(METHODS[0]));opt=p['optimized_checkpoint']
        self.optimized[mid]=opt; self.validated[mid]=p
        self.emit(dict(event='method_reused',method=mid,validated_method=p))
        return copy.deepcopy(p)

    def optimize_method_gpu(self, method, optimized_checkpoint=None):
        if method not in METHODS:raise ValueError('Unknown staged method')
        if method in self.optimized:return copy.deepcopy(self.optimized[method])
        if optimized_checkpoint is not None:return self.adopt_optimized(optimized_checkpoint)
        h=self.gpu_history(); began=time.perf_counter(); l=self.law
        bg=method==METHODS[1]; known=np.array([l.a,l.d,l.n,0.]); free0=[0,1] if bg else []; free1=free0+[3]
        a0=float(np.clip(math.log(max(len(h.times)/l.horizon,1e-8)),-12,0)); subfits=[]; nulls=[]
        def one(start,free,phase,si):
            rec=fit_start(h,self.cpu16_history,start,free,l.trigger,l.ratio,dict(method=method,phase=phase,cell_index=0,start_index=si),self.deadline)
            subfits.append(rec); self.records.append(rec)
            self.emit(dict(event='optimizer_completed',method=method,index=len(self.records)-1,record=rec))
            if not rec['success']:raise RuntimeError('Unresolved prescribed '+method+' '+phase+' start')
            return rec
        if bg:
            for si in range(2):
                start=known.copy(); start[:2]=[a0,0.] if si==0 else [float(np.clip(a0-.5,-12,0)),.1]
                nulls.append(one(start,free0,'null',si))
        else:nulls.append(self.point(h,known,method,'null','fixed_null'))
        null=min(nulls,key=lambda r:r['nll'])
        subset=self.point(h,np.r_[null['pars'][:3],0.],method,'alternative','exact_matching_null_subset')
        alternatives=[subset]
        for si,eta in enumerate((0.,.2)):
            start=np.asarray(null['pars']).copy(); start[3]=eta
            alternatives.append(one(start,free1,'alternative',si))
        alt=min(alternatives,key=lambda r:r['nll']); contrast=h.contrast(null,alt)
        witnesses=[]
        if not bg:
            eta=self.contract['law']['alternative_eta']
            point=self.point(h,[l.a,l.d,l.n,eta],method,'alternative','fixed_eta_witness')
            direct=h.contrast(null,point)
            witnesses=[dict(kind='fixed_eta_star',method=method,witness_point=point,witness_nll=point['nll'],selected_nll=alt['nll'],difference=alt['nll']-point['nll'],contrast16=direct,selected_GLR16=contrast['statistic'],direct_GLR_shortfall=direct['statistic']-contrast['statistic'])]
        else:
            if METHODS[0] not in self.optimized:raise ValueError('Known-known optimization is required for subset witnesses')
            kk=self.optimized[METHODS[0]]
            for phase,selected in (('null',null),('alternative',alt)):
                r=kk[phase]; value=h.objective(r['pars'],l.trigger,l.ratio,False)
                witnesses.append(dict(kind='cross_procedure_subset',broader=method,narrower=METHODS[0],broader_cell=0,narrower_cell=0,phase=phase,witness_nll=value,selected_nll=selected['nll'],difference=selected['nll']-value))
        out=dict(checkpoint_kind='optimized_method',binding_sha256=self.binding_sha256,input_binding=self.binding,id=method,complete=False,status='optimized_unvalidated',optimization_complete=True,
                 subfits=subfits,null=null,alternative=alt,cells=[dict(index=0,trigger=asdict(l.trigger),ratio=l.ratio,null=null,alternative=alt,exact_null_subset=subset)],
                 contrast16=contrast,witnesses=witnesses,optimization_seconds=time.perf_counter()-began,cuda_metrics=self.context.cuda_metrics(32768),
                 feature_metadata=dict(nodes16=h.nodes,panels=h.panels,stress_modes=len(l.stress.mixture[0]),preparation_seconds=h.preparation_seconds,clock_mode=h.mode,mark_nll_constant=h.mark_nll_constant))
        if bg:out['known_known_optimized_checkpoint_sha256']=self.optimized[METHODS[0]]['checkpoint_sha256']
        out=seal(out); self.optimized[method]=out
        self.emit(dict(event='procedure_optimized',method=method,optimizer_calls=len(subfits),partial_method=out))
        return copy.deepcopy(out)

    def validate_method_cpu(self, method, optimized_checkpoint=None):
        """Independently validate this method; publish complete only after all gates."""
        if method in self.validated:return copy.deepcopy(self.validated[method])
        opt=self.adopt_optimized(optimized_checkpoint) if optimized_checkpoint is not None else self._check_optimized(self.optimized[method],method)
        if method==METHODS[1] and METHODS[0] not in self.validated:raise ValueError('Background validation requires validated known-known method')
        began=time.perf_counter(); h=self.cpu32_history(); selected=[]
        for phase in ('null','alternative'):
            rec=opt[phase]; value=h.objective(rec['pars'],self.law.trigger,self.law.ratio,False); delta=abs(value-rec['nll'])
            selected.append(dict(method=method,phase=phase,cuda16_nll=rec['nll'],cpu32_nll=value,absolute_difference=delta,passed=delta<TOLERANCE))
        c32=h.contrast(opt['null'],opt['alternative']); c16=opt['contrast16']; delta=abs(c32['statistic']-c16['statistic'])
        gates=dict(statistic_16_32_difference=delta,nll_subtraction_control_gate=TOLERANCE,
                   direct_nesting_passed=c16['statistic']>=-TOLERANCE and c32['statistic']>=-TOLERANCE,
                   passed=delta<TOLERANCE and abs(c16['direct_minus_subtraction'])<TOLERANCE and abs(c32['direct_minus_subtraction'])<TOLERANCE and
                   opt['alternative']['nll']<=opt['null']['nll']+1e-8 and c16['statistic']>=-TOLERANCE and c32['statistic']>=-TOLERANCE)
        witnesses=copy.deepcopy(opt['witnesses'])
        if method==METHODS[0]:
            w=witnesses[0]; fixed32=h.contrast(opt['null'],w['witness_point'])
            witness_value32=h.objective(w['witness_point']['pars'],self.law.trigger,self.law.ratio,False)
            w.update(contrast32=fixed32,cpu32_witness_nll=witness_value32,selected_GLR32=c32['statistic'],direct_GLR32_shortfall=fixed32['statistic']-c32['statistic'])
        for w in witnesses:
            w['passed']=bool(math.isfinite(w['difference']) and w['difference']<=TOLERANCE and w.get('direct_GLR_shortfall',0.)<=TOLERANCE and w.get('direct_GLR32_shortfall',0.)<=TOLERANCE)
        passed=bool(gates['passed'] and all(g['passed'] for g in selected) and all(w['passed'] for w in witnesses))
        out=dict(id=method,checkpoint_kind='validated_method' if passed else 'unresolved_method',binding_sha256=self.binding_sha256,input_binding=self.binding,complete=passed,status='complete' if passed else 'unresolved_validation',
                 optimization_complete=True,optimized_checkpoint_sha256=opt['checkpoint_sha256'],optimized_checkpoint=opt,subfits=opt['subfits'],cells=opt['cells'],null=opt['null'],alternative=opt['alternative'],
                 contrast16=c16,contrast32=c32,gates=gates,selected_objectives=selected,witnesses=witnesses,validation_seconds=time.perf_counter()-began,
                 cuda_metrics=opt['cuda_metrics'],cpu32_feature_metadata=dict(nodes32=h.nodes,preparation_seconds=h.preparation_seconds,clock_mode=h.mode))
        if method==METHODS[1]:out['known_known_checkpoint_sha256']=self.validated[METHODS[0]]['checkpoint_sha256']
        out=seal(out)
        if passed:
            self.validated[method]=out
            self.emit(dict(event='method_validated',method=method,validated_method=out))
        else:self.emit(dict(event='method_unresolved',method=method,partial_method=out))
        return copy.deepcopy(out)

    def close(self):
        if self.closed:return
        self.closed=True
        for h in (self.gpu,self.cpu16,self.cpu32):
            if h is not None:h.close()


def optimize_case_gpu(arrays,law,*,clock_mode,contract,deadline,context,progress_callback=None):
    """Future split-topology helper, explicitly unvalidated; benchmark uses fit_case."""
    stage=StagedCase(arrays,law,clock_mode=clock_mode,contract=contract,deadline=deadline,context=context,progress_callback=progress_callback)
    try:
        for mid in METHODS:stage.optimize_method_gpu(mid)
        return clean(dict(complete=False,status='optimized_unvalidated',binding=stage.binding,methods=list(stage.optimized.values()),prescribed_optimizer_calls=len(stage.records)))
    finally:stage.close()


def validate_case_cpu(arrays,law,optimized,*,clock_mode,contract,deadline,progress_callback=None):
    """Independent CPU-only consumer of source/input-bound GPU checkpoints."""
    stage=StagedCase(arrays,law,clock_mode=clock_mode,contract=contract,deadline=deadline,progress_callback=progress_callback)
    methods=[]
    try:
        for mid in METHODS:
            opt=next(m for m in optimized['methods'] if m['id']==mid)
            method=stage.validate_method_cpu(mid,opt); methods.append(method)
            if not method['complete']:break
        return clean(dict(complete=len(methods)==2 and all(m['complete'] for m in methods),methods=methods))
    finally:stage.close()


def fit_case(arrays,law,*,clock_mode,contract,deadline,context,progress_callback=None,optimized_checkpoints=None,validated_checkpoints=None):
    """Serial benchmark: KK optimize/validate/commit, then BG optimize/validate/commit."""
    stage=StagedCase(arrays,law,clock_mode=clock_mode,contract=contract,deadline=deadline,context=context,progress_callback=progress_callback)
    result=dict(complete=False,status='partial',methods=[],records=[],planned_optimizer_calls=6,prescribed_optimizer_calls=0)
    optimized_checkpoints=optimized_checkpoints or {}; validated_checkpoints=validated_checkpoints or {}
    try:
        for mid in METHODS:
            if mid in validated_checkpoints:method=stage.adopt_validated(validated_checkpoints[mid])
            else:
                opt=stage.optimize_method_gpu(mid,optimized_checkpoints.get(mid))
                method=stage.validate_method_cpu(mid,opt)
            result['methods'].append(method)
            if not method['complete']:result['status']='unresolved_validation';break
        result['complete']=len(result['methods'])==2 and all(m['complete'] for m in result['methods'])
        if result['complete']:result['status']='complete'
    except Exception as exc:
        result.update(status='time_cap' if isinstance(exc,TimeoutError) else 'unresolved',error=f'{type(exc).__name__}: {exc}')
    finally:
        # Already validated methods survive any later optimization/validation failure.
        for mid in METHODS:
            if mid in stage.validated and not any(m['id']==mid for m in result['methods']):result['methods'].append(copy.deepcopy(stage.validated[mid]))
        result['optimized_methods']=[copy.deepcopy(stage.optimized[mid]) for mid in METHODS if mid in stage.optimized]
        result['records']=[r for mid in METHODS for r in stage.optimized.get(mid,{}).get('subfits',[])]
        # Preserve completed starts of an interrupted/unresolved method as partial records.
        keys={(r['method'],r['phase'],r['start_index']) for r in result['records']}
        result['records'] += [r for r in stage.records if (r['method'],r['phase'],r['start_index']) not in keys]
        result['prescribed_optimizer_calls']=len(result['records']); result['new_optimizer_calls']=len(stage.records)
        result['witnesses']=[w for m in result['methods'] for w in m.get('witnesses',[])]
        objectives=[g for m in result['methods'] for g in m.get('selected_objectives',[])]
        result['validation']=dict(selected_objectives=objectives,witnesses_passed=len(result['witnesses'])==3 and all(w['passed'] for w in result['witnesses']))
        result['binding']=stage.binding
        if stage.measurement_started:result['cuda_metrics']=context.cuda_metrics(32768)
        elif result['methods']:result['cuda_metrics']=result['methods'][-1].get('cuda_metrics')
        stage.close()
        result['elapsed_seconds']=time.perf_counter()-stage.began
    return clean(result)


def main():
    p=argparse.ArgumentParser(description='Six-call staged fitting library; saved-history execution requires the frozen root benchmark runner.')
    p.parse_args()
    print('fit_case(arrays,law,clock_mode=...,contract=...,deadline=...,context=...,progress_callback=...)')


if __name__=='__main__':main()
