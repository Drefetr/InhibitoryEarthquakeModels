"""Bounded FP64 CUDA quadrature for the immutable iteration-17 oracle law.

CPU event arithmetic and clock/node construction retain the CPU authority.
Only quadrature vector operations move to CUDA; no history or result cache.
"""
from dataclasses import dataclass
from pathlib import Path
import gc,hashlib,importlib.util,math,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
CPU_SOURCE=ROOT/'iteration_17/code/oracle_likelihood.py'
CPU_SHA='6e8fa9ce82cb5c2d100adaf8065138172830618da49047e9c5aa73ccfaa3bb46'
if hashlib.sha256(CPU_SOURCE.read_bytes()).hexdigest()!=CPU_SHA:raise ValueError('Iteration17 CPU authority changed')
_name='_iteration17_oracle_cpu_authority'
if _name not in sys.modules:
    _spec=importlib.util.spec_from_file_location(_name,CPU_SOURCE);cpu=importlib.util.module_from_spec(_spec);sys.modules[_name]=cpu;_spec.loader.exec_module(cpu)
else:cpu=sys.modules[_name]
KnownLaw=cpu.KnownLaw
KernelSpec=cpu.KernelSpec
law_from_setting=cpu.law_from_setting
PRIMARY_ETA=cpu.PRIMARY_ETA
OracleDeadlineExceeded=cpu.OracleDeadlineExceeded
MAX_CHUNK_NODES=32768


class CudaOracleResourceExceeded(MemoryError):pass


class CudaOracleContext:
    """One persistent CUDA context; cache only immutable kernel constants."""
    def __init__(self,device='cuda:0',max_allocated_bytes=1024*2**20):
        began=time.perf_counter()
        import torch
        if not torch.cuda.is_available():raise RuntimeError('Usable CUDA required; no silent CPU fallback')
        if not isinstance(max_allocated_bytes,int) or max_allocated_bytes<=0:raise ValueError('Positive allocation cap required')
        self.torch=torch;self.device=torch.device(device);self.max_allocated_bytes=max_allocated_bytes;self.constants={};self.closed=False
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.init();torch.cuda.synchronize(self.device)
        properties=torch.cuda.get_device_properties(self.device)
        self.metadata=dict(device=str(self.device),device_name=properties.name,compute_capability=list(torch.cuda.get_device_capability(self.device)),
            device_total_memory_bytes=int(properties.total_memory),torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
            floating_dtype='torch.float64',index_dtype='torch.int64',tf32_enabled=False,mixed_precision=False,fast_math_requested=False,
            context_initialization_seconds=time.perf_counter()-began,max_allocated_bytes=max_allocated_bytes,
            cache_scope='Kernel weights/rates only; no prepared catalogue, query, history or score cached')

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def begin_measurement(self):
        if self.closed:raise RuntimeError('CUDA oracle context closed')
        self.torch.cuda.synchronize(self.device);self.torch.cuda.reset_peak_memory_stats(self.device)
        return dict(allocated_at_start_bytes=int(self.torch.cuda.memory_allocated(self.device)),reserved_at_start_bytes=int(self.torch.cuda.memory_reserved(self.device)))

    def memory_metrics(self):
        self.torch.cuda.synchronize(self.device)
        peak=int(self.torch.cuda.max_memory_allocated(self.device))
        if peak>self.max_allocated_bytes:raise CudaOracleResourceExceeded(f'Torch peak allocation {peak} exceeds {self.max_allocated_bytes}')
        return dict(peak_cuda_allocated_bytes=peak,peak_cuda_reserved_bytes=int(self.torch.cuda.max_memory_reserved(self.device)),
            allocated_at_end_bytes=int(self.torch.cuda.memory_allocated(self.device)),reserved_at_end_bytes=int(self.torch.cuda.memory_reserved(self.device)))

    def cuda_metrics(self,chunk_nodes):
        memory=self.memory_metrics()
        return dict(dtype='float64',allow_tf32=bool(self.torch.backends.cuda.matmul.allow_tf32),mixed_precision=False,
            fast_math_requested=False,peak_allocated_bytes=memory['peak_cuda_allocated_bytes'],peak_reserved_bytes=memory['peak_cuda_reserved_bytes'],
            device=str(self.device),synchronized=True,fresh_case_buffers=True,chunk_nodes=int(chunk_nodes),max_allocated_bytes=self.max_allocated_bytes)

    def kernel_constants(self,kernel):
        if kernel.key not in self.constants:
            w,r=kernel.mixture
            self.constants[kernel.key]=(self.torch.as_tensor(w,dtype=self.torch.float64,device=self.device),self.torch.as_tensor(r,dtype=self.torch.float64,device=self.device))
        return self.constants[kernel.key]

    def reduce_block(self,block,kernel,a,d,eta,deadline=None):
        cpu._deadline(deadline);torch=self.torch;w,r=self.kernel_constants(kernel)
        began=time.perf_counter()
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            lag=torch.as_tensor(block['lag'],dtype=torch.float64,device=self.device)
            x=torch.as_tensor(block['x'],dtype=torch.float64,device=self.device)
            weight=torch.as_tensor(block['weight'],dtype=torch.float64,device=self.device)
            states=torch.as_tensor(block['states'],dtype=torch.float64,device=self.device)
            index=torch.as_tensor(block['index'],dtype=torch.int64,device=self.device)
            transfer_seconds=time.perf_counter()-began
            first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True);first.record()
            g=torch.sum(torch.exp(-lag[:,None]*r)*states[index]*w,dim=1)
            total=torch.sum(weight*torch.exp(a+d*x)*(-torch.expm1(-eta*g)))
            last.record();last.synchronize()
            compute_ms=float(first.elapsed_time(last))
            value=float(total.item())
            if not math.isfinite(value):raise FloatingPointError('Nonfinite FP64 CUDA quadrature block')
            del lag,x,weight,states,index,g,total,first,last
            memory=self.memory_metrics()
        cpu._deadline(deadline)
        return value,dict(host_transfer_seconds=transfer_seconds,cuda_compute_milliseconds=compute_ms,**memory)

    def warmup(self):
        """A tiny deterministic tensor fixture, never a saved-history score."""
        began=time.perf_counter();start=self.begin_measurement();kernel=KernelSpec('ml',1.)
        block=dict(lag=np.linspace(.001,2.,32),x=np.zeros(32),weight=np.full(32,1/32),states=np.ones((1,len(kernel.mixture[0]))),index=np.zeros(32,dtype=np.int64))
        _,metrics=self.reduce_block(block,kernel,-4.,.2,.5)
        self.torch.cuda.synchronize(self.device)
        return dict(warmup_seconds=time.perf_counter()-began,scope='32-node deterministic FP64 tensor fixture; no catalogue or scientific score',**start,**metrics)

    def close(self):
        if not self.closed:
            self.torch.cuda.synchronize(self.device);self.constants.clear();gc.collect();self.torch.cuda.empty_cache();self.closed=True
        # PyTorch's process-owned CUDA runtime context ends with the worker.


@dataclass
class PreparedOracle:
    times:np.ndarray
    lows:np.ndarray|None
    intervals:np.ndarray
    stress_amplitudes:np.ndarray
    law:object
    eta:float
    mode:str
    event_log_sum:float
    event_log_ratio_sum:float
    baseline:float
    trigger_integral:float
    preparation_seconds:float


def prepare_oracle(arrays,law,*,clock_mode,eta=PRIMARY_ETA,deadline=None):
    """Fresh CPU event pass; copied arithmetic from the pinned CPU authority."""
    began=time.perf_counter();cpu._deadline(deadline)
    if not math.isfinite(eta) or eta<0:raise ValueError('Invalid eta')
    t,q,lo,intervals,mode=cpu._clock(arrays,law,clock_mode);paired=mode=='paired_clock'
    sw,sr=law.stress.mixture;ew,er=law.trigger.mixture;s=np.zeros(len(sr));e=np.zeros(len(er));logwr=np.log(ew)+np.log(er)
    event0=[];ratios=[];integrals=[];amplitudes=np.empty(len(t))
    for j,gap in enumerate(intervals[:-1]):
        cpu._deadline(deadline);s*=np.exp(-sr*gap);e*=np.exp(-er*gap);ge=float(np.dot(s,sw));logtrigger=-math.inf
        if law.n>0 and np.any(e):
            loge=np.full_like(e,-np.inf);np.log(e,out=loge,where=e>0)
            logtrigger=math.log(law.n)+float(cpu.logsumexp(loge+logwr))
        absolute=(t[j]+lo[j]) if paired else t[j];logb=law.a+law.d*(2*absolute/law.coordinate-1)
        l0=float(np.logaddexp(logb,logtrigger));event0.append(l0)
        if eta==0 or ge==0:lr=0.
        else:
            with np.errstate(over='ignore'):penalty=float(eta*ge)
            lr=float(np.logaddexp((logb-l0)-penalty,logtrigger-l0))
        if not math.isfinite(lr):raise FloatingPointError('Unrepresentable event log ratio')
        ratios.append(lr)
        try:gm=math.exp(cpu.GAMMA*float(q[j]))/math.exp(cpu.GAMMA);kp=(1-law.ratio)*math.exp(law.beta*law.ratio*float(q[j]))
        except OverflowError as exc:raise FloatingPointError('Unrepresentable transformed mark; no clipping') from exc
        amplitudes[j]=gm;s+=gm;e+=kp
        if not np.isfinite(s).all() or not np.isfinite(e).all():raise FloatingPointError('Unrepresentable modal state; no clipping')
        duration=cpu._difference(law.horizon,0.,t[j],lo[j]) if paired else law.horizon-t[j]
        integrals.append(float(kp*np.dot(ew,-np.expm1(-er*duration))))
    baseline=math.exp(law.a-law.d)*law.horizon*float(cpu.exprel(2*law.d*law.horizon/law.coordinate))
    return PreparedOracle(t,lo,intervals,amplitudes,law,float(eta),mode,math.fsum(event0),math.fsum(ratios),baseline,math.fsum(integrals),time.perf_counter()-began)


def query_blocks(prepared,order,chunk_nodes,deadline=None):
    """Exact CPU-authority node arithmetic, batched across interval boundaries."""
    if int(order)!=order or order<=0 or int(chunk_nodes)!=chunk_nodes or not 1<=chunk_nodes<=MAX_CHUNK_NODES:raise ValueError('Invalid order/chunk bound')
    p=prepared;paired=p.mode=='paired_clock';z,w=cpu.leggauss(int(order));sw,sr=p.law.stress.mixture;s=np.zeros(len(sr))
    lags=[];xs=[];weights=[];indices=[];states=[];used=0;total_nodes=0;total_panels=0
    for j,gap in enumerate(p.intervals):
        cpu._deadline(deadline)
        if gap>0 and p.eta!=0 and np.any(s):
            a=float(p.times[j-1]) if j else 0.;low=float(p.lows[j-1]) if paired and j else 0.
            offsets=cpu.OFFSETS[cpu.OFFSETS<gap];edges=np.r_[0.,offsets,gap] if paired else np.r_[a,a+offsets,a+gap]
            query=((edges[:-1,None]+edges[1:,None])/2+(edges[1:,None]-edges[:-1,None])*z/2).ravel()
            weight=((edges[1:,None]-edges[:-1,None])*w/2).ravel();relative=query if paired else query-a
            absolute=a+(low+query) if paired else query;x=2*absolute/p.law.coordinate-1
            total_panels+=len(edges)-1;total_nodes+=len(query);first=0;state_index=None
            while first<len(query):
                if state_index is None:state_index=len(states);states.append(s.copy())
                count=min(chunk_nodes-used,len(query)-first);sl=slice(first,first+count)
                lags.append(relative[sl]);xs.append(x[sl]);weights.append(weight[sl]);indices.append(np.full(count,state_index,dtype=np.int64));used+=count;first+=count
                if used==chunk_nodes:
                    yield dict(lag=np.concatenate(lags),x=np.concatenate(xs),weight=np.concatenate(weights),index=np.concatenate(indices),states=np.stack(states),nodes=used,total_nodes=total_nodes,total_panels=total_panels)
                    lags=[];xs=[];weights=[];indices=[];states=[];used=0;state_index=None
        if j<len(p.times):s*=np.exp(-sr*gap);s+=p.stress_amplitudes[j]
    if used:yield dict(lag=np.concatenate(lags),x=np.concatenate(xs),weight=np.concatenate(weights),index=np.concatenate(indices),states=np.stack(states),nodes=used,total_nodes=total_nodes,total_panels=total_panels)


def _evaluate_prepared(p,order,context,chunk_nodes,deadline=None):
    began=time.perf_counter();terms=[];blocks=0;nodes=0;panels=0;maxnodes=0;maxstates=0;transfer=0.;compute_ms=0.;gpu_began=time.perf_counter()
    for block in query_blocks(p,order,chunk_nodes,deadline):
        value,metrics=context.reduce_block(block,p.law.stress,p.law.a,p.law.d,p.eta,deadline)
        terms.append(value);blocks+=1;nodes+=block['nodes'];panels=block['total_panels'];maxnodes=max(maxnodes,block['nodes']);maxstates=max(maxstates,len(block['states']))
        transfer+=metrics['host_transfer_seconds'];compute_ms+=metrics['cuda_compute_milliseconds']
    context.torch.cuda.synchronize(context.device);cpu._deadline(deadline);quad_seconds=time.perf_counter()-gpu_began
    c=math.fsum(terms);events=p.event_log_ratio_sum;lr=math.fsum((events,c))
    ll0=p.event_log_sum-p.baseline-p.law.n*p.trigger_integral
    ll1=math.fsum((p.event_log_sum,events,-p.baseline,c,-p.law.n*p.trigger_integral))
    if p.eta==0:lr=events=c=0.;ll1=ll0
    if not all(math.isfinite(x) for x in (lr,ll0,ll1,c)):raise FloatingPointError('Nonfinite CUDA oracle result')
    return dict(log_lr=lr,event_log_ratio_sum=events,compensator_reduction=c,nll0=-ll0,nll1=-ll1,
        analytic_baseline_compensator=p.baseline,unit_trigger_compensator=p.trigger_integral,n_events=len(p.times),eta=p.eta,beta_fixed=p.law.beta,
        clock_mode=p.mode,order=int(order),method='cuda_fp64_gauss',quadrature_nodes=nodes,panels=panels,adaptive_error_sum=0.,
        elapsed_seconds=time.perf_counter()-began,quadrature_seconds=quad_seconds,host_transfer_seconds=transfer,cuda_compute_milliseconds=compute_ms,
        maximum_working_query_nodes=maxnodes,maximum_working_interval_states=maxstates,stress_modes=len(p.law.stress.mixture[0]),trigger_modes=len(p.law.trigger.mixture[0]),blocks=blocks,
        mark_density_cancellation='Same fixed iid exponential beta and support under both simple laws; no beta refit',
        statistic='Signed full-window ground log likelihood ratio, including compensator and random count',
        event_arithmetic='Pinned CPU-authority FP64 event recurrence and stable log-space ratios; no event-term GPU reduction',
        cuda_metrics=context.cuda_metrics(chunk_nodes),**context.memory_metrics())


def evaluate_oracle_cuda(arrays,law,*,clock_mode,eta=PRIMARY_ETA,order=32,deadline=None,chunk_nodes=MAX_CHUNK_NODES,context=None):
    began=time.perf_counter();owned=context is None;ctx=CudaOracleContext() if owned else context;result=None
    try:
        start=ctx.begin_measurement();p=prepare_oracle(arrays,law,clock_mode=clock_mode,eta=eta,deadline=deadline)
        result=_evaluate_prepared(p,order,ctx,chunk_nodes,deadline)
        result.update(preparation_seconds=p.preparation_seconds,context_metadata=ctx.metadata,context_created_for_call=owned,**start)
        result['elapsed_seconds']=time.perf_counter()-began
        cpu._deadline(deadline)
        return result
    finally:
        if owned:ctx.close()
        if result is not None:result['whole_call_seconds_including_owned_cleanup']=time.perf_counter()-began


def score_refined_cuda(arrays,law,*,clock_mode,eta=PRIMARY_ETA,tolerance=1e-6,deadline=None,chunk_nodes=MAX_CHUNK_NODES,context=None):
    if tolerance<=0 or not math.isfinite(tolerance):raise ValueError('Positive finite refinement tolerance')
    began=time.perf_counter();owned=context is None;ctx=CudaOracleContext() if owned else context;output=None
    try:
        start=ctx.begin_measurement();p=prepare_oracle(arrays,law,clock_mode=clock_mode,eta=eta,deadline=deadline);values={}
        for order in (16,32):values[str(order)]=_evaluate_prepared(p,order,ctx,chunk_nodes,deadline)
        delta=abs(values['32']['log_lr']-values['16']['log_lr']);initial_delta=delta;selected=32
        if delta>tolerance:
            values['64']=_evaluate_prepared(p,64,ctx,chunk_nodes,deadline);delta=abs(values['64']['log_lr']-values['32']['log_lr']);selected=64
        output=dict(complete=bool(delta<=tolerance),refinement_passed=bool(delta<=tolerance),tolerance=tolerance,
            initial_16_32_difference=initial_delta,selected_refinement_difference=delta,selected_order=selected,result=values[str(selected)],evaluations=values,
            elapsed_seconds=time.perf_counter()-began,preparation_seconds=p.preparation_seconds,preparation_shared_across_orders=True,
            context_metadata=ctx.metadata,context_created_for_call=owned,catalogue_cache_reused=False,chunk_nodes=chunk_nodes,**start,**ctx.memory_metrics())
        output['cuda_metrics']=ctx.cuda_metrics(chunk_nodes)
        cpu._deadline(deadline)
        return output
    finally:
        if owned:ctx.close()
        if output is not None:output['whole_call_seconds_including_owned_cleanup']=time.perf_counter()-began
