"""Fixed-stress, fixed-beta nuisance-ablation likelihood and analytic gradient.

No optimizer or saved-history execution occurs on import. Scalar history caches
are O(quadrature nodes + events), with bounded modal construction blocks.
"""
from pathlib import Path
from types import SimpleNamespace
import gc,hashlib,importlib.util,math,sys,time
import numpy as np
from scipy.special import exprel,logsumexp
ROOT=Path(__file__).resolve().parents[2]
CUDA_PATH=ROOT/'iteration_18/code/cuda_oracle.py'
CUDA_SHA='be7ff7938596c0f5c3bc40962986e2ae511a0a9c1b05bc1d8d7b99008fa86505'
if hashlib.sha256(CUDA_PATH.read_bytes()).hexdigest()!=CUDA_SHA:raise ValueError('Frozen CUDA oracle source changed')
_name='_restricted20_cuda_authority'
if _name not in sys.modules:
    _spec=importlib.util.spec_from_file_location(_name,CUDA_PATH);_module=importlib.util.module_from_spec(_spec);sys.modules[_name]=_module;_spec.loader.exec_module(_module)
oracle=sys.modules[_name];cpu=oracle.cpu
KnownLaw=oracle.KnownLaw;KernelSpec=oracle.KernelSpec;law_from_setting=oracle.law_from_setting


class UnrepresentableDerivative(FloatingPointError):pass


def background_integrals(a,d,duration,coordinate):
    """Analytic integral of exp(a+d*x), and its d derivative, from zero."""
    z=2*d*duration/coordinate
    if abs(z)<.1:
        derivative=math.fsum((k+1)*z**k/math.factorial(k+2) for k in range(14))
    else:derivative=((z-1)*math.exp(z)+1)/(z*z)
    scale=math.exp(a-d)*duration
    return scale*float(exprel(z)),scale*(-float(exprel(z))+2*duration/coordinate*derivative)


def _finite(values,message):
    if not np.isfinite(values).all():raise FloatingPointError(message)


class RestrictedHistory:
    def __init__(self,arrays,law,*,clock_mode,order=16,backend='cpu',context=None,deadline=None,chunk_nodes=32768,host_limit_bytes=2*2**30):
        began=time.perf_counter();cpu._deadline(deadline)
        if backend not in ('cpu','cuda'):raise ValueError('backend must be cpu or cuda')
        if order not in (16,32,64) or not 1<=chunk_nodes<=32768:raise ValueError('Invalid quadrature policy')
        self.arrays=arrays;self.law=law;self.order=order;self.backend=backend;self.deadline=deadline;self.chunk_nodes=chunk_nodes
        self.context=context;self.owned_context=False;self.device_arrays={};self.trigger={};self.trigger_device={};self.host_limit_bytes=host_limit_bytes
        self.times,self.excess,self.lows,self.intervals,self.mode=cpu._clock(arrays,law,clock_mode)
        self.xe=np.array([2*((t+lo) if self.lows is not None else t)/law.coordinate-1 for t,lo in zip(self.times,self.lows if self.lows is not None else np.zeros(len(self.times)))])
        self.mark_nll_constant=float(-len(self.times)*math.log(law.beta)+law.beta*math.fsum(self.excess))
        self.ge=np.empty(len(self.times));self.amplitudes=np.exp(cpu.GAMMA*self.excess)/math.exp(cpu.GAMMA);_finite(self.amplitudes,'Unrepresentable stress mark; no clipping')
        weights,rates=law.stress.mixture;state=np.zeros(len(rates))
        for i,gap in enumerate(self.intervals[:-1]):
            cpu._deadline(deadline);state*=np.exp(-rates*gap);self.ge[i]=state@weights;state+=self.amplitudes[i]
        _finite(self.ge,'Unrepresentable fixed stress history')
        self.quiet_duration=float(self.times[0])+(float(self.lows[0]) if self.lows is not None else 0.) if len(self.times) else law.horizon
        if backend=='cuda' and context is None:self.context=oracle.CudaOracleContext();self.owned_context=True
        # Query geometry is exactly the pinned projected/paired oracle geometry.
        p=SimpleNamespace(times=self.times,lows=self.lows,intervals=self.intervals,stress_amplitudes=self.amplitudes,law=law,eta=1.,mode=self.mode)
        gs=[];xs=[];ws=[];self.nodes=0;self.panels=0
        for block in oracle.query_blocks(p,order,chunk_nodes,deadline):
            self.nodes+=len(block['lag']);self.panels=block['total_panels']
            if 3*self.nodes*8>host_limit_bytes:raise MemoryError('Declared scalar feature cache exceeds host allowance')
            if backend=='cuda':
                torch=self.context.torch;w,r=self.context.kernel_constants(law.stress)
                with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
                    lag=torch.as_tensor(block['lag'],dtype=torch.float64,device=self.context.device);states=torch.as_tensor(block['states'],dtype=torch.float64,device=self.context.device);index=torch.as_tensor(block['index'],dtype=torch.int64,device=self.context.device)
                    g=torch.sum(states[index]*torch.exp(-lag[:,None]*r)*w,dim=1);value=g.cpu().numpy().copy();del lag,states,index,g
                self.context.memory_metrics()
            else:value=np.sum(block['states'][block['index']]*np.exp(-block['lag'][:,None]*rates)*weights,axis=1)
            gs.append(value);xs.append(block['x']);ws.append(block['weight'])
        self.gq=np.concatenate(gs) if gs else np.empty(0);self.xq=np.concatenate(xs) if xs else np.empty(0);self.wq=np.concatenate(ws) if ws else np.empty(0)
        _finite(self.gq,'Unrepresentable quadrature stress')
        if backend=='cuda':
            for name in ('ge','xe','gq','xq','wq'):self.device_arrays[name]=self.context.torch.as_tensor(getattr(self,name),dtype=self.context.torch.float64,device=self.context.device)
            self.context.memory_metrics()
        cpu._deadline(deadline);self.preparation_seconds=time.perf_counter()-began

    def close(self):
        self.device_arrays.clear();self.trigger_device.clear();gc.collect()
        if self.context is not None:
            self.context.torch.cuda.synchronize(self.context.device)
            if self.owned_context:self.context.close()

    def trigger_features(self,kernel,ratio):
        key=(kernel.key,float(ratio))
        if key in self.trigger:return self.trigger[key]
        cpu._deadline(self.deadline);w,r=kernel.mixture;state=np.zeros(len(r));logwr=np.log(w)+np.log(r)
        with np.errstate(over='ignore'):amplitudes=(1-ratio)*np.exp(self.law.beta*ratio*self.excess)
        _finite(amplitudes,'Unrepresentable productivity mark; no clipping')
        logh=np.full(len(self.times),-np.inf)
        for i,gap in enumerate(self.intervals[:-1]):
            cpu._deadline(self.deadline);state*=np.exp(-r*gap)
            if np.any(state):
                with np.errstate(over='ignore'):h=float(state@(w*r))
                if math.isfinite(h) and h>0:logh[i]=math.log(h)
                else:
                    logs=np.full_like(state,-np.inf);np.log(state,out=logs,where=state>0);logh[i]=float(logsumexp(logs+logwr))
            state+=amplitudes[i]
        duration=np.array([cpu._difference(self.law.horizon,0.,t,lo) if self.lows is not None else self.law.horizon-t for t,lo in zip(self.times,self.lows if self.lows is not None else np.zeros(len(self.times)))])
        component=np.empty(len(duration))
        for j in range(0,len(duration),4096):
            cpu._deadline(self.deadline);sl=slice(j,j+4096);component[sl]=amplitudes[sl]*np.sum((-np.expm1(-duration[sl,None]*r))*w,axis=1)
        integral=math.fsum(component);_finite([integral],'Unrepresentable analytic trigger compensator')
        self.trigger[key]=(logh,integral,component)
        return self.trigger[key]

    def event_logs(self,pars,kernel,ratio):
        a,d,n,eta=np.asarray(pars,dtype=float);logh,_,_=self.trigger_features(kernel,ratio)
        logb=a+d*self.xe-eta*self.ge
        return np.logaddexp(logb,logh+math.log(n) if n>0 else np.full_like(logh,-np.inf))

    def objective(self,pars,kernel,ratio,gradient=True,free=None):
        """Ground NLL; fixed mark NLL constant is separately available.

        free requests just those derivatives, avoiding an unused unrepresentable
        n-boundary derivative when n is supplied. No derivative is clipped.
        """
        cpu._deadline(self.deadline);pars=np.asarray(pars,dtype=float)
        if pars.shape!=(4,) or not np.isfinite(pars).all() or pars[2]<0 or pars[3]<0:raise ValueError('Invalid physical parameters')
        free=list(range(4)) if free is None else list(free)
        if self.backend=='cuda':return self._cuda_objective(pars,kernel,ratio,gradient,free)
        a,d,n,eta=pars;logh,integral,_=self.trigger_features(kernel,ratio);logb=a+d*self.xe-eta*self.ge
        loglam=np.logaddexp(logb,logh+math.log(n) if n>0 else np.full_like(logh,-np.inf));p=np.exp(logb-loglam)
        if eta==0:j,jd=background_integrals(a,d,self.law.horizon,self.law.coordinate)
        else:
            j,jd=background_integrals(a,d,self.quiet_duration,self.law.coordinate)
            wb=self.wq*np.exp(a+d*self.xq-eta*self.gq);j+=math.fsum(wb);jd+=float(wb@self.xq)
        value=math.fsum((j,n*integral,-math.fsum(loglam)));_finite([value],'Nonfinite likelihood')
        if not gradient:return value
        result={0:j-math.fsum(p),1:jd-float(p@self.xe)}
        if 3 in free:
            wb=self.wq*np.exp(a+d*self.xq-eta*self.gq);result[3]=float(p@self.ge)-float(wb@self.gq)
        if 2 in free:
            lr=float(logsumexp(logh-loglam))
            if lr>math.log(np.finfo(float).max):raise UnrepresentableDerivative('True n-boundary derivative exceeds FP64')
            result[2]=integral-(math.exp(lr) if lr!=-math.inf else 0.)
        grad=np.array([result[i] for i in free]);_finite(grad,'Requested derivative is not representable')
        return value,grad

    def _cuda_objective(self,pars,kernel,ratio,gradient,free):
        torch=self.context.torch;a,d,n,eta=pars;logh,integral,_=self.trigger_features(kernel,ratio);key=(kernel.key,float(ratio))
        if key not in self.trigger_device:self.trigger_device[key]=torch.as_tensor(logh,dtype=torch.float64,device=self.context.device)
        h=self.trigger_device[key];ge=self.device_arrays['ge'];xe=self.device_arrays['xe'];terms=[];dterms=[];gterms=[]
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            logb=a+d*xe-eta*ge;ll=torch.logaddexp(logb,h+math.log(n) if n>0 else torch.full_like(h,-math.inf));p=torch.exp(logb-ll)
            event=float(torch.sum(ll).item());pa=float(torch.sum(p).item());pd=float(torch.sum(p*xe).item());pg=float(torch.sum(p*ge).item())
            for j in range(0,self.nodes if eta!=0 or gradient and 3 in free else 0,self.chunk_nodes):
                cpu._deadline(self.deadline);sl=slice(j,j+self.chunk_nodes);x=self.device_arrays['xq'][sl];g=self.device_arrays['gq'][sl];w=self.device_arrays['wq'][sl]
                wb=w*torch.exp(a+d*x-eta*g);terms.append(float(torch.sum(wb).item()));dterms.append(float(torch.sum(wb*x).item()));gterms.append(float(torch.sum(wb*g).item()))
            if eta==0:jb,jd=background_integrals(a,d,self.law.horizon,self.law.coordinate)
            else:jb,jd=background_integrals(a,d,self.quiet_duration,self.law.coordinate);jb+=math.fsum(terms);jd+=math.fsum(dterms)
            value=math.fsum((jb,n*integral,-event));_finite([value],'Nonfinite CUDA likelihood')
            if gradient:
                result={0:jb-pa,1:jd-pd,3:pg-math.fsum(gterms)}
                if 2 in free:
                    lr=float(torch.logsumexp(h-ll,dim=0).item())
                    if lr>math.log(np.finfo(float).max):raise UnrepresentableDerivative('True n-boundary derivative exceeds FP64')
                    result[2]=integral-(math.exp(lr) if lr!=-math.inf else 0.)
                grad=np.array([result[i] for i in free]);_finite(grad,'Requested CUDA derivative is not representable')
        self.context.memory_metrics();cpu._deadline(self.deadline)
        return (value,grad) if gradient else value

    def contrast(self,null,alternative):
        """Stable full GLR, with both background AND trigger compensator changes."""
        p0=np.asarray(null['pars']);p1=np.asarray(alternative['pars']);k0=KernelSpec(**null['trigger']);k1=KernelSpec(**alternative['trigger']);r0=null['ratio'];r1=alternative['ratio']
        events=math.fsum(self.event_logs(p1,k1,r1)-self.event_logs(p0,k0,r0))
        if p0[3]!=0:raise ValueError('Contrast requires eta0 null')
        quiet0=background_integrals(p0[0],p0[1],self.quiet_duration,self.law.coordinate)[0];quiet1=background_integrals(p1[0],p1[1],self.quiet_duration,self.law.coordinate)[0]
        difference=self.wq*(-np.exp(p0[0]+p0[1]*self.xq)*np.expm1((p1[0]-p0[0])+(p1[1]-p0[1])*self.xq-p1[3]*self.gq))
        bg=math.fsum((quiet0,-quiet1,math.fsum(difference)))
        _,_,c0=self.trigger_features(k0,r0);_,_,c1=self.trigger_features(k1,r1)
        trigger=math.fsum((p0[2]-p1[2])*c0) if k0.key==k1.key and r0==r1 else math.fsum(p0[2]*c0-p1[2]*c1)
        direct=2*math.fsum((events,bg,trigger));control=2*(self.objective(p0,k0,r0,False)-self.objective(p1,k1,r1,False))
        _finite([direct,control],'Unrepresentable GLR')
        return dict(statistic=direct,event_difference=events,background_compensator_difference=bg,trigger_compensator_difference=trigger,nll_subtraction_control=control,direct_minus_subtraction=direct-control)
