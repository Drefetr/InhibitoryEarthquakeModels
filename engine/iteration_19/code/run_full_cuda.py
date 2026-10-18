"""User-launched, completion-checkpointed CUDA scoring of the fixed 2149 histories.

Imports do not initialize Torch/CUDA. No simulation, fit or calibration is run.
"""
from pathlib import Path
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse,ctypes,importlib.util,json,math,multiprocessing as mp,queue,sys,time,traceback,uuid
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];AREA=ROOT/'iteration_19'
sys.path.insert(0,str(ROOT/'iteration_6/code'))
from checkpoint_io import RunLock,ChildJob,read_json,read_sealed,sealed_json,sha256,canonical_hash,atomic_bytes,utc_now

PROTOCOL_SHA='dc3369af795b8116cea5a1d4603f365affe5111b0100639da1090af7e88cd754'
INVENTORY_SHA='2d7ef9883147fa1d6bed60b007eecf5dd5b97ff16a4f11cfb75625be29fdad60'
BACKEND_SHA='be7ff7938596c0f5c3bc40962986e2ae511a0a9c1b05bc1d8d7b99008fa86505'
HARNESS_SHA='ccea681d683dcb4cd6fc39aabf109a84cc0cce40d0d4e75e9b6fdc89384a625b'
CONFIG_SHA='51e79769b121446a00e883db270da65efe3f656cd0d0d4c771e211c76311a8dd'
IMPORT_IDS=('null_0000','null_0841','null_1292','new_r000','new_r076')
SOURCE_NAMES=('iteration_19/code/run_full_cuda.py','iteration_19/code/analyse_full.py','iteration_19/run_full_evaluation.ps1','iteration_17/code/analyse_oracle_v1_1.py','iteration_18/code/cuda_oracle.py','iteration_18/code/benchmark_cuda.py',
              'iteration_17/code/oracle_likelihood.py','iteration_5/code/kernels.py','iteration_6/code/checkpoint_io.py')
PLAN_KEYS=('schema_version','config_path','config_sha256','config','sources','protocol','ids','entries','imports','law_sha256')
TERMINAL=('complete','refinement_failed','time_cap','error')


def require(condition,message):
    if not condition:raise ValueError(message)


def expected_ids():return [f'null_{i:04d}' for i in range(1999)]+[f'new_r{i:03d}' for i in range(150)]


def validate_config(c):
    require(c['phase17_protocol']['sha256']==PROTOCOL_SHA and c['inventory']['sha256']==INVENTORY_SHA,'Exact original science/inventory required')
    limits=dict(total_wall_seconds=14400,gpu_case_seconds=600,context_setup_seconds=60,cleanup_reserve_seconds=15,
                native_threads_per_cpu_worker=1,persistent_gpu_workers=1,maximum_GPU_allocated_MiB=1024,maximum_query_nodes_per_block=32768)
    require(c['limits']==limits,'Frozen full-run resource policy differs')
    n=c['numerical']
    require(n['dtype']=='float64' and n['no_TF32'] and n['no_mixed_precision'] and n['orders']==[16,32] and
            n['refinement_tolerance']==1e-6 and n['order64_only_on_failed_refinement'],'Numerical rules differ')
    require(c['case_ids']==expected_ids() and c['reuse']['ids']==list(IMPORT_IDS) and c['reuse']['source_repeat']==0,'Cohort or deterministic reuse differs')


def _benchmark_api():
    name='_iteration18_fullrun_validation_authority'
    if name not in sys.modules:
        p=ROOT/'iteration_18/code/benchmark_cuda.py';require(sha256(p)==HARNESS_SHA,'Benchmark validation source changed')
        spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def validate_payload(payload,entry,config):
    _benchmark_api().validate_payload(payload,'cuda',entry,config)
    require(payload['catalogue_cache_reused'] is False and payload['preparation_shared_across_orders'] is True,'Unexpected history cache')
    require(payload['cuda_metrics']['max_allocated_bytes']==1024*2**20 and payload['cuda_metrics']['chunk_nodes']==32768,'Top-level allocation policy differs')
    require(set(payload['evaluations'])==({'16','32','64'} if payload['selected_order']==64 else {'16','32'}),'Extra/missing quadrature orders')
    for value in payload['evaluations'].values():
        require(value['method']=='cuda_fp64_gauss' and value['maximum_working_query_nodes']<=32768,'Backend method/block differs')


def identity(plan,entry):
    return dict(plan_id=plan['plan_id'],id=entry['id'],seed=entry['seed'],role=entry['role'],arm=entry.get('arm'),replication=entry['replication'],
                catalogue_sha256=entry['files']['catalogue.npz']['sha256'],clock_mode=entry['clock_mode'],
                inventory_entry_sha256=canonical_hash(entry),law_sha256=plan['law_sha256'])


def case_path(run,entry):return Path(run)/'cases'/(entry['id']+'.json')


def cached(plan,entry,run,check_catalogue=True):
    path=case_path(run,entry)
    if not path.exists():return None
    value=read_sealed(path)
    require(all(value.get(k)==v for k,v in identity(plan,entry).items()) and value['status'] in TERMINAL,'Saved score authority differs '+entry['id'])
    if check_catalogue:require(sha256(ROOT/entry['files']['catalogue.npz']['path'])==value['catalogue_sha256'],'Cached catalogue bytes changed')
    if value['status']=='complete':
        validate_payload(value['payload'],entry,plan['config'])
        require(value['log_lr']==value['payload']['result']['log_lr'] and value['refinement_error']==value['payload']['selected_refinement_difference'],'Saved aliases differ')
    return value


def commit(plan,entry,run,**fields):
    old=cached(plan,entry,run)
    if old is not None:return old
    value=dict(**identity(plan,entry),updated_utc=utc_now(),**fields)
    if value['status']=='complete':validate_payload(value['payload'],entry,plan['config'])
    sealed_json(case_path(run,entry),value)
    return cached(plan,entry,run)


def _reference(root_reference):
    path=(ROOT/root_reference['path']).resolve();require(path.is_relative_to(ROOT),'Authority outside project')
    require(sha256(path)==root_reference['sha256'],'Authority bytes changed '+root_reference['path'])
    return path


def prepare(config_path,run):
    run=Path(run).resolve();config_path=Path(config_path).resolve()
    require(run.is_relative_to(AREA/'results'),'Output outside iteration19/results')
    require(sha256(config_path)==CONFIG_SHA,'Exact frozen resource amendment required')
    c=read_json(config_path);validate_config(c)
    authorities={k:_reference(c[k]) for k in ('phase17_protocol','inventory','prior_seal','backend_seal','backend','statistical_authority','benchmark_plan','benchmark_summary','benchmark_runtime_freeze')}
    protocol=read_json(authorities['phase17_protocol']);inventory=read_sealed(authorities['inventory'])
    cohort=protocol['phase1']['cohorts'];ids=cohort['reference_null_ids']+cohort['heldout_check_null_ids']+cohort['alternative_ids']
    require(ids==expected_ids() and cohort['reference_null_ids']==ids[:999] and cohort['heldout_check_null_ids']==ids[999:1999],'Exact original cohort/split required')
    by={e['id']:e for e in inventory['entries']};require(inventory['complete'] and len(by)==len(inventory['entries']) and all(i in by for i in ids),'Invalid inventory')
    entries=[by[i] for i in ids]
    for entry in protocol['authorities']:_reference(entry)
    prior=read_json(authorities['prior_seal'])
    for entry in prior['files']:
        p=ROOT/entry['path'];require(p.stat().st_size==entry['bytes'] and sha256(p)==entry['sha256'],'Prior sealed bytes changed')
    backend_seal=read_json(authorities['backend_seal'])
    for entry in backend_seal['files']:
        p=ROOT/entry['path'];require(p.stat().st_size==entry['bytes'] and sha256(p)==entry['sha256'],'Sealed backend bytes changed')
    benchmark_plan=read_sealed(authorities['benchmark_plan']);benchmark_summary=read_sealed(authorities['benchmark_summary'])
    require(benchmark_summary['backend_accepted'] and benchmark_summary['all_mandatory_tasks_complete'] and benchmark_summary['sessions_closed'] and
            benchmark_summary['plan_id']==benchmark_plan['plan_id'],'Benchmark not complete/accepted')
    for name,digest in benchmark_plan['sources'].items():require(sha256(ROOT/name)==digest,'Benchmark source changed')
    runtime=read_json(authorities['benchmark_runtime_freeze'])
    require(runtime['plan_id']==benchmark_plan['plan_id'],'Benchmark runtime plan differs')
    for item in runtime['sources']:require(sha256(ROOT/item['path'])==item['sha256'],'Benchmark freeze artifact changed')
    require(sha256(ROOT/'iteration_18/code/cuda_oracle.py')==BACKEND_SHA and sha256(ROOT/'iteration_18/code/benchmark_cuda.py')==HARNESS_SHA,'Validated backend changed')
    for e in entries:
        p=ROOT/e['files']['catalogue.npz']['path'];require(sha256(p)==e['files']['catalogue.npz']['sha256'],'Catalogue changed '+e['id'])
        require(e['clock_mode']==('paired_clock' if e['id']=='null_0841' else 'original_projected_clock'),'Clock convention changed')
    imports={}
    for id in IMPORT_IDS:
        path=authorities['benchmark_plan'].parent/'tasks'/f'gpu_{id}_r0.json';value=read_sealed(path);e=by[id]
        require(value['status']=='complete' and value['plan_id']==benchmark_plan['plan_id'] and value['id']==id and value['seed']==e['seed'] and
                value['catalogue_sha256']==e['files']['catalogue.npz']['sha256'] and value['clock_mode']==e['clock_mode'] and
                value['task']['task_id']==f'gpu_{id}_r0' and value['task']['repeat']==0,'Invalid deterministic repeat0 import')
        validate_payload(value['payload'],e,c)
        imports[id]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha256(path),source_plan_id=value['plan_id'],source_task_id=value['task']['task_id'],backend='cuda',repeat=0)
    authority=dict(schema_version=1,config_path=config_path.relative_to(ROOT).as_posix(),config_sha256=sha256(config_path),config=c,
                   sources={p:sha256(ROOT/p) for p in SOURCE_NAMES},protocol=protocol,ids=ids,entries=entries,imports=imports,
                   law_sha256=canonical_hash(protocol['phase1']['law']))
    plan=dict(authority,plan_id=canonical_hash(authority));run.mkdir(parents=True,exist_ok=True)
    with RunLock(run/'runner.lock'):
        if (run/'plan.json').exists():
            old=read_sealed(run/'plan.json');require(old['plan_id']==plan['plan_id'] and canonical_hash({k:old[k] for k in PLAN_KEYS})==plan['plan_id'],'Prepared authority changed');plan=old
        else:
            sealed_json(run/'plan.json',dict(plan,prepared_utc=utc_now()))
            for p in authority['sources']:atomic_bytes(run/'source_snapshots'/p,(ROOT/p).read_bytes())
            plan=read_sealed(run/'plan.json')
        # New-plan identities wrap the exact saved payload, never the fastest repeat.
        for id,provenance in imports.items():
            value=read_sealed(ROOT/provenance['path']);payload=value['payload'];e=by[id]
            imported=commit(plan,e,run,status='complete',payload=payload,log_lr=payload['result']['log_lr'],nll0=payload['result']['nll0'],nll1=payload['result']['nll1'],
                   refinement_error=payload['selected_refinement_difference'],origin='imported_iteration18_gpu_repeat0',import_provenance=provenance,
                   elapsed_seconds=0.,source_elapsed_seconds=value['worker_end_to_end_seconds'],cpu_seconds=0.)
            require(imported['payload']==payload and imported.get('import_provenance')==provenance and imported.get('origin')=='imported_iteration18_gpu_repeat0','Cached deterministic import differs')
        if not (run/'sessions.json').exists():
            states={e['id']:'complete' if e['id'] in imports else 'pending' for e in entries}
            write_summary(plan,run,states,[],'prepared',[])
    return plan


def process_identity(pid):
    """PID plus creation token; no Torch and no unsafe PID-reuse assumption."""
    if os.name=='nt':
        from ctypes import wintypes as w
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE
        kernel.GetProcessTimes.argtypes=[w.HANDLE,ctypes.POINTER(w.FILETIME),ctypes.POINTER(w.FILETIME),ctypes.POINTER(w.FILETIME),ctypes.POINTER(w.FILETIME)];kernel.GetProcessTimes.restype=w.BOOL
        kernel.GetExitCodeProcess.argtypes=[w.HANDLE,ctypes.POINTER(w.DWORD)];kernel.GetExitCodeProcess.restype=w.BOOL
        kernel.CloseHandle.argtypes=[w.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:
            if ctypes.get_last_error()==87:return None
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            code=w.DWORD();require(kernel.GetExitCodeProcess(handle,ctypes.byref(code)),'Cannot query old worker')
            if code.value!=259:return None
            creation=w.FILETIME();exit=w.FILETIME();user=w.FILETIME();system=w.FILETIME()
            require(kernel.GetProcessTimes(handle,ctypes.byref(creation),ctypes.byref(exit),ctypes.byref(system),ctypes.byref(user)),'Cannot identify old worker')
            return dict(pid=int(pid),creation_token=str((creation.dwHighDateTime<<32)|creation.dwLowDateTime))
        finally:kernel.CloseHandle(handle)
    try:
        text=Path(f'/proc/{pid}/stat').read_text();fields=text[text.rfind(')')+2:].split()
        if fields[0]=='Z':return None
        return dict(pid=int(pid),creation_token=fields[19])
    except FileNotFoundError:return None


def ensure_old_worker_dead(session,timeout=5.):
    old=session.get('worker_identity')
    if old is None:return
    began=time.perf_counter()
    while process_identity(old['pid'])==old:
        require(time.perf_counter()-began<timeout,'Previous owned CUDA worker still alive; refusing checkpoint adoption')
        time.sleep(.05)


def attempt_path(run,id,attempt_id,kind):return Path(run)/'attempts'/id/(attempt_id+'.'+kind+'.json')


def immutable_json(path,value):
    require(not Path(path).exists(),'Refusing attempt/event overwrite '+str(path));sealed_json(path,value)


def finish_attempt(plan,run,entry,attempt,status,**details):
    path=attempt_path(run,entry['id'],attempt['attempt_id'],'finished')
    if path.exists():
        value=read_sealed(path);require(value['plan_id']==plan['plan_id'] and value['attempt_id']==attempt['attempt_id'],'Attempt lineage differs');return value
    value=dict(**identity(plan,entry),attempt_id=attempt['attempt_id'],session_id=attempt['session_id'],status=status,finished_utc=utc_now(),**details)
    immutable_json(path,value);return read_sealed(path)


def reconcile_sessions(plan,run,sessions):
    """Only called under OS lock, before reading/adopting case checkpoints."""
    by={e['id']:e for e in plan['entries']};limits=plan['config']['limits']
    for session in sessions:
        if session['closed']:continue
        ensure_old_worker_dead(session)
        active=session.get('active_attempt');bound=float(active['cap_seconds']) if active else (limits['context_setup_seconds'] if session.get('stage')=='setup' else 0.)
        prior_other=sum(s['elapsed_seconds'] for s in sessions if s is not session)
        observed=float(session['elapsed_seconds']);remaining=max(0.,limits['total_wall_seconds']-prior_other-observed)
        assumed=min(remaining,bound+limits['cleanup_reserve_seconds'])
        session.update(elapsed_seconds=observed+assumed,closed=True,ended_utc=utc_now(),reason='unclean_bounded_inflight_reservation',
                       measured_prefix_seconds=observed,assumed_unobserved_seconds=assumed,assumption_scope='At most declared active attempt/setup bound plus cleanup; no UTC downtime',active_attempt=None,stage='closed')
        if active:
            e=by[active['id']];saved=cached(plan,e,run)
            finish_attempt(plan,run,e,active,saved['status'] if saved else 'interrupted',
                reason='unclean_parent_recovery',case_sha256=sha256(case_path(run,e)) if saved else None,elapsed_seconds=None,assumed_unobserved_budget_seconds=assumed)
    sealed_json(Path(run)/'sessions.json',dict(plan_id=plan['plan_id'],sessions=sessions))


def write_summary(plan,run,states,sessions,reason,active,write_index=True):
    counts={s:sum(v==s for v in states.values()) for s in sorted(set(states.values()))}
    complete=counts.get('complete',0);terminal=sum(v in TERMINAL for v in states.values())
    summary=dict(plan_id=plan['plan_id'],updated_utc=utc_now(),reason=reason,total_planned=len(plan['ids']),counts=counts,completed=terminal,complete_scores=complete,
                 completed_validated=complete,pending=counts.get('pending',0),all_primary_complete=complete==len(plan['ids']),terminal_complete=terminal==len(plan['ids']),
                 active=any(not s['closed'] for s in sessions),active_ids=active,sessions_closed=all(s['closed'] for s in sessions),charged_seconds=sum(s['elapsed_seconds'] for s in sessions),
                 full_campaign_scope='Fixed saved histories only; no simulation, fitting, bootstrap or downloads',
                 work_scope='Cumulative owned parent wall. Unclean recovery adds explicit bounded unobserved attempt/setup allowance, not measured time or downtime.')
    sealed_json(Path(run)/'summary.json',summary)
    sealed_json(Path(run)/'progress.json',{k:v for k,v in summary.items() if k not in ('counts',)})
    if write_index:sealed_json(Path(run)/'case_index.json',dict(plan_id=plan['plan_id'],states=states))
    return summary


def score_case(plan,entry,context,deadline):
    import numpy as np
    from cuda_oracle import law_from_setting,score_refined_cuda
    began=time.perf_counter();cpu_began=time.process_time();status='error';payload=None;error=None
    try:
        path=ROOT/entry['files']['catalogue.npz']['path'];require(sha256(path)==entry['files']['catalogue.npz']['sha256'],'Input catalogue changed')
        with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
        spec=plan['protocol']['phase1']['law'];law=law_from_setting(dict(beta=spec['beta'],horizon=spec['horizon_days'],coordinate_horizon=spec['coordinate_horizon_days'],generating_fit=spec['alternative_generating_fit']))
        payload=score_refined_cuda(arrays,law,clock_mode=entry['clock_mode'],eta=spec['alternative_eta'],tolerance=1e-6,deadline=deadline,chunk_nodes=32768,context=context)
        status='complete' if payload['complete'] and payload['refinement_passed'] else 'refinement_failed'
        if status=='complete':validate_payload(payload,entry,plan['config'])
    except TimeoutError as exc:status='time_cap';error=f'{type(exc).__name__}: {exc}'
    except Exception as exc:status='error';error=f'{type(exc).__name__}: {exc}'
    if status in ('error','time_cap'):payload=None
    fields=dict(status=status,payload=payload,error=error,elapsed_seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu_began,
                timing_scope='Fresh case input load/hash, CPU preparation, CUDA transfer/compute/synchronize/scalar return and validation; commit and IPC excluded')
    if payload is not None:fields.update(log_lr=payload['result']['log_lr'],nll0=payload['result']['nll0'],nll1=payload['result']['nll1'],refinement_error=payload['selected_refinement_difference'])
    return fields


def worker(plan,run,ready,inbox,outbox):
    began=time.perf_counter()
    while not ready.wait(.1):
        parent=mp.parent_process()
        if parent is None or not parent.is_alive() or time.perf_counter()-began>15:return
    context=None
    try:
        sys.path.insert(0,str(ROOT/'iteration_18/code'))
        from cuda_oracle import CudaOracleContext
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            context=CudaOracleContext(max_allocated_bytes=1024*2**20);warmup=context.warmup()
            outbox.put(dict(event='ready',pid=os.getpid(),warmup=warmup,context_metadata=context.metadata))
            while True:
                item=inbox.get()
                if item is None:return
                entry,attempt,deadline=item;outbox.put(dict(event='started',id=entry['id'],pid=os.getpid()))
                fields=score_case(plan,entry,context,deadline)
                value=commit(plan,entry,run,**fields,origin='iteration19_cuda_evaluation',attempt_id=attempt['attempt_id'])
                outbox.put(dict(event='finished',id=entry['id'],pid=os.getpid(),status=value['status'],elapsed_seconds=value['elapsed_seconds'],process_cpu_seconds=time.process_time()))
    except BaseException as exc:
        outbox.put(dict(event='worker_error',pid=os.getpid(),error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
    finally:
        if context is not None:context.close()


def stop_child(handle,graceful=False):
    p,job,inbox=handle
    try:
        if graceful and p.is_alive():inbox.put(None);p.join(2)
        if p.is_alive():p.terminate()
        p.join(2)
    finally:job.close()
    if p.is_alive():p.kill();p.join(2)
    require(not p.is_alive(),'Owned child remains alive; refusing parent checkpoint adoption')


def run(plan,run,*,worker_target=None,pause_after=None,invocation_began=None):
    """Only explicit CLI --resume calls this; injectable worker is for no-science fixtures."""
    run=Path(run);began=time.perf_counter() if invocation_began is None else invocation_began;cpu_began=time.process_time()
    with RunLock(run/'runner.lock'):
        require(canonical_hash({k:plan[k] for k in PLAN_KEYS})==plan['plan_id'],'Plan content differs')
        require(plan['ids']==expected_ids(),'Exact fixed cohort required')
        for p,h in plan['sources'].items():require(sha256(ROOT/p)==h,'Runtime source changed')
        require(plan['config_sha256']==CONFIG_SHA and sha256(ROOT/plan['config_path'])==plan['config_sha256'] and read_json(ROOT/plan['config_path'])==plan['config'],'Configuration bytes/content changed')
        if (run/'sessions.json').exists():
            ledger=read_sealed(run/'sessions.json');require(ledger['plan_id']==plan['plan_id'],'Session ledger belongs to another plan');sessions=ledger['sessions']
        else:sessions=[]
        reconcile_sessions(plan,run,sessions)
        by={e['id']:e for e in plan['entries']};states={i:(cached(plan,by[i],run) or {}).get('status','pending') for i in plan['ids']}
        pending=[i for i in plan['ids'] if states[i]=='pending'];limits=plan['config']['limits'];prior=sum(s['elapsed_seconds'] for s in sessions)
        # No context, child or Torch import for an already terminal bank.
        if not pending:
            summary=write_summary(plan,run,states,sessions,'complete' if all(v=='complete' for v in states.values()) else 'completed_with_unknowns',[])
            return 0 if summary['all_primary_complete'] else 2
        session_id=utc_now().replace(':','').replace('.','').replace('+','_')+'_'+uuid.uuid4().hex[:8]
        session=dict(session_id=session_id,started_utc=utc_now(),closed=False,elapsed_seconds=time.perf_counter()-began,stage='idle',active_attempt=None,worker_identity=None)
        sessions.append(session);event_number=0;handle=None;reason='complete';completed_here=0;original_exception=None;cleanup_errors=[]
        work_deadline=began+max(0.,limits['total_wall_seconds']-prior-limits['cleanup_reserve_seconds'])
        def save():
            session.update(elapsed_seconds=time.perf_counter()-began,parent_cpu_seconds=time.process_time()-cpu_began)
            sealed_json(run/'sessions.json',dict(plan_id=plan['plan_id'],sessions=sessions))
            active=[session['active_attempt']['id']] if session.get('active_attempt') else []
            return write_summary(plan,run,states,sessions,reason if session['closed'] else 'active',active,write_index=False)
        def event(kind,**details):
            nonlocal event_number
            event_number+=1;immutable_json(run/'events'/session_id/f'{event_number:06d}.json',dict(plan_id=plan['plan_id'],session_id=session_id,event=kind,utc=utc_now(),elapsed_seconds=time.perf_counter()-began,**details))
        try:
            save();event('session_started',pending=len(pending),prior_charged_seconds=prior)
            if (run/'STOP').exists():reason='stop_requested'
            elif pause_after==0:reason='requested_case_boundary_pause'
            elif time.perf_counter()>=work_deadline:reason='global_budget'
            else:
                ctx=mp.get_context('spawn');ready=ctx.Event();inbox=ctx.Queue();outbox=ctx.Queue();p=ctx.Process(target=worker if worker_target is None else worker_target,args=(plan,str(run),ready,inbox,outbox));job=ChildJob()
                handle=(p,job,inbox);session['stage']='setup';save()
                try:
                    p.start();job.assign(p);session['worker_identity']=process_identity(p.pid);require(session['worker_identity'] is not None,'Cannot identify owned worker');save();ready.set();event('worker_started',worker_identity=session['worker_identity'])
                except BaseException:
                    if p.pid is not None:stop_child(handle)
                    else:job.close()
                    handle=None;raise
                setup_began=time.perf_counter();is_ready=False
                while pending or session['active_attempt']:
                    now=time.perf_counter()
                    if (run/'STOP').exists() and session.get('active_attempt') is None:reason='stop_requested';break
                    if now>=work_deadline:reason='global_budget';break
                    active=session.get('active_attempt')
                    if not is_ready and now-setup_began>=limits['context_setup_seconds']:reason='context_setup_time_cap';break
                    if active and now>=active['deadline']:
                        reason='case_time_cap';break
                    if is_ready and active is None and pending:
                        if pause_after is not None and completed_here>=pause_after:reason='requested_case_boundary_pause';break
                        id=pending.pop(0);cap=min(float(limits['gpu_case_seconds']),max(0.,work_deadline-time.perf_counter()))
                        if cap<=0:reason='global_budget';break
                        attempt=dict(id=id,attempt_id=session_id+'_'+f'{completed_here+1:04d}',session_id=session_id,cap_seconds=cap,deadline=time.perf_counter()+cap)
                        session.update(active_attempt=attempt,stage='case')
                        immutable_json(attempt_path(run,id,attempt['attempt_id'],'started'),{**identity(plan,by[id]),**attempt,'started_utc':utc_now(),'worker_identity':session['worker_identity']})
                        save();event('dispatched',id=id,attempt_id=attempt['attempt_id'],cap_seconds=cap);inbox.put((by[id],attempt,attempt['deadline']))
                    try:message=outbox.get(timeout=.1)
                    except queue.Empty:message=None
                    if message:
                        if message['event']=='ready':
                            is_ready=True;session.update(stage='idle',warmup=message['warmup'],context_metadata=message['context_metadata']);event('worker_ready');save()
                        elif message['event']=='started':event('started',id=message['id'])
                        elif message['event']=='finished':
                            active=session['active_attempt'];require(active is not None and active['id']==message['id'],'Worker completion differs from active attempt')
                            value=cached(plan,by[message['id']],run);require(value is not None,'Worker completion missing durable case')
                            states[message['id']]=value['status'];finish_attempt(plan,run,by[message['id']],active,value['status'],case_sha256=sha256(case_path(run,by[message['id']])),elapsed_seconds=value['elapsed_seconds'],reason='worker_completed')
                            completed_here+=1;session.update(active_attempt=None,stage='idle',worker_cpu_seconds_observed=message.get('process_cpu_seconds'))
                            event('finished',id=message['id'],status=value['status'],case_elapsed_seconds=value['elapsed_seconds']);save()
                            print(json.dumps(dict(id=message['id'],status=value['status'],completed=sum(v in TERMINAL for v in states.values()),complete_scores=sum(v=='complete' for v in states.values()),planned=len(plan['ids']))),flush=True)
                            if value['status']!='complete':reason='case_failure';break
                            if (run/'STOP').exists():reason='stop_requested';break
                        elif message['event']=='worker_error':event('worker_error',error=message['error']);reason='worker_error';break
                    if not p.is_alive():reason='unexpected_worker_exit';break
        except KeyboardInterrupt:reason='keyboard_interrupt'
        except BaseException as exc:reason='parent_error:'+type(exc).__name__;original_exception=exc
        finally:
            # Establish owned child death before adopting or writing any active-case outcome.
            if handle is not None:
                try:stop_child(handle,graceful=session.get('active_attempt') is None)
                except BaseException as exc:cleanup_errors.append(exc)
            if not cleanup_errors:
                try:
                    active=session.get('active_attempt')
                    if active:
                        e=by[active['id']];value=cached(plan,e,run)
                        if value is None and reason=='case_time_cap':value=commit(plan,e,run,status='time_cap',payload=None,error='Parent hard per-case cap',elapsed_seconds=active['cap_seconds'],cpu_seconds=None,origin='iteration19_parent_cap',attempt_id=active['attempt_id'])
                        if value is not None:states[e['id']]=value['status']
                        finish_attempt(plan,run,e,active,value['status'] if value else 'interrupted',case_sha256=sha256(case_path(run,e)) if value else None,elapsed_seconds=None,reason=reason)
                        event('interrupted' if value is None else 'late_commit_adopted',id=e['id'],reason=reason,status=value['status'] if value else 'pending')
                    session.update(active_attempt=None,stage='closed')
                    if handle is not None:event('worker_exit',pid=handle[0].pid,exitcode=handle[0].exitcode)
                except BaseException as exc:cleanup_errors.append(exc)
            if cleanup_errors:
                session['reason']='cleanup_unresolved';save()
            else:
                session.update(closed=True,ended_utc=utc_now(),reason=reason);save();event('session_closed',reason=reason)
                sealed_json(run/'case_index.json',dict(plan_id=plan['plan_id'],states=states))
            if cleanup_errors:
                if original_exception is not None:original_exception.add_note('Cleanup/reporting errors: '+repr(cleanup_errors))
                else:raise RuntimeError('Owned cleanup/reporting incomplete') from cleanup_errors[0]
        if original_exception is not None:raise original_exception
        if all(v=='complete' for v in states.values()):return 0
        return 130 if reason in ('stop_requested','keyboard_interrupt','requested_case_boundary_pause') else 2


def main():
    began=time.perf_counter();parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=AREA/'configs/full_cuda_frozen.json')
    parser.add_argument('--run-dir',type=Path,default=AREA/'results/full_cuda');parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--status',action='store_true')
    parser.add_argument('--pause-after-completed',type=int,help='Engineering/manual graceful pause after this many newly completed cases')
    args=parser.parse_args();run=args.run_dir.resolve();require(run.is_relative_to(AREA/'results'),'Output outside iteration19/results')
    if args.status:
        if (run/'summary.json').exists():value=read_sealed(run/'summary.json')
        else:value=dict(prepared=(run/'plan.json').exists(),active=False,total_planned=2149,completed_validated=0,pending=2149)
        print(json.dumps(value,indent=2));return 0
    if args.pause_after_completed is not None:require(args.pause_after_completed>=0,'Pause count must be nonnegative')
    plan=prepare(args.config,run)
    if args.prepare_only:print(json.dumps(dict(prepared=True,plan_id=plan['plan_id'],imported_completed=len(IMPORT_IDS),planned=2149,new_scoring_launched=False)));return 0
    require(args.resume,'Explicit --resume required to evaluate saved histories')
    return run_full(plan,run,pause_after=args.pause_after_completed,invocation_began=began)


# Separate alias avoids shadowing run() with CLI path locals and is an easy fixture patch point.
run_full=run
if __name__=='__main__':raise SystemExit(main())
