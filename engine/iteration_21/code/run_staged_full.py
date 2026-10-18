"""User-launched, serial six-call staged evaluation of the fixed 2149 histories.

Preparation/status are metadata-only. No fit runs without explicit --resume and
the separate full-run source freeze. The scientific worker is imported intact.
"""
from pathlib import Path
import argparse, collections, multiprocessing as mp, queue, sys, time, uuid
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];AREA=ROOT/'iteration_21'
sys.path.insert(0,str(AREA/'code'))
import benchmark_staged as bench
from benchmark_staged import (RunLock,ChildJob,atomic_bytes,canonical_hash,read_json,read_sealed,sealed_json,sha256,utc_now,
                              require,immutable,identity,case_path,method_path,read_stage,cached,assemble_validated_case,lifetime_api)
BENCHMARK_SHA='6887cc4df36b4bf1cfa6f9cd8918e8cc6f25d50df7f24d447522e14eec04d79f'
require(sha256(Path(bench.__file__))==BENCHMARK_SHA,'Frozen staged worker/helper changed')
CONFIG_SHA='91d98445f0b722bf960fa0ea331f10a592547f23ae35a74723ad57cbd7a77d65'
PLAN_KEYS=list(bench.PLAN_KEYS)
LOCK_NAME='full.lock'
SOURCES=list(bench.SOURCES)+['iteration_21/code/run_staged_full.py','iteration_21/run_full_staged.ps1',
                           'iteration_21/code/analyse_full_staged.py','iteration_21/code/analyse_staged.py']
TERMINAL={'complete','failed','error','time_cap'}


def expected_ids():
    return [f'null_{i:04d}' for i in range(1999)]+[f'new_r{i:03d}' for i in range(150)]


def validate_config(c):
    x=c['execution']
    require(c['status']=='frozen_manual_staged_full_evaluation_before_execution','Full protocol is not frozen')
    require(x['ids']==expected_ids(),'Exact fixed 2149 identities/order required')
    for k,v in dict(whole_parent_wall_seconds=28800,per_history_total_wall_seconds=600,setup_wall_seconds=60,cleanup_reserve_seconds=15,GPU_workers=1,native_threads=1,
                    first_unresolved_case_stops=True,analysis_requires_all2149_complete=True,fresh_all_histories=True,import_profile_fits=False).items():
        require(x.get(k)==v,'Full resource/procedure field differs: '+k)
    f=c['future_inference']
    require(f['reference_ids']+f['heldout_null_ids']+f['alternative_ids']==expected_ids(),'Reference/check/alternative split differs')


def write_index(plan,run,states,methods):
    sealed_json(run/'case_index.json',dict(plan_id=plan['plan_id'],states=states,method_states=methods,updated_utc=utc_now()))


def write_summary(plan,run,states,methods,sessions,reason,*,write_full_index=False,state_counts=None,method_counts=None):
    counts=dict(state_counts if state_counts is not None else collections.Counter(states.values()));counts={k:v for k,v in counts.items() if v};complete=counts.get('complete',0)==len(plan['ids'])
    active=[s['active_id'] for s in sessions if not s['closed'] and s.get('active_id')]
    value=dict(plan_id=plan['plan_id'],updated_utc=utc_now(),reason=reason,total_planned=len(plan['ids']),counts=counts,
               completed_validated=counts.get('complete',0),pending=counts.get('pending',0),all_primary_complete=complete,
               terminal_complete=sum(counts.get(k,0) for k in TERMINAL)==len(plan['ids']),active=any(not s['closed'] for s in sessions),active_ids=active,
               sessions_closed=all(s['closed'] for s in sessions),charged_seconds=sum(s['elapsed_seconds'] for s in sessions),
               method_counts={m:{k:v for k,v in values.items() if v} for m,values in method_counts.items()} if method_counts is not None else {m:dict(collections.Counter(v[m] for v in methods.values())) for m in ('known_known','background_estimated')},
               no_new_simulations=True,no_fit_imports=True,analysis_requires_all2149_complete=True)
    sealed_json(run/'summary.json',value)
    if write_full_index:write_index(plan,run,states,methods)
    return value


def prepare(config_path,run):
    config_path=Path(config_path).resolve();run=Path(run).resolve();c=read_json(config_path)
    require(sha256(config_path)==CONFIG_SHA,'Full configuration bytes differ');validate_config(c)
    for ref in c['authorities']:require(sha256(ROOT/ref['path'])==ref['sha256'],'Prior authority changed: '+ref['path'])
    ref=next(r for r in c['authorities'] if '/inventory_' in r['path']);inventory=read_sealed(ROOT/ref['path']);by={e['id']:e for e in inventory['entries']}
    entries=[by[i] for i in expected_ids()]
    for e in entries:
        for ref in e['files'].values():require(sha256(ROOT/ref['path'])==ref['sha256'],'Saved authority changed: '+ref['path'])
    authority=dict(config_path=config_path.relative_to(ROOT).as_posix(),config_sha256=sha256(config_path),contract=c,sources={p:sha256(ROOT/p) for p in SOURCES},ids=expected_ids(),entries=entries)
    plan=dict(authority,plan_id=canonical_hash(authority));run.mkdir(parents=True,exist_ok=True)
    with RunLock(run/LOCK_NAME):
        if (run/'plan.json').exists():
            old=read_sealed(run/'plan.json');require(old['plan_id']==plan['plan_id']==canonical_hash({k:old[k] for k in PLAN_KEYS}),'Prepared full plan differs');plan=old
        else:
            immutable(run/'plan.json',dict(plan,prepared_utc=utc_now()))
            for p in SOURCES:atomic_bytes(run/'source_snapshots'/p,(ROOT/p).read_bytes())
            states={i:'pending' for i in plan['ids']};methods={i:dict(known_known='pending',background_estimated='pending') for i in plan['ids']}
            write_summary(plan,run,states,methods,[],'prepared',write_full_index=True)
    return plan


def update_case_state(plan,entry,run,states,methods,*,assemble=False):
    """Only this ID is read. Called once per ID at reconcile, then on its events."""
    if assemble:assemble_validated_case(plan,entry,run)
    value=cached(plan,entry,run);states[entry['id']]=value['status'] if value else 'pending'
    stage={}
    for m in ('known_known','background_estimated'):
        stage[m]='validated' if read_stage(plan,entry,run,m,'validated') else 'optimized_unvalidated' if read_stage(plan,entry,run,m,'optimized') else 'pending'
    methods[entry['id']]=stage
    return value


def reconcile_sessions(plan,run,sessions):
    life=lifetime_api();limit=plan['contract']['execution']['whole_parent_wall_seconds']
    by={e['id']:e for e in plan['entries']}
    for s in sessions:
        if s['closed']:continue
        life.ensure_old_worker_dead(s)
        prefix=s['elapsed_seconds'];other=sum(x['elapsed_seconds'] for x in sessions if x is not s)
        bound=600. if s.get('active_id') else 60. if s.get('stage')=='setup' else 0.
        assumed=min(max(0.,limit-other-prefix),bound+15.)
        if s.get('active_id'):
            entry=by[s['active_id']];value=cached(plan,entry,run);path=run/'attempts'/(s['active_attempt_id']+'.finished.json')
            if not path.exists():immutable(path,dict(plan_id=plan['plan_id'],id=entry['id'],status=value['status'] if value else 'interrupted',reason='unclean_parent_recovery',case_sha256=sha256(case_path(run,entry['id'])) if value else None))
        s.update(closed=True,elapsed_seconds=prefix+assumed,measured_prefix_seconds=prefix,assumed_unobserved_seconds=assumed,
                 reason='unclean_bounded_recovery',active_id=None,active_attempt_id=None,stage='closed',ended_utc=utc_now())
    sealed_json(run/'sessions.json',dict(plan_id=plan['plan_id'],sessions=sessions))


def worker(plan,run_name,ready,inbox,outbox):
    # Actual callable is unchanged; wrapper is spawn-importable under this script.
    return bench.worker(plan,run_name,ready,inbox,outbox)


def run_full(plan,run,start,worker_target=None):
    """No scientific work unless explicitly called by --resume after source freeze."""
    run=Path(run);life=lifetime_api();limits=plan['contract']['execution'];limit=float(limits['whole_parent_wall_seconds']);reserve=float(limits['cleanup_reserve_seconds'])
    reason='active';handle=None;active=None
    with RunLock(run/LOCK_NAME):
        require(canonical_hash({k:plan[k] for k in PLAN_KEYS})==plan['plan_id'],'Canonical full plan differs')
        require(plan['ids']==expected_ids(),'Fixed cohort differs');validate_config(plan['contract'])
        require(sha256(ROOT/plan['config_path'])==plan['config_sha256']==CONFIG_SHA and read_json(ROOT/plan['config_path'])==plan['contract'],'Full config changed')
        require(set(plan['sources'])==set(SOURCES),'Exact full runtime source set differs')
        for p,h in plan['sources'].items():require(sha256(ROOT/p)==h,'Runtime source changed: '+p)
        freeze=read_json(AREA/'logs/full_runtime_source_freeze.json')
        require(freeze['plan_id']==plan['plan_id'] and freeze['validation_passed'] is True,'Full runtime source freeze missing')
        ledger=read_sealed(run/'sessions.json') if (run/'sessions.json').exists() else dict(plan_id=plan['plan_id'],sessions=[])
        require(ledger['plan_id']==plan['plan_id'],'Session ledger belongs to another plan');sessions=ledger['sessions']
        reconcile_sessions(plan,run,sessions)
        by={e['id']:e for e in plan['entries']};states={};methods={}
        # Full validation exactly once per invocation. No full-bank payload read in save/progress.
        for e in plan['entries']:update_case_state(plan,e,run,states,methods,assemble=True)
        prior=sum(s['elapsed_seconds'] for s in sessions)
        if any(v in TERMINAL-{'complete'} for v in states.values()):
            write_summary(plan,run,states,methods,sessions,'retained_failed_case_blocks_resume',write_full_index=True);return 2
        pending=collections.deque(i for i in plan['ids'] if states[i]=='pending')
        if not pending:
            write_summary(plan,run,states,methods,sessions,'complete',write_full_index=True);return 0
        if prior+time.perf_counter()-start+reserve>=limit:
            write_summary(plan,run,states,methods,sessions,'global_budget',write_full_index=True);return 2
        sid=uuid.uuid4().hex;s=dict(session_id=sid,started_utc=utc_now(),closed=False,elapsed_seconds=time.perf_counter()-start,active_id=None,active_attempt_id=None,worker_identity=None,stage='setup')
        sessions.append(s);deadline=start+max(0.,limit-prior-reserve);event_number=0
        state_counts=collections.Counter(states.values());method_counts={m:collections.Counter(v[m] for v in methods.values()) for m in ('known_known','background_estimated')}
        def refresh(entry):
            ident=entry['id'];old=states[ident];old_methods=dict(methods[ident]);value=update_case_state(plan,entry,run,states,methods)
            state_counts[old]-=1;state_counts[states[ident]]+=1
            for m in method_counts:method_counts[m][old_methods[m]]-=1;method_counts[m][methods[ident][m]]+=1
            return value
        def save():
            s['elapsed_seconds']=time.perf_counter()-start
            sealed_json(run/'sessions.json',dict(plan_id=plan['plan_id'],sessions=sessions))
            return write_summary(plan,run,states,methods,sessions,reason if s['closed'] else 'active',state_counts=state_counts,method_counts=method_counts)
        def event(kind,**values):
            nonlocal event_number
            event_number+=1
            immutable(run/'events'/sid/f'{event_number:06d}.json',dict(plan_id=plan['plan_id'],session_id=sid,event=kind,utc=utc_now(),elapsed_seconds=time.perf_counter()-start,**values))
        try:
            save();event('session_started',pending=len(pending),prior_charged_seconds=prior)
            if (run/'STOP').exists():reason='stop_requested'
            else:
                ctx=mp.get_context('spawn');ready=ctx.Event();inbox=ctx.Queue();outbox=ctx.Queue();p=ctx.Process(target=worker if worker_target is None else worker_target,args=(plan,str(run),ready,inbox,outbox));job=ChildJob();handle=(p,job,inbox)
                try:
                    p.start();job.assign(p);s['worker_identity']=life.process_identity(p.pid);require(s['worker_identity'] is not None,'Cannot identify owned full worker');save();ready.set();event('worker_started',worker_identity=s['worker_identity'])
                except BaseException:
                    if p.pid is None:job.close()
                    else:life.stop_child(handle)
                    handle=None;raise
                setup=time.perf_counter();is_ready=False
                while pending or active:
                    now=time.perf_counter()
                    if now>=deadline:reason='global_budget';break
                    if active and now>=active['deadline']:reason='case_time_cap';break
                    if not is_ready and now-setup>=limits['setup_wall_seconds']:reason='setup_time_cap';break
                    if (run/'STOP').exists() and active is None:reason='stop_requested';break
                    if is_ready and active is None and pending:
                        ident=pending.popleft();active=dict(id=ident,attempt_id=sid+'_'+ident,deadline=min(deadline,time.perf_counter()+limits['per_history_total_wall_seconds']))
                        s.update(active_id=ident,active_attempt_id=active['attempt_id'],stage='case')
                        immutable(run/'attempts'/(active['attempt_id']+'.started.json'),dict(**identity(plan,by[ident]),attempt_id=active['attempt_id'],started_utc=utc_now(),worker_identity=s['worker_identity']))
                        save();event('dispatched',id=ident,attempt_id=active['attempt_id']);inbox.put((by[ident],active['attempt_id'],active['deadline']))
                    try:message=outbox.get(timeout=.1)
                    except queue.Empty:message=None
                    if message:
                        if message['event']=='ready':
                            is_ready=True;s.update(stage='idle',warmup=message['warmup'],context_metadata=message['metadata']);save();event('worker_ready')
                        elif message['event']=='progress':
                            require(active and message['id']==active['id'],'Unexpected stage progress identity')
                            # The scientific worker has already sealed and validated this one stage.
                            partial=read_sealed(ROOT/message['path']);kind=partial['progress']['event']
                            if kind in ('method_validated','procedure_optimized'):refresh(by[active['id']])
                            save()
                        elif message['event']=='finished':
                            require(active and message['id']==active['id'],'Unexpected full worker completion')
                            ident=active['id'];value=refresh(by[ident]);require(value is not None,'Missing durable case')
                            immutable(run/'attempts'/(active['attempt_id']+'.finished.json'),dict(plan_id=plan['plan_id'],id=ident,status=value['status'],case_sha256=sha256(case_path(run,ident))))
                            event('finished',id=ident,status=value['status']);active=None;s.update(active_id=None,active_attempt_id=None,stage='idle');save()
                            print(f"{ident}: {value['status']}; validated {state_counts['complete']}/{len(plan['ids'])}; {value.get('elapsed_seconds')}s",flush=True)
                            if value['status']!='complete':reason='first_unresolved_case';break
                        elif message['event']=='worker_error':s['worker_error']=message;reason='worker_error';break
                    if not p.is_alive() and (pending or active):reason='unexpected_worker_exit';break
        except KeyboardInterrupt:reason='keyboard_interrupt'
        except BaseException as exc:
            reason='parent_error:'+type(exc).__name__;s['parent_error']=str(exc);raise
        finally:
            # No read/adoption/commit may precede proof that the owned worker died.
            if handle is not None:
                p,job,inbox=handle
                if p.pid is None:job.close()
                else:
                    if active is None and p.is_alive():inbox.put(None);p.join(10)
                    life.stop_child(handle);s['worker_exitcode']=p.exitcode
            if active:
                entry=by[active['id']];value=refresh(entry)
                if value is None and reason in ('case_time_cap','global_budget','worker_error'):
                    status='time_cap' if reason in ('case_time_cap','global_budget') else 'error'
                    immutable(case_path(run,active['id']),dict(**identity(plan,entry),attempt_id=active['attempt_id'],status=status,payload=None,error=reason,elapsed_seconds=None,partial_records_retained=True))
                    value=refresh(entry)
                finish=run/'attempts'/(active['attempt_id']+'.finished.json')
                if not finish.exists():immutable(finish,dict(plan_id=plan['plan_id'],id=active['id'],status=value['status'] if value else 'interrupted',reason=reason,case_sha256=sha256(case_path(run,active['id'])) if value else None))
                event('attempt_closed',id=active['id'],status=value['status'] if value else 'interrupted',reason=reason)
            s.update(closed=True,active_id=None,active_attempt_id=None,stage='closed',ended_utc=utc_now(),reason=reason);save()
            write_index(plan,run,states,methods)
        final=write_summary(plan,run,states,methods,sessions,'complete' if state_counts.get('complete',0)==len(plan['ids']) else reason,state_counts=state_counts,method_counts=method_counts)
        return 0 if final['all_primary_complete'] else 130 if reason in ('keyboard_interrupt','stop_requested') else 2


def main():
    start=time.perf_counter();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=AREA/'configs/staged_full_frozen.json');p.add_argument('--run-dir',type=Path,default=AREA/'results/full_staged')
    modes=p.add_mutually_exclusive_group(required=True);modes.add_argument('--prepare-only',action='store_true');modes.add_argument('--resume',action='store_true');modes.add_argument('--status',action='store_true')
    args=p.parse_args();run=args.run_dir.resolve();require(run.is_relative_to(AREA/'results'),'Output outside iteration21/results')
    if args.status:
        print(read_sealed(run/'summary.json') if (run/'summary.json').exists() else dict(prepared=False,total_planned=2149,completed_validated=0,pending=2149,active=False));return 0
    plan=prepare(args.config.resolve(),run)
    if args.prepare_only:print(dict(prepared=True,plan_id=plan['plan_id'],total_planned=2149,scientific_work_launched=False));return 0
    return run_full(plan,run,start)


if __name__=='__main__':raise SystemExit(main())
