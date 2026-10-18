"""Explicit, bounded five-history cost benchmark. Never estimates power or size."""
from pathlib import Path
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import importlib.util
import math
import multiprocessing as mp
import queue
import sys
import time
import traceback
import uuid
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / 'iteration_21'
sys.path.insert(0, str(ROOT/'iteration_6/code'))
from checkpoint_io import (ChildJob, RunLock, atomic_bytes, canonical_hash,
                           read_json, read_sealed, sealed_json, sha256, utc_now)

IDS = ['null_0000', 'null_0841', 'null_1292', 'new_r000', 'new_r076']
SOURCES = [
    'iteration_21/code/benchmark_staged.py', 'iteration_21/code/staged_fit.py',
    'iteration_20/code/restricted_fit.py', 'iteration_20/code/restricted_history.py', 'iteration_18/code/cuda_oracle.py',
    'iteration_17/code/oracle_likelihood.py', 'iteration_19/code/run_full_cuda.py',
    'iteration_12/code/paired_history.py', 'iteration_12/code/stable_history.py',
    'iteration_5/code/empirical_process.py', 'iteration_5/code/kernels.py',
    'iteration_6/code/checkpoint_io.py',
]
PLAN_KEYS = ['config_path', 'config_sha256', 'contract', 'sources', 'ids', 'entries']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def lifetime_api():
    """Reuse only the sealed PID/token and owned-child cleanup implementation."""
    path = ROOT/'iteration_19/code/run_full_cuda.py'
    require(sha256(path) == '93f59620d6f7e9a0eccbf1ae3575429634080459b1ea38601b070a89ee906dbb',
            'Sealed lifetime helper changed')
    name = '_iteration21_lifetime'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def immutable(path, value):
    require(not path.exists(), 'Refusing completed-record overwrite: '+str(path))
    sealed_json(path, value)


def prepare(config_path, run):
    c = read_json(config_path)
    require(c['status'] == 'frozen_staged_cost_benchmark_before_saved_history_fits', 'Protocol is not frozen')
    require(c['benchmark']['ids'] == IDS and c['benchmark']['whole_parent_wall_seconds'] == 900 and
            c['benchmark']['per_history_total_wall_seconds'] == 300, 'Benchmark resource/cohort differs')
    for ref in c['authorities']:
        require(sha256(ROOT/ref['path']) == ref['sha256'], 'Prior authority changed: '+ref['path'])
    inventory_ref = next(v for v in c['authorities'] if '/inventory_' in v['path'])
    inventory = read_sealed(ROOT/inventory_ref['path'])
    by = {v['id']: v for v in inventory['entries']}
    entries = [by[id] for id in IDS]
    for e in entries:
        for ref in e['files'].values():
            require(sha256(ROOT/ref['path']) == ref['sha256'], 'Benchmark source changed: '+ref['path'])
    authority = dict(config_path=config_path.relative_to(ROOT).as_posix(), config_sha256=sha256(config_path),
                     contract=c, sources={p: sha256(ROOT/p) for p in SOURCES}, ids=IDS, entries=entries)
    plan = dict(authority, plan_id=canonical_hash(authority))
    run.mkdir(parents=True, exist_ok=True)
    with RunLock(run/'benchmark.lock'):
        if (run/'plan.json').exists():
            old = read_sealed(run/'plan.json')
            require(old['plan_id'] == plan['plan_id'] == canonical_hash({k: old[k] for k in PLAN_KEYS}),
                    'Prepared plan changed; retain old results and use a new amendment')
            plan = old
        else:
            immutable(run/'plan.json', dict(plan, prepared_utc=utc_now()))
            for p in SOURCES:
                atomic_bytes(run/'source_snapshots'/p, (ROOT/p).read_bytes())
            write_summary(plan, run, [], 'prepared')
    return plan


def identity(plan, e):
    return dict(plan_id=plan['plan_id'], id=e['id'], seed=e['seed'], clock_mode=e['clock_mode'],
                catalogue_sha256=e['files']['catalogue.npz']['sha256'],
                inventory_entry_sha256=canonical_hash(e))


def case_path(run, id):
    return run/'cases'/(id+'.json')


def method_path(run, id, method, kind):
    return run/'methods'/id/method/(kind+'.json')


def stage_binding(plan, entry, method):
    return dict(**identity(plan,entry), method_id=method, law_sha256=canonical_hash(plan['contract']['law']),
                source_hashes=plan['sources'], validator_source_sha256=plan['sources']['iteration_21/code/staged_fit.py'])


_BINDINGS={}

def staged_api():
    sys.path.insert(0,str(AREA/'code'))
    import staged_fit
    return staged_fit


def expected_binding(plan, entry):
    key=(plan['plan_id'],entry['id'],entry['files']['catalogue.npz']['sha256'])
    if key not in _BINDINGS:
        import numpy as np
        path=ROOT/entry['files']['catalogue.npz']['path']
        require(sha256(path)==key[2], 'Stage catalogue changed')
        with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
        spec=plan['contract']['law']; api=staged_api()
        law=api.law_from_setting(dict(beta=spec['beta'],horizon=spec['horizon_days'],coordinate_horizon=spec['coordinate_horizon_days'],generating_fit=spec['alternative_generating_fit']))
        _BINDINGS[key]=api.input_binding(arrays,law,entry['clock_mode'],plan['contract'])
    return _BINDINGS[key]


def read_stage(plan, entry, run, method, kind):
    path=method_path(run,entry['id'],method,kind)
    if not path.exists():return None
    value=read_sealed(path)
    require(all(value.get(k)==v for k,v in stage_binding(plan,entry,method).items()), 'Stage identity differs')
    record=value['payload']; digest=record.get('checkpoint_sha256')
    require(digest==canonical_hash({k:v for k,v in record.items() if k!='checkpoint_sha256'}), 'Stage inner digest differs')
    require(record['id']==method and record['checkpoint_kind']==('validated_method' if kind=='validated' else 'optimized_method'), 'Stage kind differs')
    quota=2 if method=='known_known' else 4
    opt=record['optimized_checkpoint'] if kind=='validated' else record
    require(opt['optimization_complete'] is True and opt['complete'] is False and
            len(opt['subfits'])==quota and all(r['success'] is True for r in opt['subfits']), 'Stage start quota differs')
    if kind=='validated':
        optimized=read_stage(plan,entry,run,method,'optimized')
        require(optimized is not None and value['optimization_artifact_sha256']==sha256(method_path(run,entry['id'],method,'optimized')) and
                record['optimized_checkpoint_sha256']==optimized['payload']['checkpoint_sha256'] and opt==optimized['payload'], 'Optimization dependency differs')
        require(record['complete'] is True and record['status']=='complete' and record['gates']['passed'] is True and
                len(record['selected_objectives'])==2 and all(g['passed'] is True and math.isfinite(g['absolute_difference']) and g['absolute_difference']<1e-4 for g in record['selected_objectives']) and
                len(record['witnesses'])==(1 if method=='known_known' else 2) and all(w['passed'] is True for w in record['witnesses']), 'Incomplete method gates')
        if method=='background_estimated':
            kk=read_stage(plan,entry,run,'known_known','validated')
            require(kk is not None and value['known_known_artifact_sha256']==sha256(method_path(run,entry['id'],'known_known','validated')) and
                    record['known_known_checkpoint_sha256']==kk['payload']['checkpoint_sha256'], 'Known-known dependency differs')
    api=staged_api(); binding=expected_binding(plan,entry)
    if kind=='optimized':api.validate_optimized_record(record,plan['contract'],binding=binding)
    else:api.validate_method_record(record,plan['contract'],binding=binding,known_known=kk['payload'] if method=='background_estimated' else None)
    return value


def store_stage(plan, entry, run, method, kind, payload, attempt):
    value=dict(**stage_binding(plan,entry,method), attempt_id=attempt, payload=payload, completed_utc=utc_now())
    if kind=='validated':
        value['optimization_artifact_sha256']=sha256(method_path(run,entry['id'],method,'optimized'))
        if method=='background_estimated':value['known_known_artifact_sha256']=sha256(method_path(run,entry['id'],'known_known','validated'))
    immutable(method_path(run,entry['id'],method,kind),value)
    read_stage(plan,entry,run,method,kind)


def cached(plan, e, run):
    p=case_path(run,e['id'])
    if not p.exists():return None
    v=read_sealed(p)
    require(all(v.get(k)==x for k,x in identity(plan,e).items()), 'Cached case identity changed')
    require(v['status'] in ('complete','failed','error','time_cap'), 'Invalid terminal status')
    if v['status']=='complete':
        payload=v['payload']
        require(payload['complete'] is True and payload['status']=='complete' and payload['prescribed_optimizer_calls']==6 and
                len(payload['records'])==6 and all(r['success'] is True for r in payload['records']), 'Six-call completion differs')
        require([m['id'] for m in payload['methods']]==['known_known','background_estimated'], 'Methods differ')
        require([r for m in payload['methods'] for r in m['subfits']]==payload['records'], 'Case optimization lineage differs')
        metrics=payload.get('cuda_metrics') or {}
        require(metrics.get('dtype')=='float64' and all(metrics.get(k) is False for k in ('allow_tf32','mixed_precision','fast_math_requested')) and
                metrics.get('peak_allocated_bytes',2**31)<=2**30, 'CUDA precision/memory metadata differs')
        for method in payload['methods']:
            stage=read_stage(plan,e,run,method['id'],'validated')
            require(stage is not None and stage['payload']==method, 'Case method lacks durable validated stage')
    return v


def assemble_validated_case(plan,entry,run):
    if case_path(run,entry['id']).exists():return None
    stages=[read_stage(plan,entry,run,m,'validated') for m in ('known_known','background_estimated')]
    if not all(stages):return None
    methods=[s['payload'] for s in stages]; records=[r for m in methods for r in m['subfits']]
    payload=dict(complete=True,status='complete',methods=methods,records=records,planned_optimizer_calls=6,
                 prescribed_optimizer_calls=6,new_optimizer_calls=0,
                 witnesses=[w for m in methods for w in m['witnesses']],
                 validation=dict(selected_objectives=[g for m in methods for g in m['selected_objectives']],witnesses_passed=True),
                 cuda_metrics=methods[-1].get('cuda_metrics'),assembled_from_validated_stages=True)
    immutable(case_path(run,entry['id']),dict(**identity(plan,entry),attempt_id='metadata_only_validated_stage_recovery',status='complete',payload=payload,
              elapsed_seconds=None,cpu_seconds=None,completed_utc=utc_now(),resource_note='Completed method stages retain timing; interrupted case wall unknown, charged in session recovery. No new GPU/CPU score.'))
    return cached(plan,entry,run)


def write_summary(plan, run, sessions, reason):
    states = {e['id']: (cached(plan, e, run) or {}).get('status', 'pending') for e in plan['entries']}
    counts = {s: list(states.values()).count(s) for s in sorted(set(states.values()))}
    complete = counts.get('complete', 0) == len(plan['entries'])
    method_states={e['id']:{m:('validated' if read_stage(plan,e,run,m,'validated') else 'optimized_unvalidated' if read_stage(plan,e,run,m,'optimized') else 'pending') for m in ('known_known','background_estimated')} for e in plan['entries']}
    value = dict(plan_id=plan['plan_id'], updated_utc=utc_now(), reason=reason, states=states, counts=counts,
                 all_five_complete=complete, method_states=method_states, active=any(not s['closed'] for s in sessions),
                 active_ids=[s['active_id'] for s in sessions if not s['closed'] and s.get('active_id')],
                 charged_seconds=sum(s['elapsed_seconds'] for s in sessions),
                 sessions_closed=all(s['closed'] for s in sessions),
                 full_refit_campaign_launched=False, no_power_or_size_estimate=True)
    sealed_json(run/'summary.json', value)
    return value


def worker(plan, run_name, ready, inbox, outbox):
    run = Path(run_name)
    context = None
    began = time.perf_counter()
    while not ready.wait(.1):
        parent = mp.parent_process()
        if parent is None or not parent.is_alive() or time.perf_counter()-began > 15:
            return
    try:
        import numpy as np
        sys.path.insert(0, str(AREA/'code'))
        from staged_fit import fit_case, CudaOracleContext, law_from_setting
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            context = CudaOracleContext(max_allocated_bytes=1024*2**20)
            warmup = context.warmup()
            outbox.put(dict(event='ready', warmup=warmup, metadata=context.metadata))
            while True:
                task = inbox.get()
                if task is None:
                    return
                entry, attempt, deadline = task
                start = time.perf_counter(); cpu_start = time.process_time(); index = 0
                def progress(value):
                    nonlocal index
                    index += 1
                    if value['event']=='procedure_optimized':
                        store_stage(plan,entry,run,value['method'],'optimized',value['partial_method'],attempt)
                    elif value['event']=='method_validated':
                        store_stage(plan,entry,run,value['method'],'validated',value['validated_method'],attempt)
                    path = run/'partials'/entry['id']/attempt/f'{index:05d}.json'
                    immutable(path, dict(**identity(plan, entry), attempt_id=attempt,
                                         elapsed_seconds=time.perf_counter()-start, progress=value))
                    outbox.put(dict(event='progress', id=entry['id'], path=path.relative_to(ROOT).as_posix(),
                                    elapsed_seconds=time.perf_counter()-start))
                status = 'error'; payload = None; error = None
                try:
                    path = ROOT/entry['files']['catalogue.npz']['path']
                    require(sha256(path) == entry['files']['catalogue.npz']['sha256'], 'Input history changed')
                    with np.load(path, allow_pickle=False) as data:
                        arrays = {k: data[k] for k in data.files}
                    spec = plan['contract']['law']
                    law = law_from_setting(dict(beta=spec['beta'], horizon=spec['horizon_days'],
                        coordinate_horizon=spec['coordinate_horizon_days'], generating_fit=spec['alternative_generating_fit']))
                    optimized={m:v['payload'] for m in ('known_known','background_estimated') if (v:=read_stage(plan,entry,run,m,'optimized')) is not None}
                    validated={m:v['payload'] for m in ('known_known','background_estimated') if (v:=read_stage(plan,entry,run,m,'validated')) is not None}
                    payload = fit_case(arrays, law, clock_mode=entry['clock_mode'], contract=plan['contract'],
                                       deadline=deadline, context=context, progress_callback=progress,
                                       optimized_checkpoints=optimized, validated_checkpoints=validated)
                    status = 'complete' if payload['complete'] else 'time_cap' if payload.get('status') == 'time_cap' else 'failed'
                except TimeoutError as exc:
                    status = 'time_cap'; error = str(exc)
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'; progress(dict(event='exception', error=error, traceback=traceback.format_exc()))
                value = dict(**identity(plan, entry), attempt_id=attempt, status=status, payload=payload,
                             error=error, partial_records=index, elapsed_seconds=time.perf_counter()-start,
                             cpu_seconds=time.process_time()-cpu_start, completed_utc=utc_now())
                immutable(case_path(run, entry['id']), value)
                outbox.put(dict(event='finished', id=entry['id'], status=status))
                if status != 'complete':
                    return
    except BaseException as exc:
        outbox.put(dict(event='worker_error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
    finally:
        if context is not None:
            context.close()


def run_benchmark(plan, run, invocation_began, worker_target=None):
    life = lifetime_api()
    limit = 900.; reserve = 15.; reason = 'active'; handle = None; active = None
    with RunLock(run/'benchmark.lock'):
        require(canonical_hash({k: plan[k] for k in PLAN_KEYS}) == plan['plan_id'], 'Canonical plan differs')
        require(sha256(ROOT/plan['config_path']) == plan['config_sha256'] and
                read_json(ROOT/plan['config_path']) == plan['contract'], 'Frozen config differs')
        for p, h in plan['sources'].items():
            require(sha256(ROOT/p) == h, 'Runtime source changed: '+p)
        freeze = read_json(AREA/'logs/runtime_source_freeze.json')
        require(freeze['plan_id'] == plan['plan_id'] and freeze['validation_passed'] is True,
                'Pre-execution validation/source freeze missing')
        ledger = read_sealed(run/'sessions.json') if (run/'sessions.json').exists() else dict(plan_id=plan['plan_id'], sessions=[])
        require(ledger['plan_id'] == plan['plan_id'], 'Session ledger belongs to another plan')
        sessions = ledger['sessions']
        for s in sessions:
            if not s['closed']:
                life.ensure_old_worker_dead(s)
                prefix = s['elapsed_seconds']; other = sum(x['elapsed_seconds'] for x in sessions if x is not s)
                bound = 300. if s.get('active_id') else 60. if s.get('stage') == 'setup' else 0.
                assumed = min(max(0., limit-other-prefix), bound+reserve)
                if s.get('active_id'):
                    e = next(v for v in plan['entries'] if v['id'] == s['active_id'])
                    old = cached(plan, e, run)
                    finish = run/'attempts'/(s['active_attempt_id']+'.finished.json')
                    if not finish.exists():
                        immutable(finish, dict(plan_id=plan['plan_id'], id=e['id'],
                            status=old['status'] if old else 'interrupted', reason='unclean_parent_recovery',
                            case_sha256=sha256(case_path(run, e['id'])) if old else None))
                s.update(closed=True, elapsed_seconds=prefix+assumed, measured_prefix_seconds=prefix,
                         assumed_unobserved_seconds=assumed, reason='unclean_bounded_recovery',
                         active_id=None, active_attempt_id=None, stage='closed')
        sealed_json(run/'sessions.json', dict(plan_id=plan['plan_id'], sessions=sessions))
        prior = sum(s['elapsed_seconds'] for s in sessions)
        by = {e['id']: e for e in plan['entries']}
        for e in plan['entries']:assemble_validated_case(plan,e,run)
        saved = {id: cached(plan, by[id], run) for id in IDS}
        if any(v and v['status'] != 'complete' for v in saved.values()):
            write_summary(plan, run, sessions, 'retained_failed_case_blocks_resume')
            return 2
        pending = [id for id in IDS if saved[id] is None]
        if not pending:
            write_summary(plan, run, sessions, 'complete')
            return 0
        session_id = uuid.uuid4().hex
        s = dict(session_id=session_id, started_utc=utc_now(), closed=False, elapsed_seconds=0.,
                 active_id=None, active_attempt_id=None, worker_identity=None, stage='setup')
        sessions.append(s)
        deadline = invocation_began+max(0., limit-prior-reserve)
        def save():
            s['elapsed_seconds'] = time.perf_counter()-invocation_began
            sealed_json(run/'sessions.json', dict(plan_id=plan['plan_id'], sessions=sessions))
            write_summary(plan, run, sessions, reason)
        try:
            save()
            if (run/'STOP').exists():
                reason = 'stop_requested'
            elif time.perf_counter() >= deadline:
                reason = 'global_budget'
            else:
                ctx = mp.get_context('spawn'); ready = ctx.Event(); inbox = ctx.Queue(); outbox = ctx.Queue()
                p = ctx.Process(target=worker if worker_target is None else worker_target,
                                args=(plan, str(run), ready, inbox, outbox))
                job = ChildJob(); handle = (p, job, inbox)
                p.start(); job.assign(p); s['worker_identity'] = life.process_identity(p.pid)
                require(s['worker_identity'] is not None, 'Cannot identify owned worker')
                save(); ready.set(); setup_started = time.perf_counter(); is_ready = False
                while pending or active:
                    now = time.perf_counter()
                    if now >= deadline:
                        reason = 'global_budget'; break
                    if active and now >= active['deadline']:
                        reason = 'case_time_cap'; break
                    if not is_ready and now-setup_started >= 60:
                        reason = 'setup_time_cap'; break
                    if (run/'STOP').exists() and active is None:
                        reason = 'stop_requested'; break
                    if is_ready and active is None and pending:
                        id = pending.pop(0)
                        active = dict(id=id, attempt_id=session_id+'_'+id,
                                      deadline=min(deadline, time.perf_counter()+300.))
                        s.update(active_id=id, active_attempt_id=active['attempt_id'], stage='case')
                        immutable(run/'attempts'/(active['attempt_id']+'.started.json'),
                                  dict(**identity(plan, by[id]), attempt_id=active['attempt_id'], started_utc=utc_now()))
                        save(); inbox.put((by[id], active['attempt_id'], active['deadline']))
                    try:
                        message = outbox.get(timeout=.1)
                    except queue.Empty:
                        message = None
                    if message:
                        if message['event'] == 'ready':
                            is_ready = True; s.update(stage='idle', warmup=message['warmup'], context_metadata=message['metadata']); save()
                        elif message['event'] == 'progress':
                            s['elapsed_seconds']=time.perf_counter()-invocation_began
                            sealed_json(run/'sessions.json', dict(plan_id=plan['plan_id'],sessions=sessions))
                        elif message['event'] == 'finished':
                            require(active and active['id'] == message['id'], 'Unexpected worker completion')
                            value = cached(plan, by[message['id']], run)
                            require(value is not None, 'Completion has no durable record')
                            immutable(run/'attempts'/(active['attempt_id']+'.finished.json'),
                                      dict(plan_id=plan['plan_id'], id=active['id'], status=value['status'],
                                           case_sha256=sha256(case_path(run, active['id']))))
                            print(f"{active['id']}: {value['status']}; {value['elapsed_seconds']}s; cost-only, no power tally", flush=True)
                            active = None; s.update(active_id=None, active_attempt_id=None, stage='idle'); save()
                            if value['status'] != 'complete':
                                reason = 'first_unresolved_case'; break
                        elif message['event'] == 'worker_error':
                            s['worker_error'] = message; reason = 'worker_error'; break
                    if not p.is_alive() and (pending or active):
                        reason = 'unexpected_worker_exit'; break
        except KeyboardInterrupt:
            reason = 'keyboard_interrupt'
        except BaseException as exc:
            reason = 'parent_error:'+type(exc).__name__
            s['parent_error'] = str(exc)
            raise
        finally:
            # Preserve an unclosed ledger if cleanup cannot establish child death.
            if handle is not None:
                p, job, inbox = handle
                if p.pid is not None:
                    if active is None and p.is_alive():
                        inbox.put(None); p.join(10)
                    life.stop_child(handle)
                    s['worker_exitcode'] = p.exitcode
                else:
                    job.close()
            if active:
                entry = by[active['id']]; value = cached(plan, entry, run)
                if value is None and reason in ('case_time_cap', 'global_budget'):
                    immutable(case_path(run, active['id']), dict(**identity(plan, entry),
                        attempt_id=active['attempt_id'], status='time_cap', payload=None,
                        error=reason, elapsed_seconds=None, partial_records_retained=True))
                    value = cached(plan, entry, run)
                finish_path = run/'attempts'/(active['attempt_id']+'.finished.json')
                if not finish_path.exists():
                    immutable(finish_path, dict(plan_id=plan['plan_id'], id=active['id'],
                        status=value['status'] if value else 'interrupted', reason=reason,
                        case_sha256=sha256(case_path(run, active['id'])) if value else None))
            s.update(closed=True, active_id=None, active_attempt_id=None,
                     stage='closed', ended_utc=utc_now(), reason=reason)
            save()
        final = write_summary(plan, run, sessions, reason)
        return 0 if final['all_five_complete'] else 130 if reason in ('keyboard_interrupt', 'stop_requested') else 2


def main():
    began = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=AREA/'configs/staged_benchmark_frozen.json')
    parser.add_argument('--run-dir', type=Path, default=AREA/'results/benchmark')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args(); run = args.run_dir.resolve()
    require(run.is_relative_to(AREA/'results'), 'Output outside iteration21/results')
    if args.status:
        print(read_sealed(run/'summary.json') if (run/'summary.json').exists() else {'prepared': False})
        return 0
    plan = prepare(args.config.resolve(), run)
    if args.prepare_only:
        print(dict(prepared=True, plan_id=plan['plan_id'], saved_history_fit_launched=False))
        return 0
    require(args.resume, 'Explicit --resume required; preparation alone cannot fit histories')
    return run_benchmark(plan, run, began)


if __name__ == '__main__':
    raise SystemExit(main())
