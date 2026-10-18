"""Checkpointed level-only matched-null calibration. Separate benchmark/full plans; status is read-only."""
from pathlib import Path
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
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
AREA = ROOT / 'iteration_26'
RUN = AREA / 'results/full_calibration'
CONFIG = AREA / 'configs/full_protocol_frozen.json'
sys.path.insert(0, str(ROOT / 'iteration_6/code'))
from checkpoint_io import (ChildJob, RunLock, atomic_bytes, canonical_hash,
                           read_json, read_sealed, sealed_json, sha256, utc_now)

IDS = [f'null_{i:04d}' for i in range(1999)] + [f'new_r{i:03d}' for i in range(150)]
PILOT_IDS = ['null_0000','null_0841','null_1292','new_r000','new_r076',
             'null_1107','null_1166','new_r100','null_1345','new_r140']
BENCHMARK_IDS = PILOT_IDS + ['null_0057','null_0642']
PRIOR_PLAN = ROOT / 'iteration_21/results/full_staged/plan.json'
OWN_SOURCES = ['iteration_26/code/run_level_calibration.py',
               'iteration_26/code/level_only_fit.py',
               'iteration_26/code/calibrate_level_only.py',
               'iteration_26/code/review_benchmark.py',
               'iteration_26/run_level_calibration.ps1',
               'iteration_24/code/profile_optimizer.py']
PLAN_KEYS = ['config_path','config_sha256','contract','sources','ids','entries','benchmarks']
CLOSURE_ALLOWANCE_SECONDS = 1.


def require(ok, message):
    if not ok:
        raise ValueError(message)


def immutable(path, value):
    require(not Path(path).exists(), 'Refusing completed-record overwrite: '+str(path))
    sealed_json(path, value)


def lifetime():
    path = ROOT / 'iteration_19/code/run_full_cuda.py'
    require(sha256(path) == '93f59620d6f7e9a0eccbf1ae3575429634080459b1ea38601b070a89ee906dbb', 'Lifetime helper changed')
    name = '_ablation_lifetime'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def validate_contract(c):
    stage = c['stage']; ids = BENCHMARK_IDS if stage == 'benchmark' else IDS
    require(stage in ('benchmark','full') and c['case_ids'] == ids and
            c['status'] == 'frozen_before_new_level_calibration_outcomes', 'Stage/role cohort differs')
    roles = c['future_inference']
    require(roles['reference_ids'] == IDS[:999] and roles['heldout_null_ids'] == IDS[999:1999] and
            roles['alternative_ids'] == IDS[1999:], 'Exact999/1000/150 calibration roles required')
    r = c['resources']
    require(r['total_parent_wall_seconds'] == (2400 if stage=='benchmark' else 28800) and
            r['per_case_wall_seconds'] == {i:(1800 if i=='null_1292' else 900) for i in ids} and
            r['validation_reserve_seconds'] == {i:(180 if i=='null_1292' else 45) for i in ids} and
            r['setup_wall_seconds']==60 and r['cleanup_reserve_seconds']==15 and
            r['GPU_workers']==1 and r['native_threads']==1, 'Resource contract differs')
    require(c['baseline_authority']=='uniform_iteration21_known_known_for_all2149' and
            c['calibration']['center']=='raw_original_backend_contrast16_statistic' and
            c['calibration']['alpha']==.05 and c['calibration']['ties']=='greater_than_or_equal' and
            c['calibration']['epsilon_floor']==1e-6, 'Scientific policy differs')
    return ids


def prepare(config=CONFIG, run=RUN):
    c = read_json(config); ids = validate_contract(c)
    for record in c['authorities']:
        require(sha256(ROOT/record['path']) == record['sha256'], 'Prior authority changed: '+record['path'])
    prior = read_sealed(PRIOR_PLAN)
    require(c['law'] == read_sealed(ROOT/'iteration_25/results/ablation_pilot/plan.json')['contract']['law'], 'Generating/observation law changed')
    declared = {e['id']:e for e in prior['entries']}
    require(list(declared)==IDS and len({e['seed'] for e in declared.values()})==2149, 'Prior complete cohort differs')
    entries = [declared[i] for i in ids]; benchmarks={}
    for entry in entries:
        for record in entry['files'].values():
            require(sha256(ROOT/record['path']) == record['sha256'], 'History artifact changed: '+record['path'])
        path = f"iteration_21/results/full_staged/cases/{entry['id']}.json"
        baseline = read_sealed(ROOT/path)
        require(baseline['status']=='complete' and baseline['plan_id']==prior['plan_id'] and baseline['id']==entry['id'], 'Baseline case differs')
        benchmarks[entry['id']] = dict(path=path,sha256=sha256(ROOT/path),plan_id=prior['plan_id'])
    sources = {p:sha256(ROOT/p) for p in list(prior['sources'])+OWN_SOURCES}
    require(all(sources[p] == h for p,h in prior['sources'].items()), 'Prior numerical source changed')
    body = dict(config_path=Path(config).relative_to(ROOT).as_posix(),config_sha256=sha256(config),contract=c,
                sources=sources,ids=ids,entries=entries,benchmarks=benchmarks)
    plan = dict(body,plan_id=canonical_hash(body),prepared_utc=utc_now()); run=Path(run)
    run.mkdir(parents=True,exist_ok=True)
    with RunLock(run/'pilot.lock'):
        if (run/'plan.json').exists():
            old=read_sealed(run/'plan.json');require(old['plan_id']==plan['plan_id'],'Prepared plan differs');return old
        immutable(run/'plan.json',plan)
        for path in sources:atomic_bytes(run/'source_snapshots'/path,(ROOT/path).read_bytes())
        write_summary(plan,run,{i:'pending' for i in ids},[],'prepared')
    return plan


def guard(plan):
    require(canonical_hash({k:plan[k] for k in PLAN_KEYS})==plan['plan_id'],'Plan seal differs')
    c=plan['contract']; ids=validate_contract(c)
    require(plan['ids']==ids and [e['id'] for e in plan['entries']]==ids,'Stage plan identities differ')
    require(sha256(ROOT/plan['config_path'])==plan['config_sha256'] and read_json(ROOT/plan['config_path'])==c,'Protocol bytes differ')
    for path,digest in plan['sources'].items():require(sha256(ROOT/path)==digest,'Runtime source changed: '+path)
    freeze=read_sealed(AREA/'logs/runtime_source_freeze.json')
    stage=c['stage']
    require(freeze['plan_ids'][stage]==plan['plan_id'] and freeze['sources']==plan['sources'] and freeze['validation_passed'] is True,'Source freeze differs')
    require(set(freeze['review_roles'])=={'A','C'},'Actual A/C reviews required')
    for role,path in freeze['review_roles'].items():
        require(sha256(ROOT/path)==freeze['review_snapshots'][path],'Review snapshot changed')
        review=read_sealed(ROOT/path)
        require(review['role']==role and review['passed'] is True and review['plan_ids']==freeze['plan_ids'] and review['source_hashes']==plan['sources'],'Reviewer decision/binding differs')
    for path,digest in freeze['validation_snapshots'].items():require(sha256(ROOT/path)==digest,'Validation snapshot changed')
    if stage=='full':
        gate=read_sealed(AREA/'reviews/benchmark_resource_gate.json')
        require(gate['passed'] is True and gate['full_plan_id']==plan['plan_id'] and gate['benchmark_plan_id']==freeze['plan_ids']['benchmark'] and
                gate['source_hashes']==plan['sources'],'Passed benchmark resource gate required')
        for path,digest in gate['evidence_hashes'].items():require(sha256(ROOT/path)==digest,'Benchmark evidence changed')
        require(set(gate['review_roles'])=={'A','C'},'Actual A/C benchmark reviews required')
        summary_path='iteration_26/results/benchmark/summary.json'
        require(summary_path in gate['evidence_hashes'],'Benchmark summary hash required')
        for role,path in gate['review_roles'].items():
            require(path in gate['evidence_hashes'],'Benchmark reviewer snapshot hash required')
            review=read_sealed(ROOT/path)
            require(review['stage']=='benchmark_postrun' and review['role']==role and review['passed'] is True and
                    review['benchmark_summary_sha256']==gate['evidence_hashes'][summary_path] and
                    review['plan_ids']==freeze['plan_ids'] and review['source_hashes']==plan['sources'],'Benchmark reviewer binding differs')
    for b in plan['benchmarks'].values():require(sha256(ROOT/b['path'])==b['sha256'],'Saved baseline changed')


def identity(plan, entry):
    return dict(plan_id=plan['plan_id'], id=entry['id'], seed=entry['seed'], clock_mode=entry['clock_mode'],
                inventory_entry_sha256=canonical_hash(entry),
                catalogue_sha256=entry['files']['catalogue.npz']['sha256'],
                benchmark_sha256=plan['benchmarks'][entry['id']]['sha256'])


def cached(plan,entry,run):
    path = Path(run)/'cases'/(entry['id']+'.json')
    if not path.exists():
        return None
    case = read_sealed(path)
    require(all(case.get(k) == v for k,v in identity(plan,entry).items()), 'Cached case identity differs')
    require(case['status'] in ('complete','failed','error','time_cap'), 'Invalid case state')
    if case['status'] == 'complete':
        sys.path.insert(0,str(AREA/'code'))
        from level_only_fit import validate_result
        require(validate_result(case['payload'],plan['contract']) is True, 'Completed numerical payload differs')
        require(case['payload']['feature_metadata']['clock_mode']==entry['clock_mode'], 'Payload clock differs from case authority')
        require(case['payload']['baseline_binding']['id']==entry['id'], 'Payload baseline history differs from case authority')
        require(case['payload']['baseline_binding']['plan_id']==plan['benchmarks'][entry['id']]['plan_id'], 'Payload baseline plan differs')
        b=plan['benchmarks'][entry['id']]
        require(sha256(ROOT/b['path'])==b['sha256'],'Cached baseline bytes differ')
        baseline=read_sealed(ROOT/b['path'])
        require(baseline['id']==entry['id'] and baseline['catalogue_sha256']==entry['files']['catalogue.npz']['sha256'] and
                case['payload']['baseline_binding']['canonical_case_sha256']==canonical_hash(baseline),'Cached baseline content/input linkage differs')
        kk=baseline['payload']['methods'][0];compact=case['payload']['baseline_method']
        require(kk['id']=='known_known' and case['payload']['baseline_binding']['saved_input_binding']=={k:kk['input_binding'][k] for k in ('arrays','law','clock_mode')} and
                all(compact[p]['pars']==kk[p]['pars'] and compact['saved_'+p+'_nll']==kk[p]['nll'] for p in ('null','alternative')) and
                all(compact['saved_contrast'+str(o)]==kk['contrast'+str(o)] for o in (16,32)), 'Cached compact baseline detached from actual authority')
    return case


def write_summary(plan,run,states,sessions,reason):
    value = dict(plan_id=plan['plan_id'],updated_utc=utc_now(),reason=reason,states=states,
                 counts={s:list(states.values()).count(s) for s in sorted(set(states.values()))},
                 all_complete=all(s=='complete' for s in states.values()),sessions_closed=all(s['closed'] for s in sessions),
                 charged_seconds=sum(s['elapsed_seconds'] for s in sessions),
                 active_ids=[s['active_id'] for s in sessions if not s['closed'] and s.get('active_id')],
                 new_models=['level_only'],benchmarks_reused=True,stage=plan['contract']['stage'],
                 active=any(not s['closed'] for s in sessions),total_planned=len(plan['ids']),
                 terminal_complete=all(v=='complete' for v in states.values()) and bool(sessions) and all(s['closed'] for s in sessions),
                 power_reestimated=False,new_simulations=False,new_catalogue=False)
    sealed_json(Path(run)/'summary.json',value)
    return value


def worker(plan,run_name,ready,inbox,outbox):
    run = Path(run_name)
    start = time.perf_counter()
    while not ready.wait(.05):
        parent = mp.parent_process()
        if parent is None or not parent.is_alive() or time.perf_counter()-start > 15:
            return
    context = None
    try:
        import numpy as np
        from threadpoolctl import threadpool_limits
        sys.path.insert(0,str(AREA/'code'))
        from level_only_fit import fit_level_case,validate_result
        sys.path.insert(0,str(ROOT/'iteration_18/code'))
        sys.path.insert(0,str(ROOT/'iteration_17/code'))
        from cuda_oracle import CudaOracleContext
        from oracle_likelihood import law_from_setting
        with threadpool_limits(limits=1):
            context = CudaOracleContext(max_allocated_bytes=2**30)
            outbox.put(dict(event='ready',warmup=context.warmup(),context_metadata=context.metadata))
            while True:
                task = inbox.get()
                if task is None:
                    return
                entry,attempt,deadline = task
                began = time.perf_counter();cpu_began = time.process_time();index = 0
                def progress(value):
                    nonlocal index
                    index += 1
                    path = run/'partials'/entry['id']/attempt/f'{index:05d}.json'
                    immutable(path,dict(**identity(plan,entry),attempt_id=attempt,elapsed_seconds=time.perf_counter()-began,progress=value))
                    outbox.put(dict(event='progress',id=entry['id'],path=path.relative_to(ROOT).as_posix()))
                payload = None;error = None;status = 'error'
                try:
                    for record in entry['files'].values():
                        require(sha256(ROOT/record['path']) == record['sha256'], 'History source changed')
                    with np.load(ROOT/entry['files']['catalogue.npz']['path'],allow_pickle=False) as z:
                        arrays = {k:z[k] for k in z.files}
                    b = plan['benchmarks'][entry['id']]
                    require(sha256(ROOT/b['path']) == b['sha256'], 'Benchmark source changed')
                    benchmark = read_sealed(ROOT/b['path'])
                    s = plan['contract']['law']
                    law = law_from_setting(dict(beta=s['beta'],horizon=s['horizon_days'],coordinate_horizon=s['coordinate_horizon_days'],generating_fit=s['alternative_generating_fit']))
                    payload = fit_level_case(arrays,law,clock_mode=entry['clock_mode'],protocol=plan['contract'],
                                                baseline_case=benchmark,context=context,deadline=deadline,progress_callback=progress)
                    if payload.get('complete') is True:
                        require(validate_result(payload,plan['contract']) is True, 'Produced complete payload failed pure validation')
                        status = 'complete'
                    else:
                        status = 'time_cap' if payload.get('status')=='time_cap' else 'failed'
                except TimeoutError as exc:
                    status = 'time_cap';error = str(exc)
                except Exception as exc:
                    error = type(exc).__name__+': '+str(exc)
                    progress(dict(event='exception',error=error,traceback=traceback.format_exc()))
                path = run/'cases'/(entry['id']+'.json')
                immutable(path,dict(**identity(plan,entry),attempt_id=attempt,status=status,payload=payload,error=error,
                                    elapsed_seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu_began,
                                    partial_records=index,completed_utc=utc_now()))
                outbox.put(dict(event='finished',id=entry['id'],status=status))
                if status != 'complete':
                    return
    except BaseException as exc:
        outbox.put(dict(event='worker_error',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc()))
    finally:
        if context is not None:
            context.close()


def execute(plan,run=RUN,began=None,worker_target=None):
    began = time.perf_counter() if began is None else began
    run = Path(run);life = lifetime();resources = plan['contract']['resources']
    limit = resources['total_parent_wall_seconds'];reserve = resources['cleanup_reserve_seconds']
    ids = plan['ids']
    handle = None;active = None;reason = 'preflight';states = {i:'pending' for i in ids}
    with RunLock(run/'pilot.lock'):
        guard(plan)
        ledger_path = run/'sessions.json'
        marker = run/'execution_started.json'
        require(ledger_path.exists() or not marker.exists(), 'Execution marker without ledger; refusing budget reset')
        require(ledger_path.exists() or not any((run/d).exists() for d in ('cases','attempts','partials')), 'Scientific traces without ledger; refusing budget reset')
        if not marker.exists():
            immutable(marker,dict(plan_id=plan['plan_id'],first_invoked_utc=utc_now(),scope='Execution accounting began; not proof a fit was launched'))
        else:
            require(read_sealed(marker)['plan_id']==plan['plan_id'], 'Execution marker belongs to another plan')
        ledger = read_sealed(ledger_path) if ledger_path.exists() else dict(plan_id=plan['plan_id'],sessions=[])
        require(ledger['plan_id'] == plan['plan_id'], 'Wrong session ledger')
        sessions = ledger['sessions']
        require(len({v['session_id'] for v in sessions})==len(sessions), 'Duplicate session identity')
        for index,old in enumerate(sessions):
            require(isinstance(old['closed'],bool) and math.isfinite(old['elapsed_seconds']) and old['elapsed_seconds']>=0, 'Invalid recorded elapsed time')
            require(old['closed'] or index==len(sessions)-1, 'Nonterminal open session')
        for old in sessions:
            if not old['closed']:
                life.ensure_old_worker_dead(old)
                prefix = old['elapsed_seconds'];others = sum(s['elapsed_seconds'] for s in sessions if s is not old)
                bound = resources['per_case_wall_seconds'][old['active_id']] if old.get('active_id') else resources['setup_wall_seconds']
                assumed = min(max(0.,limit-others-prefix),bound+reserve)
                if old.get('active_id'):
                    entry = next(e for e in plan['entries'] if e['id']==old['active_id'])
                    terminal = cached(plan,entry,run)
                    p = run/'attempts'/(old['active_attempt_id']+'.finished.json')
                    if not p.exists():
                        immutable(p,dict(plan_id=plan['plan_id'],id=old['active_id'],attempt_id=old['active_attempt_id'],status=terminal['status'] if terminal else 'interrupted',reason='unclean_parent_recovery',case_sha256=sha256(run/'cases'/(old['active_id']+'.json')) if terminal else None))
                old.update(closed=True,elapsed_seconds=prefix+assumed,measured_prefix_seconds=prefix,
                           assumed_unobserved_seconds=assumed,reason='unclean_bounded_inflight_reservation',
                           active_id=None,active_attempt_id=None,stage='closed',ended_utc=utc_now())
        prior = sum(s['elapsed_seconds'] for s in sessions)
        s = dict(session_id=uuid.uuid4().hex,started_utc=utc_now(),closed=False,elapsed_seconds=0.,
                 active_id=None,active_attempt_id=None,worker_identity=None,stage='preflight')
        sessions.append(s)
        def save():
            s['elapsed_seconds'] = time.perf_counter()-began
            sealed_json(ledger_path,dict(plan_id=plan['plan_id'],sessions=sessions))
        try:
            save()
            saved = {e['id']:cached(plan,e,run) for e in plan['entries']}
            states = {i:(saved[i] or {}).get('status','pending') for i in ids}
            pending = [i for i in ids if saved[i] is None]
            if any(v and v['status']!='complete' for v in saved.values()):
                reason = 'retained_failure_blocks_resume'
            elif (run/'fatal.json').exists():
                reason = 'retained_fatal_stop_blocks_resume'
            elif not pending:
                reason = 'complete'
            elif (run/'STOP').exists():
                reason = 'stop_requested'
            else:
                deadline = began+max(0.,limit-prior-reserve-CLOSURE_ALLOWANCE_SECONDS)
                if time.perf_counter() >= deadline:
                    reason = 'global_budget'
                else:
                    by = {e['id']:e for e in plan['entries']}
                    ctx = mp.get_context('spawn');ready = ctx.Event();inbox = ctx.Queue();outbox = ctx.Queue()
                    process = ctx.Process(target=worker if worker_target is None else worker_target,args=(plan,str(run),ready,inbox,outbox))
                    job = ChildJob();handle = (process,job,inbox)
                    process.start();job.assign(process);s['worker_identity'] = life.process_identity(process.pid)
                    require(s['worker_identity'] is not None, 'Owned worker identity unavailable')
                    s['stage'] = 'setup';save();ready.set();setup_began = time.perf_counter();is_ready = False
                    while pending or active:
                        now = time.perf_counter()
                        if now >= deadline:
                            reason = 'global_budget';break
                        if active and now >= active['deadline']:
                            reason = 'case_time_cap';break
                        if not is_ready and now-setup_began >= resources['setup_wall_seconds']:
                            reason = 'setup_time_cap';break
                        if active is None and (run/'STOP').exists():
                            reason = 'stop_requested';break
                        if is_ready and active is None and pending:
                            identifier = pending.pop(0);attempt = s['session_id']+'_'+identifier
                            active = dict(id=identifier,attempt_id=attempt,deadline=min(deadline,time.perf_counter()+resources['per_case_wall_seconds'][identifier]))
                            s.update(active_id=identifier,active_attempt_id=attempt,stage='case')
                            immutable(run/'attempts'/(attempt+'.started.json'),dict(**identity(plan,by[identifier]),attempt_id=attempt,started_utc=utc_now()))
                            save();inbox.put((by[identifier],attempt,active['deadline']))
                        try:
                            message = outbox.get(timeout=.05)
                        except queue.Empty:
                            message = None
                        if message:
                            if message['event']=='ready':
                                is_ready = True;s.update(stage='idle',warmup=message['warmup'],context_metadata=message['context_metadata']);save()
                            elif message['event']=='progress':
                                save();checkpoint = read_sealed(ROOT/message['path']);event = checkpoint['progress']
                                print(f"{message['id']}: {event.get('stage',event.get('event','checkpoint'))}; {checkpoint['elapsed_seconds']:.1f}s",flush=True)
                            elif message['event']=='finished':
                                require(active and active['id']==message['id'], 'Unexpected completion')
                                case = cached(plan,by[message['id']],run);require(case is not None,'No durable completed case')
                                states[message['id']] = case['status']
                                immutable(run/'attempts'/(active['attempt_id']+'.finished.json'),dict(plan_id=plan['plan_id'],id=message['id'],attempt_id=active['attempt_id'],status=case['status'],case_sha256=sha256(run/'cases'/(message['id']+'.json'))))
                                print(f"{message['id']}: {case['status']}; {case['elapsed_seconds']:.3f}s",flush=True)
                                active = None;s.update(active_id=None,active_attempt_id=None,stage='idle');save()
                                write_summary(plan,run,states,sessions,'active')
                                if case['status']!='complete':
                                    reason = 'first_unresolved_case';break
                            elif message['event']=='worker_error':
                                s['worker_error'] = message;reason = 'worker_error';break
                        if not process.is_alive() and (pending or active):
                            reason = 'unexpected_worker_exit';break
                    else:
                        reason = 'complete'
        except KeyboardInterrupt:
            reason = 'keyboard_interrupt'
        except BaseException as exc:
            reason = 'parent_error';s['error'] = type(exc).__name__+': '+str(exc)
        finally:
            # A cleanup failure deliberately leaves the durable session OPEN;
            # next invocation confirms death before conservative recovery debit.
            if handle is not None:
                process,job,inbox = handle
                if process.pid is not None:
                    if active is None and process.is_alive():
                        inbox.put(None);process.join(2)
                    life.stop_child(handle)
                    life.ensure_old_worker_dead(s)
                    s['worker_exitcode'] = process.exitcode
                else:
                    job.close()
            if active:
                entry = next(e for e in plan['entries'] if e['id']==active['id'])
                case = cached(plan,entry,run)
                if case is None and reason in ('case_time_cap','global_budget'):
                    immutable(run/'cases'/(active['id']+'.json'),dict(**identity(plan,entry),attempt_id=active['attempt_id'],status='time_cap',payload=None,error=reason,partial_records_retained=True))
                    states[active['id']] = 'time_cap'
                elif case:
                    states[active['id']] = case['status']
                finished = run/'attempts'/(active['attempt_id']+'.finished.json')
                if not finished.exists():
                    immutable(finished,dict(plan_id=plan['plan_id'],id=active['id'],attempt_id=active['attempt_id'],status=states[active['id']] if case or states[active['id']]=='time_cap' else 'interrupted',reason=reason))
            if reason in ('setup_time_cap','worker_error','unexpected_worker_exit','parent_error'):
                if not (run/'fatal.json').exists():
                    immutable(run/'fatal.json',dict(plan_id=plan['plan_id'],reason=reason,session_id=s['session_id'],recorded_utc=utc_now(),error=s.get('error'),worker_error=s.get('worker_error'),policy='Stop for review; ordinary resume may not silently retry a fatal execution failure'))
            measured = time.perf_counter()-began
            s.update(closed=True,measured_elapsed_seconds=measured,closure_allowance_seconds=CLOSURE_ALLOWANCE_SECONDS,
                     elapsed_seconds=measured+CLOSURE_ALLOWANCE_SECONDS,active_id=None,active_attempt_id=None,
                     stage='closed',ended_utc=utc_now(),reason=reason,
                     accounting_scope='Invocation before preflight through owned-worker cleanup and final payload adoption; final ledger/summary writes debited one operational second, not a proved latency bound')
            sealed_json(ledger_path,dict(plan_id=plan['plan_id'],sessions=sessions))
            final = write_summary(plan,run,states,sessions,reason)
        return 0 if final['all_complete'] else 130 if reason in ('keyboard_interrupt','stop_requested') else 2


def main():
    began = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare-only',action='store_true')
    modes.add_argument('--resume',action='store_true')
    modes.add_argument('--status',action='store_true')
    parser.add_argument('--stage',choices=['benchmark','full'],default='full')
    args = parser.parse_args()
    run = AREA/('results/benchmark' if args.stage=='benchmark' else 'results/full_calibration')
    config = AREA/('configs/benchmark_protocol_frozen.json' if args.stage=='benchmark' else 'configs/full_protocol_frozen.json')
    if args.status:
        value = read_sealed(run/'summary.json') if (run/'summary.json').exists() else dict(prepared=False)
        if (run/'sessions.json').exists():
            ledger = read_sealed(run/'sessions.json')
            value['last_durable_charged_seconds'] = sum(s['elapsed_seconds'] for s in ledger['sessions'])
        value['status_scope'] = 'Most recent completed checkpoint; not a live elapsed-time meter'
        print(value)
        return 0
    require(os.name=='nt','Production runner requires Windows job containment')
    if args.prepare_only:
        print(dict(plan_id=prepare(config,run)['plan_id'],saved_history_fit_launched=False))
        return 0
    require((run/'plan.json').exists(),'Run --prepare-only before manual evaluation')
    plan=read_sealed(run/'plan.json')
    require(plan['contract']['stage']==args.stage,'Requested stage differs from prepared plan')
    return execute(plan,run,began)


if __name__=='__main__':
    mp.freeze_support()
    raise SystemExit(main())
