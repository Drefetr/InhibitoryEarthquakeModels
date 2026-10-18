"""Strict terminal collector for a separately launched full staged evaluation.

Reads validated artifacts only. Never prepares a run, evaluates a likelihood,
fits, simulates, or imports benchmark fits into the full cohort.
"""
from pathlib import Path
import argparse
from collections import Counter
import csv
import importlib.util
import io
import math
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / 'iteration_21'
sys.path.insert(0, str(ROOT / 'iteration_6/code'))
from checkpoint_io import RunLock, atomic_bytes, canonical_hash, read_json, read_sealed, sealed_json, sha256, utc_now

CONFIG_SHA = '91d98445f0b722bf960fa0ea331f10a592547f23ae35a74723ad57cbd7a77d65'
ADAPTER_SHA = '6ca14b20809ca282bf94c2eec362ab04f8f75701a0d72c1c9f9c7b125975e0c8'
CORE_SHA = '338145457dd54d90ac929fda32224adcb7c02870e884724341f6222a4fce6f44'
OLD_ANALYSIS_SHA = 'c1ecd219c66b63997bf04b816dca80bb855465b79051e35b0f696a562d4ddb86'
ORACLE_POINTER_SHA = '14401a9f52c5d9f8311641f0bb9c96593e2455d582552d67c414640cb6e3f54b'
INVENTORY_SHA = '2d7ef9883147fa1d6bed60b007eecf5dd5b97ff16a4f11cfb75625be29fdad60'
PLAN_KEYS = ('config_path', 'config_sha256', 'contract', 'sources', 'ids', 'entries')
IDS = [f'null_{i:04d}' for i in range(1999)] + [f'new_r{i:03d}' for i in range(150)]
METHODS = ('known_known', 'background_estimated')
REQUIRED_SOURCES = {
    'iteration_21/code/run_staged_full.py', 'iteration_21/code/benchmark_staged.py',
    'iteration_21/code/staged_fit.py', 'iteration_21/code/analyse_full_staged.py',
    'iteration_21/code/analyse_staged.py', 'iteration_20/code/restricted_fit.py',
    'iteration_20/code/restricted_history.py', 'iteration_18/code/cuda_oracle.py',
    'iteration_17/code/oracle_likelihood.py', 'iteration_6/code/checkpoint_io.py',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def module(path, name, expected=None):
    if expected is not None:
        require(sha256(ROOT / path) == expected, 'Immutable implementation changed: ' + path)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
    return sys.modules[name]


def adapter():
    return module('iteration_21/code/analyse_staged.py', '_iteration21_fixed_statistics', ADAPTER_SHA)


def validate_terminal(plan, summary, ledger, index):
    """No score reads before strict whole-cohort completion and closed sessions."""
    require(summary['plan_id'] == ledger['plan_id'] == index['plan_id'] == plan['plan_id'], 'Terminal ledger/plan differs')
    sessions = ledger['sessions']
    require(sessions and summary['sessions_closed'] is True and summary['active'] is False and
            not summary['active_ids'] and not summary.get('paused', False) and
            all(s['closed'] is True and s['stage'] == 'closed' and not s.get('active_id') and
                not s.get('active_attempt_id') and not s.get('active_attempt') for s in sessions), 'Active or paused run')
    require(plan['ids'] == IDS and index['states'] == {i: 'complete' for i in IDS} and
            summary['counts'] == {'complete': 2149} and summary['total_planned'] == 2149 and
            summary['all_primary_complete'] is True and summary['terminal_complete'] is True,
            'Full2149 completion required; pending or failed prefix refused')
    require(index.get('method_states') == {i: {m: 'validated' for m in METHODS} for i in IDS} and
            summary.get('method_counts') == {m: {'validated': 2149} for m in METHODS}, 'Complete method index/counts differ')
    charge = sum(s['elapsed_seconds'] for s in sessions)
    require(math.isfinite(charge) and charge >= 0 and charge == summary['charged_seconds'], 'Session accounting differs')
    return charge


def validate_plan(plan, run):
    require(plan['plan_id'] == canonical_hash({k: plan[k] for k in PLAN_KEYS}), 'Canonical full plan differs')
    require(plan['config_sha256'] == CONFIG_SHA and sha256(ROOT / plan['config_path']) == CONFIG_SHA and
            read_json(ROOT / plan['config_path']) == plan['contract'], 'Frozen full configuration differs')
    c = plan['contract']; e = c['execution']
    require(c['status'] == 'frozen_manual_staged_full_evaluation_before_execution' and
            e['ids'] == IDS and e['whole_parent_wall_seconds'] == 28800 and e['per_history_total_wall_seconds'] == 600 and
            e['setup_wall_seconds'] == 60 and e['cleanup_reserve_seconds'] == 15 and
            e['fresh_all_histories'] is True and e['import_profile_fits'] is False and
            e['analysis_requires_all2149_complete'] is True, 'Manual full-run policy differs')
    require(REQUIRED_SOURCES.issubset(plan['sources']), 'Critical runtime authority omitted')
    require(plan['sources']['iteration_21/code/staged_fit.py'] == CORE_SHA and
            plan['sources']['iteration_21/code/analyse_staged.py'] == ADAPTER_SHA, 'Scientific stage/statistical adapter changed')
    for path, digest in plan['sources'].items():
        require(sha256(ROOT / path) == digest and sha256(run / 'source_snapshots' / path) == digest, 'Runtime source changed: ' + path)
    for authority in c['authorities']:
        require(sha256(ROOT / authority['path']) == authority['sha256'], 'Prior authority changed: ' + authority['path'])
    freeze = read_json(AREA / 'logs/full_runtime_source_freeze.json')
    require(freeze['plan_id'] == plan['plan_id'] and freeze['validation_passed'] is True and freeze['sources'] == plan['sources'],
            'Full runtime freeze differs')
    inv_ref = next(r for r in c['authorities'] if '/inventory_' in r['path'])
    require(inv_ref['sha256'] == INVENTORY_SHA, 'Inventory authority differs')
    inventory = read_sealed(ROOT / inv_ref['path'])
    entries = validate_inventory_entries(inventory, plan['entries'])
    adapter().validate_prospective_protocol(c)
    return entries


def validate_inventory_entries(inventory, planned_entries):
    """Actual inventory schema check without reading any scientific scores."""
    entries = {r['id']: r for r in inventory['entries']}
    require(len(entries) == len(inventory['entries']) == 2299 and planned_entries == [entries[i] for i in IDS] and
            len({entries[i]['seed'] for i in IDS}) == 2149, 'Fixed identity/seed inventory differs')
    require([i for i in IDS if entries[i]['clock_mode'] == 'paired_clock'] == ['null_0841'] and
            all(entries[i]['clock_mode'] == ('paired_clock' if i == 'null_0841' else 'original_projected_clock') for i in IDS),
            'Clock conventions differ')
    return entries


def load_oracle(entries, law):
    """Read accepted19 raw scores and recalibrate, never reuse its threshold."""
    pointer_path = ROOT / 'iteration_19/results/full_cuda/latest_analysis.json'
    require(sha256(pointer_path) == ORACLE_POINTER_SHA, 'Completed oracle pointer changed')
    pointer = read_sealed(pointer_path); folder = ROOT / pointer['path']
    require(folder.resolve().is_relative_to(ROOT / 'iteration_19/results/analysis'), 'Oracle analysis path differs')
    for name, digest in pointer['files'].items():
        require(sha256(folder / name) == digest, 'Completed oracle analysis changed')
    published = read_sealed(folder / 'summary.json'); manifest = read_sealed(folder / 'manifest.json')
    require(pointer['analysis_signature'] == published['analysis_signature'] == manifest['analysis_signature'], 'Oracle analysis signature differs')
    api = module('iteration_19/code/analyse_full.py', '_iteration21_accepted_oracle19', OLD_ANALYSIS_SHA)
    old_run = ROOT / 'iteration_19/results/full_cuda'
    plan = read_sealed(old_run / 'plan.json'); summary = read_sealed(old_run / 'summary.json')
    ledger = read_sealed(old_run / 'sessions.json'); index = read_sealed(old_run / 'case_index.json')
    require(ledger['plan_id'] == plan['plan_id'] == pointer['plan_id'], 'Oracle ledger plan differs')
    api.validate_terminal(plan, summary, ledger['sessions'], index)
    api.validate_plan(plan)
    require(plan['protocol']['phase1']['law'] == law, 'Oracle and estimated-eta supplied laws differ')
    require(plan['entries'] == [entries[i] for i in IDS] and set(published['score_file_sha256']) == set(IDS) and
            manifest['source_scores'] == published['score_file_sha256'], 'Oracle catalogue/cohort authority differs')
    values = {}; hashes = {}
    for entry in plan['entries']:
        i = entry['id']; path = old_run / 'cases' / (i + '.json')
        require(sha256(path) == published['score_file_sha256'][i], 'Accepted oracle case bytes changed')
        record = read_sealed(path)
        require(record['status'] == 'complete', 'Accepted oracle case became unresolved')
        values[i] = api.validate_cuda_payload(record, entry, plan)
        api.check_import(record, entry, plan)
        require(values[i] is not None, 'Oracle numerical gate unresolved')
        hashes[i] = sha256(path)
    return values, dict(plan_id=plan['plan_id'], pointer_sha256=ORACLE_POINTER_SHA,
                        summary_sha256=pointer['files']['summary.json'], manifest_sha256=pointer['files']['manifest.json'],
                        source_scores=hashes)


def collect_locked(run):
    plan = read_sealed(run / 'plan.json'); summary = read_sealed(run / 'summary.json')
    ledger = read_sealed(run / 'sessions.json'); index = read_sealed(run / 'case_index.json')
    charge = validate_terminal(plan, summary, ledger, index)  # before any stage/statistic load
    entries = validate_plan(plan, run)
    runtime = module('iteration_21/code/run_staged_full.py', '_iteration21_full_metadata', plan['sources']['iteration_21/code/run_staged_full.py'])
    require(tuple(runtime.PLAN_KEYS) == PLAN_KEYS and set(plan['sources']) == set(runtime.SOURCES), 'Full runner schema/source inventory differs')
    require({p.stem for p in (run / 'cases').glob('*.json')} == set(IDS), 'Missing or extra terminal case authority')
    values = {m: {} for m in METHODS}; case_hashes = {}; method_hashes = {}
    for i in IDS:
        entry = entries[i]
        for ref in entry['files'].values():
            require(sha256(ROOT / ref['path']) == ref['sha256'], 'Original input/authority changed: ' + ref['path'])
        case = runtime.cached(plan, entry, run)  # exact NPZ-bound pure checkpoint validators
        require(case is not None and case['status'] == 'complete' and not case.get('import_provenance'), 'Invalid/imported full case')
        require([m['id'] for m in case['payload']['methods']] == list(METHODS), 'Full method quota differs')
        case_hashes[i] = sha256(run / 'cases' / (i + '.json'))
        method_hashes[i] = {}
        for m in case['payload']['methods']:
            mid = m['id']; validated = runtime.read_stage(plan, entry, run, mid, 'validated')
            require(validated is not None and validated['payload'] == m, 'Case/stage payload differs')
            values[mid][i] = adapter().method_score(m)
            method_hashes[i][mid] = {kind: sha256(runtime.method_path(run, i, mid, kind)) for kind in ('optimized', 'validated')}
    oracle, oracle_authority = load_oracle(entries, plan['contract']['law'])
    terminal = dict(execution_kind='full_staged_study', separately_authorized_full_study=True,
                    sessions_closed=True, active=False, active_ids=[], terminal=True, ids=IDS, states=index['states'])
    result = adapter().analyse_terminal_values(plan['contract'], values, oracle, terminal)
    result.update(scope='Conditional realized-bank frequencies under the fixed supplied law; exploratory comparison on previously inspected histories. No plug-in-bootstrap power, additive nuisance attribution or global optimization certificate.',
                  inference_limitations=['Intervals condition on each realized reference bank and fixed generating law; bank and generating-parameter uncertainty are omitted.',
                                        'Procedure development followed earlier inspection of these histories; intervals are not selection-adjusted confirmation.',
                                        'Finite-start optimization and empirical numerical sensitivity gates do not prove global optima or rigorous integration bounds.'],
                  created_utc=utc_now(), plan_id=plan['plan_id'], config_sha256=CONFIG_SHA,
                  runner_summary_sha256=sha256(run / 'summary.json'), runner_sessions_sha256=sha256(run / 'sessions.json'),
                  runner_index_sha256=sha256(run / 'case_index.json'), plan_sha256=sha256(run / 'plan.json'),
                  source_case_sha256=case_hashes, source_method_sha256=method_hashes, oracle_authority=oracle_authority,
                  summarizer_sha256=sha256(Path(__file__)), adapter_sha256=ADAPTER_SHA,
                  charged_seconds=charge, budget_overrun_seconds=max(0., charge - 28800.), planned=2149,
                  complete_method_history_pairs=4298, prescribed_optimizer_records=12894)
    return result


def resolved(result):
    return all(result['methods'][name][cohort]['unknown'] == 0 for name in ('oracle',) + METHODS
               for cohort in ('heldout_null_size', 'alternative_frequency'))


def publish(result, run):
    signature = canonical_hash({key: result[key] for key in ('plan_id', 'config_sha256', 'plan_sha256',
        'source_case_sha256', 'source_method_sha256', 'oracle_authority', 'summarizer_sha256', 'adapter_sha256', 'charged_seconds')})
    latest = run / 'latest_analysis.json'
    if latest.exists():
        old = read_sealed(latest)
        if old['analysis_signature'] == signature:
            folder = (ROOT / old['path']).resolve()
            require(folder.is_relative_to(AREA / 'results/analysis'), 'Analysis path outside iteration21')
            for name, digest in old['files'].items():
                require(sha256(folder / name) == digest, 'Completed analysis bytes changed')
            saved = read_sealed(folder / 'summary.json')
            require(saved['analysis_signature'] == signature, 'Completed analysis signature differs')
            return folder, saved, True
    folder = AREA / 'results/analysis' / utc_now().replace(':', '').replace('.', '').replace('+', '_')
    folder.mkdir(parents=True, exist_ok=False)
    result = dict(result, analysis_signature=signature)
    sealed_json(folder / 'summary.json', result)
    stream = io.StringIO(newline='')
    names = ('oracle',) + METHODS
    fields = ['id'] + [name + '_' + key for name in names for key in ('rank', 'reject', 'reason')]
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    require([r['id'] for r in result['outcomes']] == IDS[999:], 'Exact1150 query quota differs')
    for row in result['outcomes']:
        writer.writerow(dict(id=row['id'], **{name + '_' + key: row[name][key] for name in names for key in ('rank', 'reject', 'reason')}))
    atomic_bytes(folder / 'outcomes.csv', stream.getvalue().encode('utf-8'))
    receipts = {'runner_summary.json': 'summary.json', 'runner_sessions.json': 'sessions.json', 'runner_case_index.json': 'case_index.json'}
    for name, original in receipts.items():
        atomic_bytes(folder / name, (run / original).read_bytes())
    files = {name: sha256(folder / name) for name in ('summary.json', 'outcomes.csv', *receipts)}
    sealed_json(folder / 'manifest.json', dict(plan_id=result['plan_id'], analysis_signature=signature, files=files,
        source_case_sha256=result['source_case_sha256'], source_method_sha256=result['source_method_sha256'],
        oracle_authority=result['oracle_authority'], summarizer_sha256=result['summarizer_sha256'], adapter_sha256=ADAPTER_SHA))
    files['manifest.json'] = sha256(folder / 'manifest.json')
    sealed_json(latest, dict(plan_id=result['plan_id'], analysis_signature=signature, path=folder.relative_to(ROOT).as_posix(), files=files))
    return folder, read_sealed(folder / 'summary.json'), False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=AREA / 'results/full_staged')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args(); run = args.run_dir.resolve()
    require(run.is_relative_to(AREA / 'results'), 'Run outside iteration21/results')
    if args.status:
        print(read_sealed(run / 'latest_analysis.json') if (run / 'latest_analysis.json').exists() else {'analysis_available': False})
        return 0
    require((run / 'plan.json').exists(), 'Run not prepared; collector never prepares runs')
    with RunLock(run / 'full.lock'):
        result = collect_locked(run)
        folder, result, reused = publish(result, run)
    print(folder); print('analysis_reused=' + str(reused).lower())
    return 0 if resolved(result) else 2


if __name__ == '__main__':
    raise SystemExit(main())
