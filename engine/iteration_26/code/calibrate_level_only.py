"""Pure terminal calibration of a complete, frozen level-only run.

No fitting, history evaluation, simulation or CUDA context is created here.
The CLI refuses a benchmark, incomplete cohort, open ledger or changed source.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import io
import math
import sys
import time
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / 'iteration_26'
sys.path.insert(0, str(ROOT / 'iteration_6/code'))
from checkpoint_io import (RunLock, atomic_bytes, canonical_hash, read_json,
                           read_sealed, sealed_json, sha256, utc_now)

REFERENCES = [f'null_{i:04d}' for i in range(999)]
CHECKS = [f'null_{i:04d}' for i in range(999, 1999)]
ALTERNATIVES = [f'new_r{i:03d}' for i in range(150)]
IDS = REFERENCES + CHECKS + ALTERNATIVES
QUERIES = CHECKS + ALTERNATIVES
METHODS = ('level_only', 'known_known', 'background_estimated')
STATS_PATH = 'iteration_17/code/analyse_oracle_v1_1.py'
STATS_SHA = '5d6a987185484295fbc172f81783ebff80f10459e695a130efe53005c19d612d'
PRIOR_PLAN_PATH = 'iteration_21/results/full_staged/plan.json'
PRIOR_PLAN_SHA = 'a0d09cd8918e686bb07916fd166e81cf300a23ebcc0c180a118cedf2c3758df7'
OLD_ANALYSIS_PATH = 'iteration_21/results/analysis/2026-10-17T004327174461_0000/summary.json'
OLD_ANALYSIS_SHA = '56da11135e920e0ced7ee5173d0bfe064a3802ecaf566d6ba7b6d11b75123823'
ANALYSIS_SECONDS = 600.
CLOSURE_ALLOWANCE_SECONDS = 1.
NUMERICAL_ENVELOPES_ARE_RIGOROUS_ERROR_BOUNDS = False


def require(ok, message):
    if not ok:
        raise ValueError(message)


def module(relative, name, expected):
    path = ROOT / relative
    require(sha256(path) == expected, 'Pinned source changed: ' + relative)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        api = importlib.util.module_from_spec(spec)
        sys.modules[name] = api
        spec.loader.exec_module(api)
    require(Path(sys.modules[name].__file__).resolve() == path.resolve(),
            'Unexpected imported module: ' + name)
    return sys.modules[name]


def statistics():
    return module(STATS_PATH, '_iteration26_frozen_rank_authority', STATS_SHA)


def validate_roles(protocol):
    f = protocol['future_inference']
    require(f['reference_ids'] == REFERENCES and f['heldout_null_ids'] == CHECKS and
            f['alternative_ids'] == ALTERNATIVES, 'Exact 999/1000/150 roles differ')
    require(protocol['stage'] == 'full' and protocol['case_ids'] == IDS,
            'Calibration requires the full plan; benchmark is cost evidence only')
    c = protocol['calibration']
    require(c['center'] == 'raw_original_backend_contrast16_statistic' and
            c['alpha'] == .05 and c['ties'] == 'greater_than_or_equal' and
            c['epsilon_floor'] == 1e-6 and
            c['epsilon'] == 'max(1e-6,2*sum_phase_abs(CPU32_NLL-original_backend_CUDA16_NLL))' and
            c['full_cohort_barrier'] is True and
            c['analysis_parent_wall_seconds'] == ANALYSIS_SECONDS, 'Frozen rank policy differs')


def validate_terminal(plan, summary, ledger):
    """Run before reading any history statistic or accepted old decision."""
    validate_roles(plan['contract'])
    require(plan['ids'] == IDS and [e['id'] for e in plan['entries']] == IDS,
            'Full identity/order quota differs')
    require(summary['plan_id'] == ledger['plan_id'] == plan['plan_id'],
            'Terminal plan identity differs')
    require(summary['stage'] == 'full' and summary['reason'] == 'complete' and
            summary['total_planned'] == 2149 and
            summary['new_models'] == ['level_only'] and
            summary['all_complete'] is True and summary['terminal_complete'] is True and
            summary['sessions_closed'] is True and summary['active'] is False and
            summary['active_ids'] == [], 'Execution is not complete, closed and inactive')
    require(summary['states'] == dict.fromkeys(IDS, 'complete') and
            summary['counts'] == {'complete': 2149}, 'Missing, failed or extra identities')
    sessions = ledger['sessions']
    require(sessions and len({s['session_id'] for s in sessions}) == len(sessions),
            'Missing or duplicate closed sessions')
    for s in sessions:
        require(s['closed'] is True and s.get('stage') == 'closed' and
                s.get('active_id') is None and s.get('active_attempt_id') is None and
                math.isfinite(s['elapsed_seconds']) and s['elapsed_seconds'] >= 0,
                'Open/invalid session or active attempt')
    require(not any(s.get('error') or s.get('worker_error') for s in sessions),
            'Retained parent/worker error requires separate review')
    launched = [s for s in sessions if s.get('worker_identity') is not None]
    require(launched and launched[-1].get('worker_exitcode') == 0 and
            launched[-1].get('reason') == 'complete',
            'Final launched worker generation did not complete and exit zero')
    charge = sum(s['elapsed_seconds'] for s in sessions)
    require(math.isfinite(summary['charged_seconds']) and
            abs(summary['charged_seconds'] - charge) <= 1e-8 and
            charge <= plan['contract']['resources']['total_parent_wall_seconds'],
            'Terminal resource charge differs or exceeds frozen full cap')
    return charge


def method_score(payload):
    """Arithmetic only; the caller must first invoke the full pure D validator."""
    require(payload.get('complete') is True and payload.get('status') == 'complete' and
            [m['id'] for m in payload['methods']] == ['level_only'],
            'Incomplete or wrong procedure')
    m = payload['methods'][0]
    require(m.get('complete') is True and m.get('status') == 'complete',
            'Optimization-only method is not analysis-ready')
    rows = m['selected_objectives']
    require(len(rows) == 2 and [r['phase'] for r in rows] == ['null', 'alternative'],
            'Selected phase quota differs')
    deltas = []
    controls16 = {r['label']: r for r in payload['old_backend_controls']}
    controls32 = {r['label']: r for r in payload['cpu32_controls']}
    for row in rows:
        label = 'new_level_only_' + row['phase']
        v16 = controls16[label]['old_backend16_nll']
        v32 = controls32[label]['independent32_nll']
        d = abs(v32 - v16)
        require(row['cuda16_nll'] == v16 and row['cpu32_nll'] == v32 and
                row['absolute_difference'] == d and row['passed'] is True and
                math.isfinite(d) and d < 1e-4, 'Original-backend selected gate differs')
        deltas.append(d)
    t16, t32 = (m['contrast' + str(o)]['statistic'] for o in (16, 32))
    require(math.isfinite(t16) and math.isfinite(t32) and
            abs(t16 - t32) < 1e-4 and min(t16, t32) >= -1e-4,
            'Direct order/nesting gate differs')
    score = statistics().score_pair(t16, 2 * sum(deltas))
    recorded = payload['calibration_score']
    require(recorded['central_order'] == 16 and recorded['epsilon_floor'] == 1e-6 and
            recorded['statistic'] == score[0] and recorded['numerical_envelope'] == score[1] and
            recorded['interval'] == [score[0] - score[1], score[0] + score[1]],
            'Recorded empirical score/envelope differs')
    return score


def relabel_paired(pairs, first, second):
    old = statistics().paired(pairs)
    return dict(requested=old['requested'], complete=old['complete'],
                first_method=first, second_method=second,
                table=dict(neither=old['table']['neither'],
                           first_only=old['table']['oracle_only'],
                           second_only=old['table']['fitted_only'],
                           both=old['table']['both'], unknown=old['table']['unknown']),
                first_minus_second=old['oracle_minus_fitted'],
                descriptive_MCSE=old['descriptive_MCSE'],
                accounting_bounds=old['accounting_bounds'])


def validate_old_outcomes(rows):
    require(len(rows) == 1150 and [r['id'] for r in rows] == QUERIES,
            'Saved old query identities/order differ')
    result = {}
    for row in rows:
        result[row['id']] = {}
        for name in METHODS[1:]:
            d = row[name]
            require(set(('rank', 'rank_sensitivity_range', 'reject', 'reason')).issubset(d) and
                    (d['reject'] is None or type(d['reject']) is bool),
                    'Invalid saved old own-bank decision')
            result[row['id']][name] = d
    return result


def analyse_values(protocol, values, old_rows, terminal_validated=False, deadline=None):
    """Pure literal-testable arithmetic; never used on a benchmark or prefix."""
    require(terminal_validated is True, 'Terminal validation must precede rank arithmetic')
    validate_roles(protocol)
    require(set(values) == set(IDS), 'Full fixed-denominator score map required')
    require(all(values[i] is not None for i in REFERENCES),
            'Incomplete 999-reference bank: all decisions withheld')
    stats = statistics()
    for value in values.values():
        if value is not None:
            require(len(value) == 2, 'Invalid score pair')
            stats.score_pair(*value)
    old = validate_old_outcomes(old_rows)
    bank = [values[i] for i in REFERENCES]
    outcomes = []
    for i in QUERIES:
        if deadline is not None:
            require(time.perf_counter() < deadline, 'Pure calibration budget exhausted')
        outcomes.append(dict(id=i, level_only=stats.classify(values[i], bank), **old[i]))
    by = {r['id']: r for r in outcomes}
    methods = {}
    for name in METHODS:
        methods[name] = dict(reference_requested=999, reference_resolved=999,
                             reference_complete=True,
                             heldout_null_size=stats.tally([by[i][name]['reject'] for i in CHECKS]),
                             alternative_frequency=stats.tally([by[i][name]['reject'] for i in ALTERNATIVES]))
    paired = {}
    for second in METHODS[1:]:
        paired['level_only_minus_' + second] = {
            label: relabel_paired([(by[i]['level_only']['reject'], by[i][second]['reject'])
                                  for i in identities], 'level_only', second)
            for label, identities in [('heldout_null_size', CHECKS),
                                      ('alternative_frequency', ALTERNATIVES)]}
    return dict(methods=methods, paired=paired, outcomes=outcomes,
                planned=2149, reference_requested=999, query_requested=1150,
                statistical_authority_sha256=STATS_SHA,
                numerical_envelopes_are_rigorous_error_bounds=False,
                confirmatory_exact_nominal_size_claim=False,
                selection_adjusted_intervals=False,
                scope='Exploratory matched-null reused-data calibration; conditional realized-bank frequencies under one fixed law.',
                limitations=[
                    'Recipe choice followed inspection of the reused histories; the full selection-plus-scoring process is not proved exchangeable.',
                    'Nominal conditional CP intervals omit reference-bank, generating-law, numerical-geometry and adaptive-selection uncertainty.',
                    'Shared-bank decisions are dependent unconditionally; a realized bank conditional rate need not equal or be below .05.',
                    'Case-specific clock/repair provenance and finite searches limit ideal exchangeability and optimization claims.',
                    'No equal-size, causal power attribution, physical-mechanism or universal information-limit claim.'])


def checkpoint(deadline):
    require(time.perf_counter() < deadline, 'Pure calibration budget exhausted')


def collect_locked(run, deadline):
    plan = read_sealed(run / 'plan.json')
    summary = read_sealed(run / 'summary.json')
    ledger = read_sealed(run / 'sessions.json')
    require(not (run / 'fatal.json').exists(), 'Retained fatal barrier blocks calibration')
    charge = validate_terminal(plan, summary, ledger)  # before reading scientific values
    require(canonical_hash({k: plan[k] for k in
            ('config_path', 'config_sha256', 'contract', 'sources', 'ids', 'entries', 'benchmarks')})
            == plan['plan_id'], 'Plan canonical seal differs')
    runtime_path = 'iteration_26/code/run_level_calibration.py'
    runtime = module(runtime_path, '_iteration26_terminal_runtime', plan['sources'][runtime_path])
    runtime.guard(plan)  # includes immutable review/source freeze and benchmark resource gate
    life = runtime.lifetime()
    for session in ledger['sessions']:
        identity = session.get('worker_identity')
        require(identity is None or life.process_identity(identity['pid']) != identity,
                'Owned worker generation remains alive')
    checkpoint(deadline)
    require(sha256(ROOT / PRIOR_PLAN_PATH) == PRIOR_PLAN_SHA, 'Uniform old21 plan changed')
    prior = read_sealed(ROOT / PRIOR_PLAN_PATH)
    require(prior['ids'] == IDS and prior['entries'] == plan['entries'] and
            prior['contract']['law'] == plan['contract']['law'], 'Old21 full cohort/law changed')
    require({p.stem for p in (run / 'cases').glob('*.json')} == set(IDS),
            'Missing or extra full case authority')
    fitter_path = 'iteration_26/code/level_only_fit.py'
    fitter = module(fitter_path, '_iteration26_terminal_fitter', plan['sources'][fitter_path])
    values = {}; case_hashes = {}; baseline_hashes = {}; receipts = []
    for entry in plan['entries']:
        checkpoint(deadline)
        i = entry['id']
        for ref in entry['files'].values():
            require(sha256(ROOT / ref['path']) == ref['sha256'], 'Changed history source: ' + ref['path'])
        b = plan['benchmarks'][i]
        require(b['path'] == f'iteration_21/results/full_staged/cases/{i}.json' and
                b['plan_id'] == prior['plan_id'] and sha256(ROOT / b['path']) == b['sha256'],
                'Uniform old21 floor authority differs')
        baseline = read_sealed(ROOT / b['path'])
        case = runtime.cached(plan, entry, run)
        require(case is not None and case['status'] == 'complete' and
                not case.get('import_provenance'), 'Invalid/imported full case')
        payload = case['payload']
        # runtime.cached has already invoked the source-pinned pure D validator;
        # repeating the full trace pass would consume analysis budget without a new check.
        binding = payload['baseline_binding']
        require(binding['id'] == i and binding['plan_id'] == prior['plan_id'] and
                binding['canonical_case_sha256'] == canonical_hash(baseline),
                'Payload detached from actual full old21 floor record')
        require(payload['input_binding']['clock_mode'] == entry['clock_mode'] and
                payload['input_binding']['sources'] == fitter.profile.authority.source_pins(),
                'Input clock/numerical source binding differs')
        old_kk = baseline['payload']['methods'][0]
        require(old_kk['id'] == 'known_known', 'Actual baseline first procedure differs')
        compact = payload['baseline_method']
        for phase in ('null', 'alternative'):
            require(compact[phase]['pars'] == old_kk[phase]['pars'] and
                    compact['saved_' + phase + '_nll'] == old_kk[phase]['nll'],
                    'Compact floor detached from actual old21 selected point')
        require(all(compact['saved_contrast' + str(o)] == old_kk['contrast' + str(o)]
                    for o in (16, 32)), 'Compact saved direct contrast detached from old21')
        # Input binding hashes are checked directly: no history construction or likelihood evaluation.
        arrays = payload['input_binding']['arrays']
        with np.load(ROOT / entry['files']['catalogue.npz']['path'], allow_pickle=False) as saved:
            require(set(arrays) == set(saved.files), 'Input array quota differs')
            for name in saved.files:
                value = saved[name]
                require(arrays[name]['shape'] == list(value.shape) and
                        arrays[name]['dtype'] == value.dtype.str and
                        arrays[name]['sha256'] == hashlib.sha256(
                            np.ascontiguousarray(value).tobytes()).hexdigest(),
                        'Payload array bytes/dtype/shape differ: ' + i + '/' + name)
        values[i] = method_score(payload)
        case_hashes[i] = sha256(run / 'cases' / (i + '.json'))
        baseline_hashes[i] = b['sha256']
        receipts.append(dict(id=i, role=('null_reference' if i in REFERENCES else
                                       'heldout_null' if i in CHECKS else 'alternative'),
                             statistic16=values[i][0], numerical_envelope=values[i][1],
                             statistic32=payload['methods'][0]['contrast32']['statistic'],
                             case_sha256=case_hashes[i], baseline_sha256=b['sha256']))
    checkpoint(deadline)
    require(sha256(ROOT / OLD_ANALYSIS_PATH) == OLD_ANALYSIS_SHA, 'Accepted old own-bank outcomes changed')
    old = read_sealed(ROOT / OLD_ANALYSIS_PATH)
    require(old['plan_id'] == prior['plan_id'] and
            old['source_case_sha256'] == baseline_hashes,
            'Accepted old analysis plan or exact floor-case authority differs')
    result = analyse_values(plan['contract'], values, old['outcomes'], True, deadline)
    result.update(created_utc=utc_now(), plan_id=plan['plan_id'],
                  config_sha256=plan['config_sha256'], source_hashes=plan['sources'],
                  plan_sha256=sha256(run / 'plan.json'), source_case_sha256=case_hashes,
                  source_baseline_sha256=baseline_hashes, score_receipts=receipts,
                  prior_own_bank_analysis=dict(path=OLD_ANALYSIS_PATH, sha256=OLD_ANALYSIS_SHA,
                                              plan_id=prior['plan_id'], recomputed=False),
                  runner_summary_sha256=sha256(run / 'summary.json'),
                  runner_sessions_sha256=sha256(run / 'sessions.json'),
                  full_run_charged_seconds=charge,
                  summarizer_sha256=sha256(Path(__file__)),
                  analysis_budget_seconds=ANALYSIS_SECONDS)
    return result


def publish(result, run, started, deadline):
    checkpoint(deadline)
    folder = AREA / 'results/analysis' / utc_now().replace(':', '').replace('.', '').replace('+', '_')
    folder.mkdir(parents=True, exist_ok=False)
    fields = ['id'] + [m + '_' + k for m in METHODS for k in ('rank', 'reject', 'reason')]
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in result['outcomes']:
        writer.writerow(dict(id=row['id'], **{m + '_' + k: row[m][k]
                                            for m in METHODS for k in ('rank', 'reject', 'reason')}))
    atomic_bytes(folder / 'outcomes.csv', stream.getvalue().encode())
    for name in ('plan.json', 'summary.json', 'sessions.json'):
        checkpoint(deadline)
        atomic_bytes(folder / ('runner_' + name), (run / name).read_bytes())
    result['analysis_signature'] = canonical_hash({k: result[k] for k in
        ('plan_id', 'config_sha256', 'source_case_sha256', 'source_baseline_sha256',
         'prior_own_bank_analysis', 'summarizer_sha256')})
    checkpoint(deadline)
    result.update(analysis_measured_seconds_before_final_publication=time.perf_counter() - started,
                  closure_allowance_seconds=CLOSURE_ALLOWANCE_SECONDS,
                  closure_allowance_is_measured_or_proven_bound=False)
    sealed_json(folder / 'summary.json', result)
    files = {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}
    checkpoint(deadline)
    sealed_json(folder / 'manifest.json', dict(plan_id=result['plan_id'],
        analysis_signature=result['analysis_signature'], files=files,
        source_case_sha256=result['source_case_sha256'],
        source_baseline_sha256=result['source_baseline_sha256'],
        statistical_authority_sha256=STATS_SHA,
        measured_seconds_before_manifest=time.perf_counter() - started,
        closure_allowance_seconds=CLOSURE_ALLOWANCE_SECONDS,
        closure_allowance_is_measured_or_proven_bound=False))
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=AREA / 'results/full_calibration')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--collect', action='store_true',
                        help='Explicit collection request; collection is also the default without --status')
    args = parser.parse_args()
    started = time.perf_counter(); deadline = started + ANALYSIS_SECONDS - CLOSURE_ALLOWANCE_SECONDS
    run = args.run_dir.resolve()
    require(run.is_relative_to(AREA / 'results'), 'Run outside iteration26/results')
    if args.status:
        summary = read_sealed(run / 'summary.json')
        print({k: summary.get(k) for k in ('plan_id', 'stage', 'all_complete', 'terminal_complete',
                                         'sessions_closed', 'active', 'total_planned', 'counts')})
        return 0
    require((run / 'plan.json').exists(), 'Collector never prepares or starts a fit')
    with RunLock(run / 'pilot.lock'):
        result = collect_locked(run, deadline)
        folder = publish(result, run, started, deadline)
    elapsed = time.perf_counter() - started
    require(elapsed <= ANALYSIS_SECONDS, 'Pure analysis exceeded 600 seconds; publication requires resource review')
    print(folder)
    print('analysis_measured_seconds_through_publication_and_lock_release=' + str(elapsed))
    return 0 if all(result['methods'][m][c]['unknown'] == 0 for m in METHODS
                    for c in ('heldout_null_size', 'alternative_frequency')) else 2


if __name__ == '__main__':
    raise SystemExit(main())
