"""Prospective pure two-method analysis adapter. No saved-data collector or CLI.

This is a draft for a separately authorized full study. It cannot analyze the
five-history cost benchmark. A future collector must validate each source,
catalogue and method checkpoint before supplying these explicit value maps.
"""
from pathlib import Path
import hashlib
import importlib.util
import math
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
AUTHORITY = ROOT / 'iteration_17/code/analyse_oracle_v1_1.py'
AUTHORITY_SHA = '5d6a987185484295fbc172f81783ebff80f10459e695a130efe53005c19d612d'
if hashlib.sha256(AUTHORITY.read_bytes()).hexdigest() != AUTHORITY_SHA:
    raise ValueError('Immutable statistical authority changed')
spec = importlib.util.spec_from_file_location('_iteration21_pinned_statistics', AUTHORITY)
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)

METHODS = ('known_known', 'background_estimated')
REFERENCES = [f'null_{r:04d}' for r in range(999)]
CHECKS = [f'null_{r:04d}' for r in range(999, 1999)]
ALTERNATIVES = [f'new_r{r:03d}' for r in range(150)]
ALL = REFERENCES + CHECKS + ALTERNATIVES
QUERIES = CHECKS + ALTERNATIVES
UNRESOLVED = {'unresolved', 'time_cap', 'error', 'interrupted', 'dependency_unknown'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_prospective_protocol(protocol):
    p = protocol['future_inference']
    require(p['reference_ids'] == REFERENCES and p['heldout_null_ids'] == CHECKS and
            p['alternative_ids'] == ALTERNATIVES, 'Fixed calibration split differs')
    require(p['methods_each_own_null_calibration'] is True, 'Each procedure requires its own reference bank')
    require([m['id'] for m in protocol['procedures']] == list(METHODS) and
            [m['prescribed_minimizations'] for m in protocol['procedures']] == [2, 4], 'Procedure family differs')
    require(protocol['law']['alternative_eta'] == .4885010300470942 and
            protocol['law']['beta'] == 2.849401764894274 and
            protocol['law']['horizon_days'] == protocol['law']['coordinate_horizon_days'] == 24394., 'Supplied laws differ')


def validate_terminal_evidence(evidence):
    """Require a separately frozen whole-cohort authorization, never a prefix.

    These flags are not a substitute for a future collector's sealed provenance
    checks. The caller must bind the complete evidence to that future plan.
    """
    require(evidence.get('execution_kind') == 'full_staged_study' and
            evidence.get('separately_authorized_full_study') is True,
            'Engineering benchmark is not a calibrated full study')
    require(evidence.get('sessions_closed') is True and evidence.get('active') is False and
            not evidence.get('active_ids') and evidence.get('terminal') is True,
            'Active, paused or pending-prefix analysis prohibited')
    require(evidence.get('ids') == ALL, 'Whole-cohort identity/order differs')
    states = evidence.get('states')
    require(isinstance(states, dict) and set(states) == set(ALL), 'Missing or unplanned terminal identities')
    require(all(s in {'complete', 'unresolved', 'time_cap', 'error'} for s in states.values()),
            'Nonterminal cohort state')


def validate_value_map(values, name):
    require(isinstance(values, dict) and set(values) == set(ALL), name + ': omitted or extra identities')
    for value in values.values():
        if value is not None:
            require(isinstance(value, (list, tuple)) and len(value) == 2, 'Expected score and numerical envelope')
            stats.score_pair(*value)


def method_score(validated_method):
    """Arithmetic after the future collector validates all checkpoint evidence.

    The envelope is an empirical sensitivity rule, not a proven error bound.
    The direct statistic is preserved, including a tolerated tiny negative.
    """
    require(validated_method.get('complete') is True and validated_method.get('status') == 'complete',
            'Optimization-only artifact is not analysis-ready')
    selected = validated_method.get('selected_objectives', [])
    require(len(selected) == 2 and [g.get('phase') for g in selected] == ['null', 'alternative'],
            'Selected objective quota differs')
    deltas = []
    for g in selected:
        d = abs(g['cpu32_nll'] - g['cuda16_nll'])
        require(math.isfinite(d) and d < 1e-4 and d == g['absolute_difference'] and g['passed'] is True,
                'Unvalidated selected objective')
        deltas.append(d)
    a = validated_method['contrast16']['statistic']
    b = validated_method['contrast32']['statistic']
    require(math.isfinite(a) and math.isfinite(b) and abs(a - b) < 1e-4 and min(a, b) >= -1e-4,
            'Direct GLR order/nesting gate failed')
    return stats.score_pair(a, 2 * sum(deltas))


def relabel_paired(pairs, first, second):
    """Reuse frozen paired arithmetic without misleading oracle/fitted labels."""
    old = stats.paired(pairs)
    return dict(requested=old['requested'], complete=old['complete'],
                first_method=first, second_method=second,
                table=dict(neither=old['table']['neither'], first_only=old['table']['oracle_only'],
                           second_only=old['table']['fitted_only'], both=old['table']['both'],
                           unknown=old['table']['unknown']),
                first_minus_second=old['oracle_minus_fitted'], descriptive_MCSE=old['descriptive_MCSE'],
                accounting_bounds=old['accounting_bounds'])


def analyse_terminal_values(protocol, method_values, oracle_values, terminal_evidence):
    """Pure fixed-denominator analysis; no file reads of scientific outcomes.

    Values are explicit `(direct_statistic, numerical_envelope)` or None for
    all2149 identities. Oracle scores can use their original log-LR scale:
    it has its own bank, so positive scale differences do not mix thresholds.
    """
    validate_terminal_evidence(terminal_evidence)  # before any value inspection
    validate_prospective_protocol(protocol)
    require(set(method_values) == set(METHODS), 'Both planned procedures required')
    validate_value_map(oracle_values, 'separately pinned iteration19 oracle')
    for name in METHODS:
        validate_value_map(method_values[name], name)
    require(all(method_values[METHODS[1]][i] is None or method_values[METHODS[0]][i] is not None for i in ALL),
            'Background method cannot validate without known-known dependency')
    values = dict(oracle=oracle_values, **method_values)
    outcomes = {i: dict(id=i) for i in QUERIES}
    results = {}
    for name, data in values.items():
        bank = [data[i] for i in REFERENCES]
        decisions = {}
        for i in QUERIES:
            decision = stats.classify(data[i], bank)
            outcomes[i][name] = decision
            decisions[i] = decision['reject']
        results[name] = dict(reference_requested=999, reference_resolved=sum(x is not None for x in bank),
                             reference_complete=all(x is not None for x in bank),
                             heldout_null_size=stats.tally([decisions[i] for i in CHECKS]),
                             alternative_frequency=stats.tally([decisions[i] for i in ALTERNATIVES]))
    comparisons = {}
    for first, second in [('oracle', METHODS[0]), METHODS]:
        comparisons[first + '_minus_' + second] = {
            label: relabel_paired([(outcomes[i][first]['reject'], outcomes[i][second]['reject']) for i in ids], first, second)
            for label, ids in [('heldout_null_size', CHECKS), ('alternative_frequency', ALTERNATIVES)]
        }
    sizes = {name: r['heldout_null_size']['conditional_frequency'] for name, r in results.items()}
    return dict(methods=results, paired=comparisons, outcomes=list(outcomes.values()),
                absolute_size_point_gaps={name: None if sizes[c['heldout_null_size']['first_method']] is None or sizes[c['heldout_null_size']['second_method']] is None
                                          else abs(sizes[c['heldout_null_size']['first_method']] - sizes[c['heldout_null_size']['second_method']])
                                          for name, c in comparisons.items()},
                scope='Draft conditional realized-bank frequencies under the fixed supplied law. No plug-in bootstrap power, additive nuisance attribution or global optimization certificate.',
                statistical_authority_sha256=AUTHORITY_SHA,
                numerical_envelopes_are_rigorous_error_bounds=False)


if __name__ == '__main__':
    raise SystemExit('Draft pure adapter only: no scientific collector or execution command is authorized here.')
