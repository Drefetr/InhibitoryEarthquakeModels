"""Export reviewed saved-summary accounting only; no ranking, RNG or fitting."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

CELLS = ['I00', 'I10', 'I01', 'I11']
ROLES = {'reference': 999, 'null_query': 1000, 'alternative_query': 150}
MODES = ('bank_only', 'bank_and_paired_query')
QUANTILES = ('0.025', '0.5', '0.975')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def endpoint_pair(pair, limits):
    require(isinstance(pair, list) and len(pair) == 2 and all(finite(v) for v in pair)
            and limits[0] <= pair[0] <= pair[1] <= limits[1], 'invalid endpoint pair')


def validate(data):
    require(data.get('status') == 'complete_saved_output_report' and type(data.get('replicates')) is int and data['replicates'] == 2000, 'complete2000 saved report required')
    require(finite(data.get('alpha')) and data['alpha'] == .05, 'alpha must equal .05')
    require(data.get('planned_roles') == ROLES and data.get('cells') == CELLS, 'exact planned roles/cells')
    controlled = data['controlled_statistic_counts']
    require(set(controlled) == set(ROLES), 'controlled roles')
    for role, n in ROLES.items():
        require(set(controlled[role]) == set(CELLS) and all(type(v) is int and 0 <= v <= n for v in controlled[role].values()), 'controlled counts')
    require(set(data['calibration']) == set(ROLES)-{'reference'} and set(data['bootstrap']) == set(ROLES)-{'reference'}, 'query roles')
    for role in ('null_query', 'alternative_query'):
        n = ROLES[role]; base = data['calibration'][role]
        require(base['planned_queries'] == n and set(base['decisions']) == set(CELLS) and set(base['rejection_bounds']) == set(CELLS), 'planned query identity')
        for cell in CELLS:
            counts = base['decisions'][cell]
            require(set(counts) == {'R', 'NR', 'U'} and all(type(v) is int and v >= 0 for v in counts.values()) and sum(counts.values()) == n, 'planned R/NR/U')
            require(base['rejection_bounds'][cell] == [counts['R']/n, (counts['R']+counts['U'])/n], 'planned denominator rates')
        endpoint_pair(base['interaction_bounds'], (-2, 2))
        point = base['interaction']; resolved = all(base['decisions'][c]['U'] == 0 for c in CELLS)
        require(point['status'] == ('resolved' if resolved else 'unresolved'), 'interaction resolution')
        require((finite(point['mean']) and finite(point['mcse']) and point['mcse'] >= 0) if resolved else point['mean'] is None and point['mcse'] is None, 'unavailable point must remain null')
        require(set(data['bootstrap'][role]) == set(MODES), 'bootstrap modes')
        for mode in MODES:
            boot = data['bootstrap'][role][mode]
            statuses = boot['point_mcse_status_counts']
            require(set(statuses) <= {'resolved', 'unresolved', 'insufficient'} and all(type(v) is int and v >= 0 for v in statuses.values()) and sum(statuses.values()) == 2000, 'bootstrap statuses')
            require(set(boot['rejection_endpoint_quantiles']) == set(CELLS), 'bootstrap cells')
            for obj, limits in [(boot['rejection_endpoint_quantiles'][c], (0, 1)) for c in CELLS] + [(boot['interaction_endpoint_quantiles'], (-2, 2))]:
                require(set(obj) == {'lower', 'upper'}, 'separate lower/upper quantiles')
                for endpoint in ('lower', 'upper'):
                    q = obj[endpoint]
                    require(set(q) == set(QUANTILES) and all(finite(q[k]) for k in QUANTILES), 'finite type7 quantiles')
                    require(limits[0] <= q['0.025'] <= q['0.5'] <= q['0.975'] <= limits[1], 'ordered quantiles')
                require(all(obj['lower'][q] <= obj['upper'][q] for q in QUANTILES), 'lower/upper quantile order')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.summary.read_bytes(); data = json.loads(raw); validate(data)
    matrix = []
    sensitivity = []
    for role in ('null_query', 'alternative_query'):
        base = data['calibration'][role]
        for cell in CELLS:
            matrix.append({'role': role, 'cell': cell, 'planned_references': 999, 'planned_queries': ROLES[role], 'controlled_references': data['controlled_statistic_counts']['reference'][cell], 'controlled_queries': data['controlled_statistic_counts'][role][cell], **base['decisions'][cell], 'rejection_lower': base['rejection_bounds'][cell][0], 'rejection_upper': base['rejection_bounds'][cell][1], 'interaction_lower': base['interaction_bounds'][0], 'interaction_upper': base['interaction_bounds'][1], 'interaction_status': base['interaction']['status'], 'interaction_mean': base['interaction']['mean'], 'interaction_mcse': base['interaction']['mcse']})
        for mode in MODES:
            boot = data['bootstrap'][role][mode]
            for label, endpoints in [(c, boot['rejection_endpoint_quantiles'][c]) for c in CELLS] + [('interaction', boot['interaction_endpoint_quantiles'])]:
                for endpoint in ('lower', 'upper'):
                    sensitivity.append({'role': role, 'mode': mode, 'quantity': label, 'endpoint': endpoint, 'replicates': 2000, 'q025': endpoints[endpoint]['0.025'], 'q500': endpoints[endpoint]['0.5'], 'q975': endpoints[endpoint]['0.975'], **{'point_mcse_'+s: boot['point_mcse_status_counts'].get(s, 0) for s in ('resolved', 'unresolved', 'insufficient')}})
    args.output.mkdir(parents=True, exist_ok=False)
    outputs = []
    for name, rows in [('F_MATRIX_ACCOUNTING.csv', matrix), ('F_BOOTSTRAP_ENDPOINT_SENSITIVITY.csv', sensitivity)]:
        target = args.output/name
        with target.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        outputs.append({'name': name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    result = {'input': {'name': args.summary.name, 'sha256': hashlib.sha256(raw).hexdigest()}, 'csv_outputs': outputs, 'alpha': data['alpha'], 'planned_roles': ROLES, 'cells': CELLS, 'replicates': 2000, 'matrix_accounting': matrix, 'bootstrap_endpoint_sensitivity': sensitivity, 'controlled_statistic_counts': data['controlled_statistic_counts'], 'interactions': {r: data['calibration'][r]['interaction'] for r in ('null_query', 'alternative_query')}, 'interpretation': 'Planned-denominator accounting bounds and descriptive deterministiclinear/type7 endpoint quantiles; no confidence-coverage claim. Unknowns retained. JSON null and blank CSV point mean/MCSE denote unavailable, never zero. No new ranking or resampling.'}
    with (args.output/'F_RESULT_TABLES.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__':
    main()
