"""Plot verified saved F04 accounting bounds; no scientific reevaluation.

Source-only until separately reviewed and leased for saved-output postprocessing.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

CELLS = ['I00', 'I10', 'I01', 'I11']
ROLES = [('null_query', 1000, 'Null queries'), ('alternative_query', 150, 'Alternative queries')]
TITLE = 'Four-procedure F calibration: planned-denominator bounds'
CAPTION = ('Segments show planned-denominator rejection accounting bounds [R/N, (R+U)/N], '
           'not confidence intervals. The lower endpoint is marked. R=rejection, NR=nonrejection, '
           'U=unknown; unknown decisions remain in the upper bound. Controlled counts describe '
           'available saved statistics, not resolved calibration decisions.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(summary):
    alpha = summary.get('alpha')
    require(type(alpha) in (int, float) and math.isfinite(alpha) and 0 < alpha < 1, 'finite alpha strictly between zero and one required')
    require(summary.get('status') == 'complete_saved_output_report', 'complete saved report required')
    require(summary.get('replicates') == 2000, 'completed report identity')
    require(summary.get('cells') == CELLS, 'four-cell order')
    require(summary.get('planned_roles') == {'reference': 999, 'null_query': 1000, 'alternative_query': 150}, 'planned role denominators')
    require(set(summary['calibration']) == {r for r, _, _ in ROLES}, 'query roles')
    controlled = summary['controlled_statistic_counts']
    require(set(controlled) == {'reference', 'null_query', 'alternative_query'}, 'controlled roles')
    for role, n in summary['planned_roles'].items():
        require(set(controlled[role]) == set(CELLS), 'controlled cells')
        require(all(type(controlled[role][c]) is int and 0 <= controlled[role][c] <= n for c in CELLS), 'controlled count')
    rows = {}
    for role, n, _ in ROLES:
        data = summary['calibration'][role]
        require(data['planned_queries'] == n and set(data['decisions']) == set(CELLS) and set(data['rejection_bounds']) == set(CELLS), 'query identity')
        rows[role] = []
        for cell in CELLS:
            counts = data['decisions'][cell]
            require(set(counts) == {'R', 'NR', 'U'} and all(type(x) is int and x >= 0 for x in counts.values()) and sum(counts.values()) == n, 'R/NR/U must sum to planned N')
            expected = [counts['R']/n, (counts['R']+counts['U'])/n]
            actual = data['rejection_bounds'][cell]
            require(isinstance(actual, list) and len(actual) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in actual) and actual == expected, 'saved bounds must equal planned-denominator rates')
            rows[role].append({'cell': cell, 'N': n, **counts, 'bounds': expected, 'controlled_query_statistics': controlled[role][cell], 'controlled_reference_statistics': controlled['reference'][cell]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.summary.read_bytes()
    summary = json.loads(raw)
    rows = verify(summary)
    alpha = summary['alpha']
    title = TITLE+' (alpha='+format(alpha, '.15g')+')'
    caption = 'Alpha='+format(alpha, '.15g')+'. '+CAPTION
    require(not args.output.exists(), 'fresh output directory required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11, 'figure.dpi': 150, 'savefig.dpi': 150})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    for axis, (role, n, label) in zip(axes, ROLES):
        for y, row in enumerate(rows[role]):
            lower, upper = row['bounds']
            axis.hlines(y, lower, upper, color='#1f5a85', linewidth=5, alpha=.8)
            axis.plot(lower, y, marker='o', color='#102f46', markersize=6)
            text = ('R={R}  NR={NR}  U={U}\nControlled query={controlled_query_statistics}/{N}; '
                    'reference={controlled_reference_statistics}/999').format(**row)
            axis.text(1.035, y, text, transform=axis.get_yaxis_transform(), va='center', fontsize=8, clip_on=False)
        axis.set_title(label+' (planned N='+str(n)+')', loc='left')
        axis.set_xlim(0, 1)
        axis.set_ylim(3.5, -.5)
        axis.set_yticks(range(4), CELLS)
        axis.tick_params(labelleft=True)
        axis.set_xticks([0, .25, .5, .75, 1])
        axis.set_xlabel('Rejection proportion: accounting lower to upper bound')
        axis.grid(axis='x', alpha=.22)
        for side in ('top', 'right'): axis.spines[side].set_visible(False)
    fig.suptitle(title, x=.04, ha='left', fontsize=14)
    fig.text(.04, .035, 'Segments: planned-denominator bounds, not confidence intervals. Lower endpoint marked.\n'
             'R=rejection; NR=nonrejection; U=unknown. Controlled statistics and decision resolution are distinct.', fontsize=9, va='bottom')
    fig.subplots_adjust(left=.06, right=.78, bottom=.22, top=.85, wspace=1.25)
    args.output.mkdir(parents=True)
    image = args.output/'F04_planned_denominator_bounds.png'
    fig.savefig(image, metadata={'Software': 'F04 saved-output plotter', 'Title': title, 'Description': caption})
    plt.close(fig)
    metadata = {'input': {'name': args.summary.name, 'sha256': hashlib.sha256(raw).hexdigest()},
                'output': {'name': image.name, 'sha256': hashlib.sha256(image.read_bytes()).hexdigest()},
                'title': title, 'caption': caption, 'alpha': alpha, 'cell_order': CELLS, 'verified_rates_and_counts': rows,
                'scope': 'Saved JSON display only. No bootstrap plot, imputation, likelihood or history evaluation.'}
    with (args.output/'plot_metadata.json').open('x', encoding='utf-8') as target:
        json.dump(metadata, target, indent=2, allow_nan=False)
        target.write('\n')


if __name__ == '__main__':
    main()
