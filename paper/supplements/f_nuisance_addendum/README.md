# F continuation: background nuisance estimation

This manuscript companion is an editorial revision dated 22 October 2026. Scientific execution was 19 October UTC (20 October in New Zealand at completion). `main.tex` includes `results.tex`; compile using the parent manuscript's reviewed LaTeX workflow. The revision has been compiled with Tectonic 0.17.0 and all seven rendered pages have been independently inspected.

The packet contains saved summary/table/figure replay only. `data/summary.json` preserves audited aggregate values in compact form; the three tables and three PNGs are exact copies of the saved checkpoint. `data/geometry_summary.json` is the safe public summary from the sealed source/results archive, with the private execution field omitted. The three `tools/` scripts are byte-identical copies of the reviewed saved-artifact implementations. No fitting or bootstrap recalibration source bundle is included.

Run from this addendum directory, using new output directories and the compatible Python, Matplotlib and NumPy environment recorded with the parent delivery:

```sh
python -B tools/f_saved_tables04.py --summary data/summary.json --output replayed_tables
python -B tools/f_saved_plot04.py --summary data/summary.json --output replayed_matrix
```

To plot the saved geometry summary, call `plots` directly; do not call the history-collecting command-line entrypoint:

```python
import json
import sys
from pathlib import Path
sys.path.insert(0, 'tools')
from f_geometry_report import plots
output = Path('replayed_geometry')
output.mkdir(exist_ok=False)
plots(json.loads(Path('data/geometry_summary.json').read_text()), output)
```

These commands were executed serially on 22 October 2026. Both regenerated CSVs and all three PNGs matched the supplied files byte for byte; all PNG pixels also matched. The regenerated `F_RESULT_TABLES.json` reproduces every result field and CSV hash, but records the SHA-256 of this compact `summary.json` rather than the original, differently formatted summary. Its input-provenance field therefore differs; the JSON files are not byte-identical. Python 3.13.4, Matplotlib 3.11.2, NumPy 2.2.6 and Pillow 12.0.0 were used. See `../../../validation/F_editorial_checks.json`.

Compare regenerated files with `data/` and the corresponding `figures/` subdirectories. Image pixels may depend on plotting/font versions; investigate differences rather than assuming equivalence.

The accounting preserves all planned denominators and unknowns. Endpoint quantiles are descriptive sensitivity, not confidence intervals. Geometry is partial (6 of 11 cases completed); full-pair interaction point summaries and MCSE remain unavailable. This replay displays already saved values, without new fitting, ranking, bootstrap plans or simulations.

The continuation's finite recipe is distinct from the paper's historical numerical experiment. A separate metadata join verified the historical and continuation catalogue/clock identities and case roles. This compact packet omits the underlying bank manifests and does not independently reproduce that join. Identical input identities do not imply identical fits or decisions. Its original25-query continuation pilot is distinct from the paper's eta-planning experiment. Supplied/supplied nuisance treatment still estimates eta under the alternative; it is not a fixed-eta oracle.

Raw catalogues/history banks, bank manifests, bootstrap chunks, private execution evidence and provider transcripts are omitted. This packet cannot recalibrate the reference bank or regenerate bootstrap endpoints from their seeds. Re-fitting or full calibration replay requires separately pinned omitted inputs and the full procedure environment; saved aggregate availability does not establish a new experiment or reproduce historical RNG.
