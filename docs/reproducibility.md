# Reproduction scope

Run commands from the repository root. Python 3.13 and pinned requirements describe the tested public CPU preparation route, not the original scientific host.

| Task | Inputs | Command/status |
|---|---|---|
| Deterministic source/likelihood checks | Included fixed examples | `python -m unittest discover -s tests -v` |
| Compensator/product/paired arithmetic | Included equations/fixed parameters | `python scripts/check_mathematics.py --output build/mathematics.json` |
| Public payload/source pins | Included manifest | `python scripts/verify_repository.py` |
| Regenerate nine figures | Included aggregate/saved inputs | `python scripts/make_figures.py --output build/figures` |
| Recreate F tables and three addendum figures | Included F aggregate summaries and reviewed display tools | Follow the [F companion guide](../paper/supplements/f_nuisance_addendum/README.md); no fitting or resampling |
| Compile manuscript and both supplements | Shipped figures, Tectonic 0.17.0 | `python scripts/build_paper.py --output build/paper` |
| Observed five-family fitting | Exact external historical CSV | Install per data guide; `python scripts/run_observed.py --output build/observed.json` |
| Every historical synthetic campaign | Unshipped banks, run plans and execution states | Not a supported single-command replay in this compact release |

Observed fitting is an explicit research workload with the frozen 1,800-second cooperative wall budget, one native numerical thread and mandatory solver/control checks. It can return incomplete on different hardware. It was **not executed** during public preparation. The wrapper calls the original hash-bound recipe with its original nuisance/search/failure rules. Its output contains per-family training/test controls. Historical bounded execution receipts and publication postprocessing are not recreated by this wrapper; a new run is not a byte-exact archival replay.

Included numerical APIs permit independent use of kernels, precise-clock simulation, likelihood, nuisance fitting, amplitude profiling and heterogeneity decomposition. Historical synthetic/reference banks can be large. Saved summaries and generating-law settings do not encode every historical path, reference rank, RNG state or recovery lineage. A new simulation is not regeneration of the retained bank. See the code guide and `paper/supplements/methods_settings.json`.

For PDF production checks and complete rendering:

```sh
python scripts/validate_pdf.py --source-root paper --pdf build/paper/manuscript.pdf --log build/paper/main/main.log --output build/pdf-qa
```

Output directories must be new. Automated production checks do not replace visual inspection. All supplied public-edition pages were rendered and visually reviewed; see validation/. TeX resources can change pagination. Regenerated figures are separate from shipped inputs; compare numeric summaries and inspect appearance rather than assuming cross-version byte identity.

No fitting, simulation, bootstrap, catalogue download or historical research-history scoring ran during the original public preparation or the 22 October manuscript/addendum integration. The F research continuation ran separately on 19 October UTC; its saved findings are now reported in a dated addendum. Tests use hand-constructed examples. Historical v4 records and new preparation checks are distinguished.

The public observed protocol retains the historical scientific design but binds portable interpreter-discovery variants of two optional launchers. `provenance/historical_observed_protocol.json` retains the original byte-sealed protocol. No numerical Python authority or scientific setting was changed for publication privacy.

For manuscript navigation, public summary targets and explicit archival omissions, see the [public source and availability guide](public_readability.md). E04 remains a separate linked explanatory supplement. The [F nuisance addendum](../paper/supplements/f_nuisance_addendum.pdf) and its [source/replay guide](../paper/supplements/f_nuisance_addendum/README.md) are separate new material; saved table/figure reproduction does not refit histories or rerun the 2,000 resamples. The historical synthetic/reference banks and full historical M16 authority are not supplied by the aggregate summaries.
