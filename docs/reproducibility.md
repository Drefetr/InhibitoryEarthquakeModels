# Reproduction scope

Run commands from the repository root. Python 3.13 and pinned requirements describe the tested public CPU preparation route, not the original scientific host.

| Task | Inputs | Command/status |
|---|---|---|
| Deterministic source/likelihood checks | Included fixed examples | `python -m unittest discover -s tests -v` |
| Compensator/product/paired arithmetic | Included equations/fixed parameters | `python scripts/check_mathematics.py --output build/mathematics.json` |
| Public payload/source pins | Included manifest | `python scripts/verify_repository.py` |
| Regenerate nine figures | Included aggregate/saved inputs | `python scripts/make_figures.py --output build/figures` |
| Compile manuscript and supplement | Shipped figures, Tectonic 0.17.0 | `python scripts/build_paper.py --output build/paper` |
| Observed five-family fitting | Exact external historical CSV | Install per data guide; `python scripts/run_observed.py --output build/observed.json` |
| Every historical synthetic campaign | Unshipped banks, run plans and execution states | Not a supported single-command replay in this compact release |

Observed fitting is an explicit research workload with the frozen 1,800-second cooperative wall budget, one native numerical thread and mandatory solver/control checks. It can return incomplete on different hardware. It was **not executed** during public preparation. The wrapper calls the original hash-bound recipe with its original nuisance/search/failure rules. Its output contains per-family training/test controls. Historical bounded execution receipts and publication postprocessing are not recreated by this wrapper; a new run is not a byte-exact archival replay.

Included numerical APIs permit independent use of kernels, precise-clock simulation, likelihood, nuisance fitting, amplitude profiling and heterogeneity decomposition. Historical synthetic/reference banks can be large. Saved summaries and generating-law settings do not encode every historical path, reference rank, RNG state or recovery lineage. A new simulation is not regeneration of the retained bank. See the code guide and `paper/supplements/methods_settings.json`.

For PDF production checks and complete rendering:

```sh
python scripts/validate_pdf.py --source-root paper --pdf build/paper/manuscript.pdf --log build/paper/main/main.log --output build/pdf-qa
```

Output directories must be new. Automated production checks do not replace visual inspection. All supplied public-edition pages were rendered and visually reviewed; see validation/. TeX resources can change pagination. Regenerated figures are separate from shipped inputs; compare numeric summaries and inspect appearance rather than assuming cross-version byte identity.

No fitting, simulation, bootstrap, catalogue download or historical research-history scoring ran during public preparation. Tests use hand-constructed examples. Historical v4 records and new preparation checks are distinguished.

The public observed protocol retains the historical scientific design but binds portable interpreter-discovery variants of two optional launchers. `provenance/historical_observed_protocol.json` retains the original byte-sealed protocol. No numerical Python authority or scientific setting was changed for publication privacy.
