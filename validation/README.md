# Reproducibility checks

This directory retains compact results useful for checking the shipped manuscript and code. `minor_fixes_checks.json` records the current PDF bindings and the checks executed for the 22 October 2026 wording and bookmark corrections. `F_editorial_checks.json` is preserved as evidence for the earlier 22 October snapshot, before those minor corrections; its PDF hashes identify that earlier snapshot. Both October check records were produced from uncommitted snapshots based on `5ab396fbc9a8df432cf76ec42512e4344e6abdf6`; their snapshot status describes the review-time state, not the current Git state. They do not establish two distinct verified Git commits. The records listed below are retained from that baseline commit; they are historical evidence, not validation of the revised PDFs:

- `unit_tests.txt`: three fixed-input tests (imports, kernel normalization and exact Poisson exposure).
- `mathematics.json`: selected deterministic algebra, quadrature and arithmetic examples; not certification of every theorem or historical experiment.
- `production_checks.json`, `visual_review.json` and `toolchain.json`: baseline 65-page manuscript and 2-page supplement production results and tested build environment. The historical TeX cache was not snapshotted.
- `privacy_audit.json`, `data_exclusion_scan.json` and `external_input_preflight.txt`: public payload/input-exclusion scope. External scientific inputs remain separately required.
- `documentation_checks.json`: earlier documentation check record; current membership is authoritative in `MANIFEST.json`.

Earlier validation records are retained with their original source/PDF bindings. Detailed review correspondence, adjudications, agent task records, raw execution logs and renders are maintained locally outside this repository. These checks do not reproduce missing historical history banks or establish a physical mechanism. Run the commands in the main README to check a fresh checkout; no refits or simulations are needed for those checks.
