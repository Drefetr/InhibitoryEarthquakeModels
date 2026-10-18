# Contributing

Bug reports, reproducibility findings, mathematical objections and focused pull requests are welcome. For a scientific concern, identify the equation, proposition or figure; state the assumptions; provide a derivation or smallest reproducible input; and distinguish a counterexample from a numerical approximation.

For code changes, explain the previous and resulting behaviour, include relevant tests and record dependencies. Run `python -m unittest discover -s tests -v` and the deterministic mathematical checks. Disclose AI assistance in the pull-request description, including what you checked yourself.

Treat files listed as preserved numerical authorities in `provenance/extraction_manifest.json` as frozen baselines. Propose a new version rather than silently rewriting source pins or saved reference results. Changes to numerical procedures, parameters or scientific conclusions need explicit scientific review and newly labelled outputs. Compilation and test success do not certify a theorem or fit.

Do not commit external catalogue records, credentials, virtual environments, cached TeX/font files, large history banks or agent transcripts. Keep example data invented and clearly labelled. Avoid expensive simulation/fitting work in ordinary tests or CI.

Contributions use the licence applying to the affected material: MIT for software and CC BY 4.0 for project prose/figures/results. Submit only material you are entitled to contribute; identify third-party exceptions. This repository requires no contributor licence agreement. Authorship changes require explicit discussion; code contributions do not silently change paper authorship.

Keep discussion respectful and evidence-based. Report the environment, command, error and expected result. Scientific disagreements and negative findings are welcome. No particular response time is guaranteed.
