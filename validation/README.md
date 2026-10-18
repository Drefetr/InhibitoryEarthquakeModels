# Public preparation validation

Actual checks for this public edition are recorded here. Historical v4 verification under provenance/ applies to the older sealed archives, not this repository.

- Three deterministic unit tests passed, including imports of all fourteen exposed numerical modules, kernel normalisation and the exact constant-Poisson exposure identity on a literal history.
- The symbolic compensator cancellation, fixed no-event path, illustrative likelihood-factor derivative cases and paired MCSE arithmetic passed. These are not a proof audit of every stochastic theorem or reproduction of historical fits.
- Nine figures were regenerated from included project-derived inputs and visually inspected as a contact sheet. The catalogue figure uses aggregate bins, without per-event magnitude values.
- Tectonic 0.17.0 compiled the 64-page paper and two-page supplement. Production checks found zero undefined references/citations, missing glyphs, overfull/underfull boxes or text crossing checked margins. A Fontconfig environment warning is retained in local raw logs; no missing glyphs were detected.
- All paper pages were rendered. All six contact sheets and changed pages 1, 57, 58 and 60 were inspected; the other sixty page images are byte-identical to the previously reviewed v4 renders. Both supplement pages were rendered and inspected. This is a Codex preparation review, not independent external peer review.
- Scientific chapter sources, shipped figure bytes, selected numerical results/settings and frozen numerical Python authorities match their source pins. Sealed v4 archive hashes remain unchanged.
- Citation metadata passed the official CFF JSON schema. Local Markdown links were checked. Third-party per-event records and the earlier manuscript PDF are excluded.
- The observed-workload preflight intentionally fails for the absent external catalogue, after passing all frozen source checks. It does not launch a fit. Full observed fitting, optional GPU operations and historical campaign reruns were not executed.

See the JSON records and command logs for exact scope. Raw build logs, rendered PNGs and generated plots remain locally under ignored `build/`, keeping the public repository compact. Final archive extraction and manifest verification are detached beside the prepared ZIP to avoid circular hashing. These preparation records precede the initial push. Check the repository Actions tab for actual remote workflow status.

A broader publication privacy audit subsequently checked staged blobs, PDFs and Git metadata. A personal interpreter path was removed from an optional launcher; a second launcher was made portable, and public protocol bindings were resealed with the original protocol retained. See privacy_audit.json. No scientific Python or paper/figure content changed.
