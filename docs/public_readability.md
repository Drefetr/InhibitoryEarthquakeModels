# Public source and availability guide

The manuscript's source-supplement links are internal PDF links to Appendix A. Stable S/M identifiers and historical hashes identify archival authorities, not promises of public payload membership. The authoritative public register is [source_availability.json](../paper/supplements/source_availability.json); the unchanged [historical index](../provenance/historical_v4_source_index.json) describes sealed-v4 provenance.

The S register contains 3 included files, 1 partial representation and 23 files not distributed. In particular, S07 is not distributed. S18 is an operational adaptation and S24 is a metadata extract; neither is an assertion that every original historical byte is shipped.

## Useful public entry points

| Need | Public target | Scope |
|---|---|---|
| Mathematical arguments | Chapters 2–5 of the manuscript; [E04](../paper/supplements/explanatory_supplement.pdf) | Main proofs plus separate explanatory steps and a fixed no-event example; no general empirical self-containment claim |
| Restricted-history likelihood | [S10 implementation](../engine/iteration_20/code/restricted_history.py) | Included implementation |
| Observed model implementation | [S17 implementation](../engine/iteration_27/code/adequacy.py) | Included code, not every historical execution receipt |
| Observed search protocol | [S18 protocol](../engine/iteration_27/configs/observed_frozen.json) | Declared public operational adaptation; historical protocol retained separately |
| Catalogue selection | [S24 metadata extract](../provenance/catalogue_snapshot_metadata.json), [data guide](data.md) | Metadata and access/terms; third-party event records excluded |
| Synthetic detection and pairing | [detection summary](../paper/data/detection.json), [paired summary](../paper/data/paired.json), [settings](../paper/supplements/methods_settings.json) | Aggregate saved results and settings, not the complete historical/reference banks |
| Observed scores and selected fits | [prediction summary](../paper/data/prediction.json), [selected parameters](../paper/supplements/predictive_parameters.json) | Saved summaries/selected fields, not full M16 or complete solver lineage |
| Temporal score decomposition | [temporal summary](../paper/data/temporal.json) | Saved fixed-parameter aggregate decomposition, not a new history evaluation |

## S-register public status

This table is generated from the current public register. Original archive paths and pins remain in that register and the historical index.

| ID | Public status | Public target |
|---|---|---|
| S01 | not-distributed | Not distributed |
| S02 | not-distributed | Not distributed |
| S03 | not-distributed | Not distributed |
| S04 | not-distributed | Not distributed |
| S05 | not-distributed | Not distributed |
| S06 | not-distributed | Not distributed |
| S07 | not-distributed | Not distributed |
| S08 | not-distributed | Not distributed |
| S09 | not-distributed | Not distributed |
| S10 | included | [engine/iteration_20/code/restricted_history.py](../engine/iteration_20/code/restricted_history.py) |
| S11 | not-distributed | Not distributed |
| S12 | not-distributed | Not distributed |
| S13 | not-distributed | Not distributed |
| S14 | not-distributed | Not distributed |
| S15 | not-distributed | Not distributed |
| S16 | not-distributed | Not distributed |
| S17 | included | [engine/iteration_27/code/adequacy.py](../engine/iteration_27/code/adequacy.py) |
| S18 | included (operational adaptation) | [engine/iteration_27/configs/observed_frozen.json](../engine/iteration_27/configs/observed_frozen.json) |
| S19 | not-distributed | Not distributed |
| S20 | not-distributed | Not distributed |
| S21 | not-distributed | Not distributed |
| S22 | not-distributed | Not distributed |
| S23 | not-distributed | Not distributed |
| S24 | partially-represented (metadata extract) | [provenance/catalogue_snapshot_metadata.json](../provenance/catalogue_snapshot_metadata.json) |
| S25 | not-distributed | Not distributed |
| S26 | not-distributed | Not distributed |
| S27 | not-distributed | Not distributed |

## Separate F continuation

The [dated F addendum](../paper/supplements/f_nuisance_addendum.pdf), [editable source and reproduction guide](../paper/supplements/f_nuisance_addendum/README.md), and [scope note](F_followup.md) report the completed bounded continuation with explicit unknown outcomes. Its supplied/supplied alternative estimates eta; it is not the historical fixed-eta oracle. The new compact derived summaries, tables and three figures are separate from the unchanged historical detection/paired inputs and S/M availability registers. Raw history banks, provider transcripts and operational review histories are not included.
