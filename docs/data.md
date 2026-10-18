# External data and snapshot identity

No third-party event records are included. Tests, figure generation and paper compilation never download data.

Start with [GeoNet catalogue documentation](https://www.geonet.org.nz/data/types/eq_catalogue), [WFS](https://wfs.geonet.org.nz/), [FDSN](https://service.geonet.org.nz/fdsnws/event/1/) and the [catalogue DOI](https://doi.org/10.21420/0S8P-TZ38). Read the [provider terms and attribution policy](https://www.geonet.org.nz/policy) before use. Dataset-specific terms remain authoritative.

The historical reconstructed WFS snapshot was retrieved 2026-10-15. Selection: 1960-01-01 inclusive to 2026-10-15 exclusive; earthquake event type; magnitude >=5; longitude 165..180; latitude -48.5..-33.5; finite depth <=40 km excluding -9 km unknown-depth sentinel; other negative depths retained. Native magnitudes/scales were not homogenised. Time is days since 1960-01-01 UTC with 86,400 seconds/day and no leap-second modelling. Magnitude excess is M-5. The observation window is 24,394 days; 913 events were selected.

Exact processed CSV: 109747 bytes; SHA256 `feacccb7581d1c4c33f88874cd6b198c41639da6930a7fc35e46f910a594fb3e`. Original selection metadata and separate FDSN identity are in `provenance/catalogue_snapshot_metadata.json` and `fdsn_snapshot_metadata.json`. These are metadata, not event records. The interfaces are not independent physical catalogues.

Preserved `engine/iteration_5/code/fetch_catalogue.py` and `prepare_catalogue.py` show acquisition/filtering mechanics. Their optional guarded CLI workflows write local ignored data paths. A provider can revise past events: a fresh download with identical dates need not reproduce historical bytes, counts or fits. Historical snapshot recovery is not guaranteed from the live provider.

If you independently obtain the exact historical processed CSV:

```sh
python scripts/install_catalogue.py /path/to/geonet_M5_shallow_1960_2026.csv --check-only
python scripts/install_catalogue.py /path/to/geonet_M5_shallow_1960_2026.csv
python scripts/run_observed.py --check-inputs-only
```

The installer refuses mismatched bytes and existing destinations. It places the file in an ignored engine input directory. Analyse a new snapshot under a separately versioned protocol rather than weakening the historical guard.
