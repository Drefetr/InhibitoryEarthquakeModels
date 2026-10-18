"""Freeze a fresh public GeoNet snapshot, with byte-level HTTP provenance.

Run from the project root: python iteration_5/code/fetch_catalogue.py
This is independent acquisition, not the manuscript's missing frozen catalogue.
Only stdlib dependencies. Existing raw bytes and manifests are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
START = "1960-01-01T00:00:00.000Z"
END = "2026-10-15T00:00:00.000Z"
WFS = "https://wfs.geonet.org.nz/geonet/ows"
FDSN = "https://service.geonet.org.nz/fdsnws/event/1/query"
DOCUMENTS = {
    "wfs_documentation": "https://wfs.geonet.org.nz/",
    "fdsn_documentation": "https://www.geonet.org.nz/data/access/FDSN",
    "catalogue_documentation": "https://www.geonet.org.nz/data/types/eq_catalogue",
    "data_policy": "https://www.geonet.org.nz/policy",
    "license": "https://creativecommons.org/licenses/by/3.0/nz/",
    "catalogue_doi": "https://doi.org/10.21420/0S8P-TZ38",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def wfs_parameters(start, end, *, hits=False):
    params = dict(service="WFS", version="1.0.0", request="GetFeature",
                typeName="geonet:quake_search_v1", outputFormat="csv", maxFeatures="100000",
                cql_filter=(f"origintime>='{start}' AND origintime<'{end}' "
                            "AND magnitude>=5 AND BBOX(origin_geom,165,-48.5,180,-33.5)"))
    if hits:
        params["resultType"] = "hits"
    return params


def request_bytes(url, destination, log_path, *, attempts=4):
    """Record each attempt, then exclusively create immutable successful bytes."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace frozen response: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        started = utc_now()
        tick = time.monotonic()
        record = dict(request_url=url, method="GET", attempt=attempt,
                      requested_at_utc=started,
                      request_headers={"User-Agent": "Independent-research-catalogue-audit/1.0", "Accept-Encoding": "identity"})
        try:
            req = urllib.request.Request(url, headers=record["request_headers"])
            with urllib.request.urlopen(req, timeout=75) as response:
                data = response.read()
                record.update(status=response.status, reason=response.reason,
                              response_url=response.url, response_headers=dict(response.headers),
                              response_bytes=len(data), sha256=sha256(data))
            record.update(completed_at_utc=utc_now(), elapsed_seconds=time.monotonic()-tick,
                          destination=str(destination.relative_to(ROOT)))
            with destination.open("xb") as out:
                out.write(data)
            with Path(log_path).open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False)+"\n")
            return data, record
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            record.update(error_type=type(exc).__name__, error=str(exc),
                          completed_at_utc=utc_now(), elapsed_seconds=time.monotonic()-tick)
            if isinstance(exc, urllib.error.HTTPError):
                record.update(status=exc.code, response_headers=dict(exc.headers))
                body = exc.read()
                error_path = destination.with_name(destination.name+f".error_attempt_{attempt}")
                with error_path.open("xb") as out:
                    out.write(body)
                record.update(error_response_path=str(error_path.relative_to(ROOT)),
                              error_response_sha256=sha256(body), error_response_bytes=len(body))
            with Path(log_path).open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False)+"\n")
            print(f"Attempt {attempt}/{attempts} failed: {url}: {exc}", flush=True)
            if attempt == attempts or (isinstance(exc, urllib.error.HTTPError) and exc.code < 500 and exc.code != 429):
                raise
            time.sleep(min(2 ** (attempt-1), 8))
    raise AssertionError("unreachable")


def freeze_snapshot(*, include_prehistory=True, snapshot_name=None):
    snapshot_name = snapshot_name or "geonet_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = ROOT / "data" / "raw" / snapshot_name
    target.mkdir(parents=True, exist_ok=False)
    log = target / "http_requests.jsonl"
    manifest = dict(schema_version=1, provenance="Fresh public GeoNet snapshot; not historical manuscript replication",
                    snapshot_name=snapshot_name, started_at_utc=utc_now(),
                    requested_start_inclusive=START, requested_end_exclusive=END,
                    bbox=dict(west=165, east=180, south=-48.5, north=-33.5),
                    minimum_magnitude=5, server_depth_filter=None, server_event_type_filter=None,
                    retrieval_cap=100000, chunks=[], documentation=[], crosscheck=None)
    # This WFS facade may reject resultType; preserve evidence rather than assume
    # a full GeoServer implementation or misreport an advertised count.
    try:
        _, hitrecord = request_bytes(WFS+"?"+urllib.parse.urlencode(wfs_parameters(START, END, hits=True)), target/"wfs_hits_probe.xml", log)
        manifest["hits_probe"] = dict(supported=True, response=hitrecord)
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise
        manifest["hits_probe"] = dict(supported=False, status=400,
            note="GeoNet WFS facade rejects resultType=hits as an additional query parameter. Counts validated against full FDSN interface and below requested cap.")
    years = list(range(1960, 2030, 10))
    periods = [(str(y), f"{y}-01-01T00:00:00.000Z", min(f"{y+10}-01-01T00:00:00.000Z", END), "main") for y in years if f"{y}-01-01T00:00:00.000Z" < END]
    if include_prehistory:
        periods.append(("1950", "1950-01-01T00:00:00.000Z", START, "prehistory"))
    for label, start, end, role in periods:
        params = wfs_parameters(start, end)
        data, record = request_bytes(WFS+"?"+urllib.parse.urlencode(params), target/f"wfs_{label}.csv", log)
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        # Reject HTML/error payloads and schemas that lose the event classification.
        header = next(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
        if not {"publicid", "eventtype", "origintime", "depth", "magnitude", "magnitudetype"}.issubset(header):
            raise ValueError(f"Unexpected WFS CSV header: {header!r}")
        if len(rows) >= int(params["maxFeatures"]):
            raise ValueError(f"Possible truncation in {label}: CSV {len(rows)} reaches cap")
        manifest["chunks"].append(dict(label=label, role=role, start_inclusive=start, end_exclusive=end,
                                       rows=len(rows), hits=None, response=record))
        print(f"Frozen WFS {label}: rows={len(rows)}", flush=True)
    # A second GeoNet interface checks IDs; it is not an independent catalogue.
    params = dict(format="text", starttime=START, endtime=END, minmagnitude="5",
                  minlongitude="165", maxlongitude="180", minlatitude="-48.5", maxlatitude="-33.5", orderby="time-asc")
    data, record = request_bytes(FDSN+"?"+urllib.parse.urlencode(params), target/"fdsn_all_events.txt", log)
    manifest["crosscheck"] = dict(response=record, inclusive_end_note="FDSN end is inclusive; preparation reapplies exclusive cutoff")
    # FDSN uses a strict magnitude lower bound. Widen it to retain M=5 events.
    params["minmagnitude"] = "4.9"
    _, record = request_bytes(FDSN+"?"+urllib.parse.urlencode(params), target/"fdsn_expanded_events.txt", log)
    manifest["crosscheck"]["expanded_response"] = record
    # A previously observed interface difference, retained as a full-precision
    # WFS diagnostic without substituting FDSN's magnitude into the main data.
    params = dict(service="WFS", version="1.0.0", request="GetFeature",
                  typeName="geonet:quake_search_v1", outputFormat="csv",
                  cql_filter="publicid='2025p672513'")
    _, record = request_bytes(WFS+"?"+urllib.parse.urlencode(params), target/"wfs_interface_diagnostic.csv", log)
    manifest["crosscheck"]["diagnostic_response"] = record
    for name, url in DOCUMENTS.items():
        extension = ".html"
        try:
            data, record = request_bytes(url, target/f"geonet_{name}{extension}", log)
            manifest["documentation"].append(dict(name=name, response=record))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # Metadata URL failure is disclosed; the catalogue itself remains usable.
            manifest["documentation"].append(dict(name=name, url=url, error=str(exc), downloaded=False))
    manifest.update(completed_at_utc=utc_now(), downloader_sha256=sha256(Path(__file__).read_bytes()),
                    request_log_sha256=sha256(log.read_bytes()))
    path = target/"raw_manifest.json"
    with path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(manifest, out, indent=2, ensure_ascii=False)
        out.write("\n")
    print(str(path), flush=True)
    return path


def freeze_fdsn_supplement():
    """Full-precision preferred event information for a separate interface fit."""
    target = ROOT/"data"/"raw"/("geonet_fdsn_sensitivity_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    target.mkdir(parents=True, exist_ok=False)
    params = dict(format="xml", starttime=START, endtime=END, minmagnitude="4.9",
                  minlongitude="165", maxlongitude="180", minlatitude="-48.5", maxlatitude="-33.5", orderby="time-asc")
    manifest = dict(started_at_utc=utc_now(), purpose="Separate FDSN interface sensitivity; primary WFS data unchanged",
                    query_parameters=params, exact_selection="Local preferred magnitude >=5; earthquake; known depth <=40; bbox and half-open interval reapplied")
    _, record = request_bytes(FDSN+"?"+urllib.parse.urlencode(params), target/"fdsn_expanded_events.xml", target/"http_requests.jsonl")
    manifest["response"] = record
    code = Path(__file__).read_bytes()
    (target/"fetch_catalogue_as_used.py").write_bytes(code)
    manifest.update(completed_at_utc=utc_now(), downloader_sha256=sha256(code))
    path = target/"raw_manifest.json"
    with path.open("x", encoding="utf-8") as out:
        json.dump(manifest, out, indent=2)
        out.write("\n")
    print(path, flush=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-prehistory", action="store_true")
    parser.add_argument("--snapshot-name")
    parser.add_argument("--fdsn-supplement", action="store_true")
    args = parser.parse_args()
    if args.fdsn_supplement:
        freeze_fdsn_supplement()
    else:
        freeze_snapshot(include_prehistory=not args.no_prehistory, snapshot_name=args.snapshot_name)
