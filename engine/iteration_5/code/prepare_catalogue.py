"""Validate and prepare the frozen GeoNet WFS CSVs; record every exclusion.

python iteration_5/code/prepare_catalogue.py --manifest <raw_manifest.json>
No remote data are fetched by this program. Raw response hashes are verified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = datetime(1960, 1, 1, tzinfo=timezone.utc)
CUTOFF = datetime(2026, 10, 15, tzinfo=timezone.utc)
FIELDS = ["event_id", "time_utc", "t_days", "magnitude", "excess", "depth", "latitude", "longitude", "magnitude_type", "event_type"]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def parse_time(value):
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("UTC offset absent")
    return date.astimezone(timezone.utc)


def finite(value):
    try:
        x = float(value)
    except (ValueError, TypeError):
        return None
    return x if math.isfinite(x) else None


def select_rows(raw_rows, *, start=ORIGIN, end=CUTOFF):
    """Keep original non-sentinel negative depths; fail on conflicting IDs.

    Returns prepared rows, all-row ledger, and duplicate statistics. Reasons
    overlap in all_reasons; primary_reason makes the dropped-row sum exact.
    """
    selected, ledger, seen = [], [], {}
    duplicates = 0
    for index, raw in enumerate(raw_rows):
        event_id = raw.get("publicid", "").strip()
        reasons = []
        if not event_id:
            reasons.append("missing_event_id")
        if event_id and event_id in seen:
            values = {key: value for key, value in raw.items() if not key.startswith("_")}
            previous = {key: value for key, value in seen[event_id].items() if not key.startswith("_")}
            if values != previous:
                raise ValueError(f"Conflicting duplicate event ID {event_id}")
            reasons.append("exact_duplicate_event_id")
            duplicates += 1
        elif event_id:
            seen[event_id] = raw
        try:
            date = parse_time(raw.get("origintime", ""))
        except (ValueError, TypeError):
            date = None
            reasons.append("invalid_or_missing_origin_time")
        if date is not None and not start <= date < end:
            reasons.append("outside_time_interval")
        magnitude = finite(raw.get("magnitude"))
        depth = finite(raw.get("depth"))
        latitude = finite(raw.get("latitude"))
        longitude = finite(raw.get("longitude"))
        if magnitude is None:
            reasons.append("invalid_or_missing_magnitude")
        elif magnitude < 5:
            reasons.append("magnitude_below_5")
        if latitude is None or longitude is None:
            reasons.append("invalid_or_missing_coordinates")
        elif not (165 <= longitude <= 180 and -48.5 <= latitude <= -33.5):
            reasons.append("outside_bbox")
        event_type = raw.get("eventtype", "").strip()
        if event_type != "earthquake":
            reasons.append("event_type_not_earthquake" if event_type else "missing_event_type")
        if depth is None:
            reasons.append("invalid_or_missing_depth")
        elif depth == -9:
            reasons.append("unknown_depth_sentinel_minus_9")
        elif depth > 40:
            reasons.append("depth_over_40_km")
        record = dict(raw_index=index, event_id=event_id,
                      source_file=raw.get("_source_file", ""), source_row=raw.get("_source_row", ""),
                      included=not reasons, primary_reason=reasons[0] if reasons else "included",
                      all_reasons=reasons, raw_event_type=event_type,
                      raw_magnitude_type=raw.get("magnitudetype", ""))
        ledger.append(record)
        if not reasons:
            selected.append(dict(event_id=event_id,
                time_utc=date.isoformat().replace("+00:00", "Z"),
                t_days=(date-ORIGIN).total_seconds()/86400,
                magnitude=magnitude, excess=magnitude-5, depth=depth,
                latitude=latitude, longitude=longitude,
                magnitude_type=raw.get("magnitudetype", "").strip(), event_type=event_type))
    selected.sort(key=lambda row: (row["t_days"], row["event_id"]))
    # Do not jitter timestamps to make a singular kernel appear valid.
    return selected, ledger, dict(exact_duplicate_rows=duplicates, unique_event_ids=len(seen), conflicting_duplicate_rows=0)


def csv_bytes(rows, fields):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def verified_response(record):
    path = ROOT/record["destination"]
    data = path.read_bytes()
    if sha256(data) != record["sha256"] or len(data) != record["response_bytes"]:
        raise ValueError(f"Raw response checksum/size mismatch: {path}")
    return data


def read_wfs(record):
    data = verified_response(record)
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    for index, row in enumerate(rows, 2):
        row.update(_source_file=record["destination"], _source_row=index)
    return rows


def read_fdsn(record):
    text = verified_response(record).decode("utf-8-sig")
    lines = list(csv.reader(io.StringIO(text), delimiter="|"))
    if not lines or not lines[0][0].startswith("#EventID") or len(lines[0]) != 14:
        raise ValueError("Expected FDSN 1.2 text schema including event type")
    result = {}
    for row in lines[1:]:
        if not row:
            continue
        if len(row) != 14:
            raise ValueError(f"Unexpected FDSN row length {len(row)}")
        event_id = row[0].strip()
        if event_id in result:
            raise ValueError(f"Duplicate FDSN event ID {event_id}")
        # FDSN text service prints UTC timestamps without a literal Z.
        when = parse_time(row[1].strip()+"Z")
        if not ORIGIN <= when < CUTOFF:
            raise ValueError("FDSN inclusive endpoint returned a row outside main exclusive interval; revise cross-check with explicit endpoint ledger")
        result[event_id] = dict(event_id=event_id, time_utc=row[1].strip(),
            latitude=float(row[2]), longitude=float(row[3]), depth=float(row[4]),
            magnitude_type=row[9].strip(), magnitude=float(row[10]), event_type=row[13].strip())
    return result


def grouping(rows, field):
    return dict(sorted(Counter(row.get(field, "") for row in rows).items()))


def read_quakeml(record):
    """Stream full-precision preferred values; depth in QuakeML is metres."""
    path = ROOT/record["destination"]
    # Verify raw bytes without retaining a second 200MB copy during XML parsing.
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
            size += len(block)
    if digest.hexdigest() != record["sha256"] or size != record["response_bytes"]:
        raise ValueError("QuakeML raw checksum/size mismatch")
    ns = "{http://quakeml.org/xmlns/bed/1.2}"
    rows = []
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != ns+"event":
            continue
        def text(node, key, default=""):
            return node.findtext("/".join(ns+part for part in key.split("/")), default)
        event_id = element.attrib["publicID"].rsplit("/", 1)[-1]
        origin_id = text(element, "preferredOriginID")
        magnitude_id = text(element, "preferredMagnitudeID")
        origins = [node for node in element.findall(ns+"origin") if node.get("publicID") == origin_id]
        magnitudes = [node for node in element.findall(ns+"magnitude") if node.get("publicID") == magnitude_id]
        if not origin_id or not magnitude_id or not origins or not magnitudes:
            raise ValueError(f"Missing preferred origin/magnitude reference for {event_id}")
        origin_values = {(text(node,"time/value"), text(node,"latitude/value"), text(node,"longitude/value"), text(node,"depth/value")) for node in origins}
        magnitude_values = {(text(node,"mag/value"), text(node,"type")) for node in magnitudes}
        if len(origin_values) != 1 or len(magnitude_values) != 1:
            raise ValueError(f"Conflicting repeated preferred origin/magnitude for {event_id}")
        when, latitude, longitude, depth_m = origin_values.pop()
        magnitude, magnitude_type = magnitude_values.pop()
        depth = finite(depth_m)
        rows.append(dict(publicid=event_id, origintime=when, latitude=latitude, longitude=longitude,
                         depth="" if depth is None else str(depth/1000), magnitude=magnitude,
                         magnitudetype=magnitude_type, eventtype=text(element,"type"),
                         _source_file=record["destination"], _source_row=len(rows)+1,
                         _preferred_origin_id=origin_id, _preferred_magnitude_id=magnitude_id))
        element.clear()
    return rows


def prepare_fdsn_sensitivity(manifest_path, main_metadata_path=None, output_dir=None):
    manifest_path = Path(manifest_path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    output_dir = Path(output_dir) if output_dir else ROOT/"data"/"processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    main_metadata_path = Path(main_metadata_path) if main_metadata_path else ROOT/"data"/"processed"/"catalogue_metadata.json"
    main_metadata = json.loads(main_metadata_path.read_text())
    primary_payload = (ROOT/main_metadata["processed_csv"]).read_bytes()
    if sha256(primary_payload) != main_metadata["processed_sha256"]:
        raise ValueError("Primary WFS checksum mismatch")
    primary = {row["event_id"]:row for row in csv.DictReader(io.StringIO(primary_payload.decode()))}
    raw_rows = read_quakeml(manifest["response"])
    selected, ledger, duplicates = select_rows(raw_rows)
    output_path = output_dir/"geonet_FDSN_M5_shallow_1960_2026.csv"
    payload = csv_bytes(selected, FIELDS)
    output_path.write_bytes(payload)
    extracted = [{key:value for key,value in row.items() if not key.startswith("_")} for row in raw_rows]
    extraction_bytes = csv_bytes(extracted, ["publicid","origintime","latitude","longitude","depth","magnitude","magnitudetype","eventtype"])
    (output_dir/"fdsn_preferred_all_events.csv").write_bytes(extraction_bytes)
    ledger_bytes = ("\n".join(json.dumps(row,sort_keys=True) for row in ledger)+"\n").encode()
    (output_dir/"fdsn_filter_ledger.jsonl").write_bytes(ledger_bytes)
    separate = {row["event_id"]:row for row in selected}
    primary_ids, separate_ids = set(primary), set(separate)
    common = sorted(primary_ids & separate_ids)
    common_rows, material_differences = [], []
    numeric = ["t_days","magnitude","excess","depth","latitude","longitude"]
    maximum_deltas = {key:0 for key in numeric}
    exact_mismatch_counts = Counter()
    for event_id in common:
        w, f = primary[event_id], separate[event_id]
        pair = {"event_id":event_id}
        material = {}
        for field in FIELDS[1:]:
            pair[field+"_wfs"], pair[field+"_fdsn"] = w[field], f[field]
            if field in numeric:
                delta = float(f[field])-float(w[field])
                pair[field+"_delta_fdsn_minus_wfs"] = delta
                maximum_deltas[field] = max(maximum_deltas[field],abs(delta))
                if delta != 0:
                    exact_mismatch_counts[field] += 1
                tolerance = 1e-3/86400 if field=="t_days" else 1e-7
                if abs(delta)>tolerance:
                    material[field] = dict(WFS=float(w[field]), FDSN=float(f[field]), delta=delta)
            elif field == "time_utc":
                if parse_time(w[field]) != parse_time(f[field]):
                    exact_mismatch_counts[field] += 1
                if abs((parse_time(f[field])-parse_time(w[field])).total_seconds())>.001:
                    material[field] = dict(WFS=w[field], FDSN=f[field])
            elif w[field] != f[field]:
                exact_mismatch_counts[field] += 1
                material[field] = dict(WFS=w[field], FDSN=f[field])
        common_rows.append(pair)
        if material:
            material_differences.append(dict(event_id=event_id, differences=material))
    common_bytes = csv_bytes(common_rows,list(common_rows[0]) if common_rows else ["event_id"])
    (output_dir/"fdsn_wfs_common_selected_values.csv").write_bytes(common_bytes)
    time_counts = Counter(row["time_utc"] for row in selected)
    metadata = dict(schema_version=1, provenance="Separate FDSN full-precision preferred-origin and preferred-magnitude selection; never merged with primary WFS dataset",
        raw_manifest_path=str(manifest_path.relative_to(ROOT)), raw_manifest_sha256=sha256(manifest_bytes),
        retrieval_started_at_utc=manifest["started_at_utc"], retrieval_completed_at_utc=manifest["completed_at_utc"],
        processed_csv=str(output_path.relative_to(ROOT)) if output_path.is_relative_to(ROOT) else str(output_path),
        processed_sha256=sha256(payload), columns=FIELDS, horizon_days=(CUTOFF-ORIGIN).days,
        full_precision_source="QuakeML 1.2, referenced event preferredOriginID and preferredMagnitudeID; metres divided by 1000 for depth km",
        same_filters_as_primary=main_metadata["local_filters"], units=main_metadata["units"],
        counts=dict(raw_rows=len(raw_rows), selected=len(selected), excluded=len(raw_rows)-len(selected),
                    raw_by_type=grouping(raw_rows,"eventtype"), selected_by_type=grouping(selected,"event_type"),
                    selected_by_year=dict(sorted(Counter(row["time_utc"][:4] for row in selected).items())),
                    selected_by_magnitude_type=grouping(selected,"magnitude_type"),
                    primary_reason=dict(sorted(Counter(row["primary_reason"] for row in ledger).items())), **duplicates),
        strict_increasing_times=all(selected[i]["t_days"]<selected[i+1]["t_days"] for i in range(len(selected)-1)),
        ties={key:value for key,value in time_counts.items() if value>1}, tie_policy=main_metadata["ordering"]["tie_policy"],
        comparison=dict(primary_wfs_sha256=main_metadata["processed_sha256"], primary_count=len(primary), separate_fdsn_count=len(separate),
                        common_count=len(common), wfs_only_ids=sorted(primary_ids-separate_ids), fdsn_only_ids=sorted(separate_ids-primary_ids),
                        common_values_file="fdsn_wfs_common_selected_values.csv", common_values_sha256=sha256(common_bytes),
                        maximum_absolute_numeric_differences=maximum_deltas, exact_mismatch_counts=dict(exact_mismatch_counts),
                        material_difference_tolerances="1 millisecond in time; 1e-7 in other numeric fields; any type difference",
                        material_common_event_differences=material_differences,
                        wfs_only_records=[primary[key] for key in sorted(primary_ids-separate_ids)],
                        fdsn_only_records=[separate[key] for key in sorted(separate_ids-primary_ids)],
                        limitation="FDSN superset coverage verifies returned WFS rows, not completeness of WFS. Interface differences remain unresolved; results are separate sensitivity analyses."),
        filter_ledger_sha256=sha256(ledger_bytes), preferred_extraction_sha256=sha256(extraction_bytes),
        preparation_code_sha256=sha256(Path(__file__).read_bytes()))
    (output_dir/"fdsn_catalogue_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(N=len(selected), processed_csv=metadata["processed_csv"], processed_sha256=metadata["processed_sha256"],
                         wfs_only=metadata["comparison"]["wfs_only_ids"], fdsn_only=metadata["comparison"]["fdsn_only_ids"],
                         material_common_differences=len(material_differences)),indent=2))
    return metadata


def prepare(manifest_path, output_dir=None):
    manifest_path = Path(manifest_path).resolve()
    raw_manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest_bytes)
    output_dir = Path(output_dir) if output_dir else ROOT/"data"/"processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows, pre_rows = [], []
    for chunk in manifest["chunks"]:
        rows = read_wfs(chunk["response"])
        if len(rows) != chunk["rows"]:
            raise ValueError("Manifest row count does not match frozen CSV")
        (raw_rows if chunk["role"] == "main" else pre_rows).extend(rows)
    selected, ledger, duplicates = select_rows(raw_rows)
    pre_selected, pre_ledger, pre_duplicates = select_rows(pre_rows,
        start=datetime(1950, 1, 1, tzinfo=timezone.utc), end=ORIGIN)
    payload = csv_bytes(selected, FIELDS)
    processed_path = output_dir/"geonet_M5_shallow_1960_2026.csv"
    processed_path.write_bytes(payload)
    pre_payload = csv_bytes(pre_selected, FIELDS)
    pre_path = output_dir/"geonet_M5_shallow_1950_1959.csv"
    pre_path.write_bytes(pre_payload)
    ledger_bytes = ("\n".join(json.dumps(row, sort_keys=True) for row in ledger)+"\n").encode("utf-8")
    (output_dir/"filter_ledger.jsonl").write_bytes(ledger_bytes)
    (output_dir/"prehistory_filter_ledger.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in pre_ledger)+"\n", encoding="utf-8")
    by_time = defaultdict(list)
    for row in selected:
        by_time[row["time_utc"]].append(row["event_id"])
    ties = {key: value for key, value in by_time.items() if len(value) > 1}
    float_ties = len(selected)-len({row["t_days"] for row in selected})
    raw_by_id = {row["publicid"]: row for row in raw_rows}
    fdsn = read_fdsn(manifest["crosscheck"]["response"])
    expanded = read_fdsn(manifest["crosscheck"]["expanded_response"])
    primary_ids, expanded_ids, wfs_ids = set(fdsn), set(expanded), set(raw_by_id)
    missing = sorted(wfs_ids-primary_ids)
    expanded_missing = sorted(wfs_ids-expanded_ids)
    fdsn_selected = {key for key, row in fdsn.items() if row["event_type"] == "earthquake" and row["depth"] <= 40 and row["depth"] != -9}
    selected_ids = {row["event_id"] for row in selected}
    diagnostic_rows = read_wfs(manifest["crosscheck"]["diagnostic_response"])
    diagnostics = [{key: value for key, value in row.items() if not key.startswith("_")} for row in diagnostic_rows]
    common = wfs_ids & expanded_ids
    differences = []
    for event_id in sorted(common):
        w, f = raw_by_id[event_id], expanded[event_id]
        changed = {}
        for wk, fk in (("eventtype", "event_type"), ("magnitudetype", "magnitude_type")):
            if w[wk] != f[fk]:
                changed[wk] = {"WFS": w[wk], "FDSN": f[fk]}
        # FDSN text prints rounded depth/magnitude; compare at its precision.
        for wk in ("magnitude", "depth"):
            if abs(float(w[wk])-f[wk]) > .05000001:
                changed[wk] = {"WFS": float(w[wk]), "FDSN": f[wk]}
        if changed:
            differences.append(dict(event_id=event_id, differences=changed))
    old_cutoff = datetime(2026, 10, 14, tzinfo=timezone.utc)
    after_old = [row["event_id"] for row in selected if parse_time(row["time_utc"]) >= old_cutoff]
    mean_excess = sum(row["excess"] for row in selected)/len(selected)
    beta_aki = 1/(mean_excess+.05)
    output = dict(schema_version=1,
        provenance="Fresh timestamped public GeoNet WFS snapshot, independently prepared; original frozen catalogue and original analysis code unavailable",
        raw_manifest_path=str(manifest_path.relative_to(ROOT)), raw_manifest_sha256=sha256(raw_manifest_bytes),
        retrieval_started_at_utc=manifest["started_at_utc"], retrieval_completed_at_utc=manifest["completed_at_utc"],
        frozen_cutoff_utc=CUTOFF.isoformat().replace("+00:00", "Z"), start_inclusive_utc=ORIGIN.isoformat().replace("+00:00", "Z"),
        horizon_days=(CUTOFF-ORIGIN).total_seconds()/86400,
        processed_csv=str(processed_path.relative_to(ROOT)) if processed_path.is_relative_to(ROOT) else str(processed_path),
        processed_sha256=sha256(payload), processed_bytes=len(payload), columns=FIELDS,
        preparation_code_sha256=sha256(Path(__file__).read_bytes()),
        filter_ledger_sha256=sha256(ledger_bytes),
        units=dict(t_days="86400 SI seconds per UTC calendar day since 1960-01-01T00:00:00Z, using Python datetime (leap seconds not modelled)",
                   magnitude="GeoNet preferred magnitude in original magnitude_type; no homogenisation", excess="magnitude - 5", depth="km", latitude="degrees north", longitude="degrees east"),
        local_filters=dict(event_type="exactly earthquake", magnitude="finite >=5.0", longitude="165 <= lon <= 180", latitude="-48.5 <= lat <= -33.5", depth="finite <=40 km, excluding unknown sentinel -9 km; other negative depths retained", time="1960-01-01T00:00:00Z <= time < 2026-10-15T00:00:00Z"),
        counts=dict(raw_rows=len(raw_rows), selected=len(selected), excluded=len(raw_rows)-len(selected),
                    raw_by_type=grouping(raw_rows,"eventtype"), raw_by_magnitude_type=grouping(raw_rows,"magnitudetype"),
                    raw_by_year=dict(sorted(Counter(row["origintime"][:4] for row in raw_rows).items())),
                    selected_by_year=dict(sorted(Counter(row["time_utc"][:4] for row in selected).items())),
                    selected_by_magnitude_type=grouping(selected,"magnitude_type"), selected_by_type=grouping(selected,"event_type"),
                    primary_reason=dict(sorted(Counter(row["primary_reason"] for row in ledger).items())),
                    all_exclusion_reasons=dict(sorted(Counter(reason for row in ledger for reason in row["all_reasons"]).items())),
                    **duplicates),
        ordering=dict(raw_input_monotonically_increasing=all(raw_rows[i]["origintime"]<=raw_rows[i+1]["origintime"] for i in range(len(raw_rows)-1)),
                      processed_order="UTC origin time ascending, event ID ascending for ties", timestamp_ties=ties,
                      repeated_float_t_days=float_ties,
                      strict_increasing_times=not ties and float_ties==0,
                      tie_policy="Retain ties without jitter. Singular ML kernel fitting must fail explicitly if ties occur; no sequential likelihood on a jittered catalogue.",
                      minimum_gap_days=min((selected[i+1]["t_days"]-selected[i]["t_days"] for i in range(len(selected)-1)), default=None)),
        missing_raw_fields={key:sum(not str(row.get(key,"")).strip() for row in raw_rows) for key in ("publicid","origintime","magnitude","depth","latitude","longitude","magnitudetype","eventtype")},
        first_event=selected[0] if selected else None, last_event=selected[-1] if selected else None,
        sensitivity=dict(previous_end_exclusive_utc="2026-10-14T00:00:00Z", previous_horizon_days=(old_cutoff-ORIGIN).days,
                         selected_on_final_day=after_old, count_with_previous_cutoff=len(selected)-len(after_old),
                         negative_depth_selected=[row["event_id"] for row in selected if row["depth"]<0],
                         count_if_depth_nonnegative_required=sum(row["depth"]>=0 for row in selected),
                         unknown_depth_sentinel_raw=[row["publicid"] for row in raw_rows if finite(row.get("depth"))==-9]),
        magnitude_summary=dict(mean_excess=mean_excess, beta_continuous_mle=1/mean_excess,
                               beta_aki_0p1_rounding=beta_aki, b_aki_0p1_rounding=beta_aki/math.log(10),
                               caveat="Descriptive estimates; a 0.1 magnitude bin correction is an assumption, with heterogeneous magnitude types and native precisions; not evidence of catalogue completeness or iid marks."),
        fdsn_interface_check=dict(primary_minmagnitude=5, primary_rows=len(fdsn), primary_missing_wfs_ids=missing,
            primary_missing_wfs_magnitude_counts=dict(Counter(raw_by_id[key]["magnitude"] for key in missing)),
            primary_extra_ids=sorted(primary_ids-wfs_ids), primary_shallow_earthquakes=len(fdsn_selected),
            primary_selected_ids_missing=sorted(selected_ids-fdsn_selected), primary_selected_extra_ids=sorted(fdsn_selected-selected_ids),
            expanded_minmagnitude=4.9, expanded_rows=len(expanded), expanded_missing_wfs_ids=expanded_missing,
            expanded_contains_all_wfs_ids=not expanded_missing,
            expanded_eventtype_mismatches=[key for key in selected_ids&expanded_ids if expanded[key]["event_type"]!="earthquake"],
            common_id_count=len(common), material_field_differences_beyond_text_rounding=differences,
            diagnostic_wfs_records=diagnostics,
            completeness_limitation="Coverage of returned WFS IDs does not establish WFS completeness. A separate FDSN QuakeML selection is supplied in fdsn_catalogue_metadata.json, including its own additional qualifying event.",
            interpretation="Two interfaces of the same GeoNet catalogue. FDSN minmagnitude=5 excludes the WFS M=5.000 boundary here; expanded 4.9 query checks ID coverage. Text precision and some underlying values differ; differences remain unresolved rather than reconciled by substitution. Main data use full-precision WFS values throughout."),
        prehistory=dict(start="1950-01-01T00:00:00Z", end_exclusive="1960-01-01T00:00:00Z", raw_rows=len(pre_rows), selected=len(pre_selected),
                        processed_csv=str(pre_path.relative_to(ROOT)) if pre_path.is_relative_to(ROOT) else str(pre_path), processed_sha256=sha256(pre_payload),
                        t_days_relative_to_main_origin=True, note="Optional observed prehistory for sensitivity, never pooled into main N or T.", duplicates=pre_duplicates),
        documentation=manifest["documentation"], hits_probe=manifest["hits_probe"])
    (output_dir/"catalogue_metadata.json").write_text(json.dumps(output, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({key:output[key] for key in ("processed_csv","processed_sha256","horizon_days")} | {"N":len(selected),"expanded_fdsn_missing":len(expanded_missing),"ties":len(ties)}, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fdsn-sensitivity", action="store_true")
    args = parser.parse_args()
    if args.fdsn_sensitivity:
        prepare_fdsn_sensitivity(args.manifest, output_dir=args.output_dir)
    else:
        prepare(args.manifest, args.output_dir)
