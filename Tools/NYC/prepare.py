import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pyarrow.parquet as pq
from pyproj import CRS, Transformer
import shapefile
from shapely.geometry import shape
from shapely.ops import transform


def write(path, value):
    data = (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def prepare(raw, output, spec):
    for source in spec["sources"]:
        path = raw / source["file"]
        if path.stat().st_size != source["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Source hash mismatch: " + source["file"])
    output.mkdir(parents=True, exist_ok=False)
    ids = spec["zone_ids"]
    with zipfile.ZipFile(raw / "taxi_zones.zip") as archive:
        stem = next(n[:-4] for n in archive.namelist() if n.endswith(".shp"))
        reader = shapefile.Reader(**{ext: io.BytesIO(archive.read(stem + "." + ext)) for ext in ["shp", "shx", "dbf"]})
        crs = CRS.from_wkt(archive.read(stem + ".prj").decode())
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        areas = []
        for item in reader.iterShapeRecords():
            row = item.record.as_dict()
            if row["LocationID"] not in ids:
                continue
            original = shape(item.shape.__geo_interface__)
            geometry = transform(transformer.transform, original.simplify(spec["simplify_feet"], preserve_topology=True))
            if not geometry.is_valid:
                raise ValueError("Invalid zone geometry")
            center = geometry.representative_point()
            polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
            def points(ring):
                return [{"latitude": round(y, 6), "longitude": round(x, 6)} for x, y in ring.coords]
            index = ids.index(row["LocationID"])
            areas.append({"id": str(row["LocationID"]), "name": {"ja": row["zone"], "en": row["zone"]},
                          "district": "Manhattan", "grid": {"x": index % 5, "y": index // 5},
                          "center": {"latitude": round(center.y, 6), "longitude": round(center.x, 6)},
                          "polygons": [{"outer": points(p.exterior), "holes": [points(h) for h in p.interiors]} for p in polygons]})
    areas.sort(key=lambda a: int(a["id"]))
    if [int(a["id"]) for a in areas] != ids:
        raise ValueError("Zone inventory mismatch")
    city_hash = write(output / "city.json", {"synthetic": False, "areas": areas})
    table = pq.read_table(raw / "yellow_tripdata_2025-01.parquet", columns=["tpep_pickup_datetime", "PULocationID"])
    times = table["tpep_pickup_datetime"].to_numpy().astype("datetime64[m]")
    zones = table["PULocationID"].to_numpy()
    start = np.datetime64("2025-01-01T00:00", "m")
    end = np.datetime64("2025-02-01T00:00", "m")
    valid_time = ~np.isnat(times) & (times >= start) & (times < end)
    selected = valid_time & np.isin(zones, ids)
    bins = ((times[selected] - start).astype(np.int64) // 30).astype(int)
    columns = np.searchsorted(ids, zones[selected])
    counts = np.zeros((31 * 48, len(ids)), dtype=np.int64)
    np.add.at(counts, (bins, columns), 1)
    if int(counts.sum()) != int(selected.sum()):
        raise ValueError("Aggregation count mismatch")
    scenarios = []
    for scenario in spec["scenarios"]:
        local = datetime.fromisoformat(scenario["origin"]).replace(tzinfo=None).isoformat(timespec="minutes")
        index = int((np.datetime64(local, "m") - start).astype(np.int64) // 30)
        history = counts[index - spec["history_steps"]:index]
        truth = counts[index:index + spec["truth_steps"]]
        if history.shape != (spec["history_steps"], len(ids)) or truth.shape != (spec["truth_steps"], len(ids)):
            raise ValueError("Incomplete history or truth")
        item = {"id": scenario["id"], "name": scenario["name"], "focus_area": scenario["focus_area"],
                "forecast_origin": scenario["origin"]}
        for label, values in [("input", history), ("truth", truth)]:
            filename = scenario["id"] + "." + label + ".json"
            content = {"synthetic": False, "scenario_id": scenario["id"],
                       "series": [{"area_id": str(zone), "counts": values[:, i].tolist()} for i, zone in enumerate(ids)]}
            item[label + "_file"] = filename
            item[label + "_sha256"] = write(output / filename, content)
        scenarios.append(item)
    report = {"source_rows": table.num_rows, "invalid_or_outside_month_rows": int((~valid_time).sum()),
              "valid_time_outside_selected_zones": int((valid_time & ~selected).sum()),
              "selected_rows": int(selected.sum()), "aggregated_rows": int(counts.sum()),
              "bins": len(counts), "zones": len(ids), "crs": crs.to_string(), "simplify_feet": spec["simplify_feet"]}
    manifest = {"schema_version": 2, "synthetic": False, "city_sha256": city_hash,
                "source": {"title": spec["source_title"],
                           "url": "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page", "period": "2025-01",
                           "files": spec["sources"], "aggregation": report},
                "time": {"forecast_origin": scenarios[0]["forecast_origin"], "timezone": spec["timezone"],
                         "history_steps": spec["history_steps"], "forecast_steps": 4,
                         "simulation_future_steps": spec["truth_steps"], "interval_seconds": spec["interval_seconds"]},
                "scenarios": scenarios}
    write(output / "manifest.json", manifest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.raw, args.output, json.loads(Path(__file__).with_name("spec.json").read_text()))
