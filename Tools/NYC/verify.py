import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(data, raw=None):
    pins = json.loads(Path(__file__).with_name("resources.json").read_text())
    spec = json.loads(Path(__file__).with_name("spec.json").read_text())
    if {p.name for p in data.iterdir()} != set(pins["sha256"]):
        raise ValueError("Unexpected resource inventory")
    for name, expected in pins["sha256"].items():
        path = data / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("Resource hash mismatch: " + name)
    cache = json.loads((data / "forecasts.json").read_text())
    if cache["modelRevision"] != pins["modelRevision"] or cache["assetSHA256"] != pins["graphSHA256"]:
        raise ValueError("Cache model identity mismatch")
    sources = {k: v for k, v in pins["sha256"].items() if k != "forecasts.json"}
    if cache["sourceHashes"] != sources:
        raise ValueError("Cache source identity mismatch")
    result = {"resource_files": len(pins["sha256"]), "cache_frames": len(cache["frames"]), "independent_counts": None}
    if raw is not None:
        for source in spec["sources"]:
            path = raw / source["file"]
            if path.stat().st_size != source["bytes"] or digest(path) != source["sha256"]:
                raise ValueError("Source hash mismatch: " + source["file"])
        result["independent_counts"] = independent_counts(raw, data, spec)
    return result


def independent_counts(raw, data, spec):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(raw / "yellow_tripdata_2025-01.parquet", columns=["tpep_pickup_datetime", "PULocationID"])
    table = table.filter(pc.and_(pc.greater_equal(table["tpep_pickup_datetime"], pa.scalar(datetime(2025, 1, 1))),
                                 pc.less(table["tpep_pickup_datetime"], pa.scalar(datetime(2025, 2, 1)))))
    table = table.append_column("bin", pc.floor_temporal(table["tpep_pickup_datetime"], multiple=30, unit="minute"))
    grouped = table.group_by(["PULocationID", "bin"]).aggregate([("tpep_pickup_datetime", "count")])
    counts = {(row["PULocationID"], row["bin"]): row["tpep_pickup_datetime_count"] for row in grouped.to_pylist()}
    checked = 0
    for scene in spec["scenarios"]:
        origin = datetime.fromisoformat(scene["origin"]).replace(tzinfo=None)
        for label, offset in [("input", -spec["history_steps"]), ("truth", 0)]:
            source = json.loads((data / (scene["id"] + "." + label + ".json")).read_text())
            for series in source["series"]:
                for step, actual in enumerate(series["counts"]):
                    start = origin + timedelta(seconds=spec["interval_seconds"] * (offset + step))
                    expected = counts.get((int(series["area_id"]), start), 0)
                    if actual != expected:
                        raise ValueError((scene["id"], series["area_id"], offset + step, expected, actual))
                    checked += 1
    if checked != 61440:
        raise ValueError("Incomplete independent count comparison")
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.raw), indent=2))
