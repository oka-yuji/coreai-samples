import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from ...storage import file_hash, write_json


def verify_file(path, expected):
    actual = file_hash(Path(path))
    if actual != {key: expected[key] for key in ("bytes", "sha256")}:
        raise ValueError(f"Source identity mismatch: {Path(path).name}")
    return actual


def extract_archive(source, destination):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    with tarfile.open(source, "r:*") as archive:
        selected = {}
        occurrences = {}
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\\" in member.name:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"Unsupported archive entry: {member.name}")
            if name == PurePosixPath("."):
                if not member.isdir():
                    raise ValueError("Archive root must be a directory")
                continue
            key = str(name)
            previous = selected.get(key)
            if previous and previous.isfile() != member.isfile():
                raise ValueError(f"Conflicting archive entries: {key}")
            selected[key] = member
            occurrences[key] = occurrences.get(key, 0) + 1
        for key in selected:
            for ancestor in PurePosixPath(key).parents:
                if str(ancestor) in selected and not selected[str(ancestor)].isdir():
                    raise ValueError(f"File used as an archive directory: {ancestor}")
        if not all(name in selected for name in ("model_config.yaml", "model_weights.ckpt")):
            raise ValueError("The NeMo archive is incomplete")
        destination.mkdir(parents=True)
        for name, member in sorted(selected.items()):
            target = destination / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
    return {"entries": len(selected), "duplicates": {k: v for k, v in occurrences.items() if v > 1}}


def compare_tensor(expected, actual, normalized_max_limit=0.001):
    import numpy as np

    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if expected.shape != actual.shape or not expected.size:
        raise ValueError("Comparison requires equal, nonempty shapes")
    if not np.isfinite(expected).all():
        raise ValueError("The reference contains non-finite values")
    finite = bool(np.isfinite(actual).all())
    if not finite:
        return {"passed": False, "finite": False, "shape": list(actual.shape)}
    error = expected - actual
    maximum = float(np.max(np.abs(error)))
    scale = float(np.max(np.abs(expected)))
    normalized = maximum / max(scale, 1e-8)
    norm = float(np.linalg.norm(expected.ravel()) * np.linalg.norm(actual.ravel()))
    cosine = float(np.dot(expected.ravel(), actual.ravel()) / norm) if norm else None
    return {"passed": normalized <= normalized_max_limit, "finite": True,
            "shape": list(actual.shape), "max_abs": maximum, "normalized_max": normalized,
            "rms": float(np.sqrt(np.mean(error**2))), "cosine": cosine}


def read_contract(path):
    contract = json.loads(Path(path).read_text())
    if contract["version"] != 1 or contract["precision"] != "float32":
        raise ValueError("Only the fixed FP32 research profile is supported")
    return contract


def prepare(contract_path, model_archive, codec_archive, output):
    contract = read_contract(contract_path)
    model_identity = verify_file(model_archive, contract["model"])
    codec_identity = verify_file(codec_archive, contract["codec"])
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    model_layout = extract_archive(model_archive, output / "magpie")
    codec_layout = extract_archive(codec_archive, output / "codec")
    write_json(output / "source.json", {"complete": True, "contract": contract,
                                       "model": model_identity, "codec": codec_identity,
                                       "archives": {"model": model_layout, "codec": codec_layout}})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.contract, args.model, args.codec, args.output)
