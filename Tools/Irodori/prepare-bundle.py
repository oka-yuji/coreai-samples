import argparse, ctypes, hashlib, json, os, shutil, subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "CoreAISamplesKit/Sources/SpeechAssets/Resources/irodori-assets.json"
REFERENCE_RULES = ROOT / "CoreAISamplesKit/Sources/SpeechAssets/Resources/irodori-reference-assets.json"
TOKENIZER_SHA = "6a0734cf21c802169defaffe719bc2ef12bb9d0be37e54b61ed27aa89394723d"

def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

def rename_exclusive(source, destination):
    system = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    rename = system.renamex_np
    rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(os.fsencode(source), os.fsencode(destination), 4) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))

def asset_directory(artifacts, spec, reference_artifacts=None):
    name = spec["stem"] + ".aimodel"
    if reference_artifacts is not None and (reference_artifacts / name).is_dir():
        return reference_artifacts / name
    return artifacts / name

def verify_files(artifacts, tokenizer, rules, reference_artifacts=None):
    for spec in rules:
        for name, expected in spec["files"].items():
            file = asset_directory(artifacts, spec, reference_artifacts) / name
            if file.stat().st_size != expected["bytes"] or sha256(file) != expected["sha256"]:
                raise ValueError("Model file does not match the verified FP32 version: " + str(file))
    if sha256(tokenizer) != TOKENIZER_SHA:
        raise ValueError("Tokenizer SHA-256 mismatch")

def make_manifest(reference_inventory=None):
    payload = {
        "format": "irodori-coreai-v2" if reference_inventory is not None else "irodori-coreai-v1",
        "modelID": "Aratako/Irodori-TTS-v4.1-Small-MF",
        "precision": "fp32", "sampleRate": 48000
    }
    if reference_inventory is not None:
        payload["assetInventorySHA256"] = hashlib.sha256(reference_inventory).hexdigest()
    return (json.dumps(payload, indent=2) + "\n").encode()

def prepare(artifacts, tokenizer, destination, reference_artifacts=None):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    inventory = (REFERENCE_RULES if reference_artifacts is not None else RULES).read_bytes()
    rules = json.loads(inventory)
    verify_files(artifacts, tokenizer, rules, reference_artifacts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".irodori-", dir=destination.parent))
    try:
        (staging / "artifacts").mkdir()
        for spec in rules:
            name = spec["stem"] + ".aimodel"
            target = staging / "artifacts" / name
            target.mkdir()
            for relative in spec["files"]:
                source = (asset_directory(artifacts, spec, reference_artifacts) / relative).resolve(strict=True)
                if source.stat().st_dev == target.stat().st_dev:
                    subprocess.run(["/bin/cp", "-c", str(source), str(target / relative)], check=True)
                else:
                    shutil.copyfile(source, target / relative)
        shutil.copyfile(tokenizer, staging / "tokenizer.json")
        (staging / "irodori.json").write_bytes(make_manifest(inventory if reference_artifacts is not None else None))
        verify_files(staging / "artifacts", staging / "tokenizer.json", rules)
        rename_exclusive(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-artifacts", type=Path)
    args = parser.parse_args()
    print(prepare(args.artifacts, args.tokenizer, args.output, args.reference_artifacts))
