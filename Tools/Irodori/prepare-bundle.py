import argparse, ctypes, hashlib, json, os, shutil, subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "CoreAISamplesKit/Sources/SpeechAssets/Resources/irodori-assets.json"
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

def verify_files(artifacts, tokenizer, rules):
    for spec in rules:
        for name, expected in spec["files"].items():
            file = artifacts / (spec["stem"] + ".aimodel") / name
            if file.stat().st_size != expected["bytes"] or sha256(file) != expected["sha256"]:
                raise ValueError("Model file does not match the verified FP32 version: " + str(file))
    if sha256(tokenizer) != TOKENIZER_SHA:
        raise ValueError("Tokenizer SHA-256 mismatch")

def prepare(artifacts, tokenizer, destination):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    rules = json.loads(RULES.read_text())
    verify_files(artifacts, tokenizer, rules)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".irodori-", dir=destination.parent))
    try:
        (staging / "artifacts").mkdir()
        for spec in rules:
            name = spec["stem"] + ".aimodel"
            target = staging / "artifacts" / name
            target.mkdir()
            for relative in spec["files"]:
                source = (artifacts / name / relative).resolve(strict=True)
                subprocess.run(["/bin/cp", "-c", str(source), str(target / relative)], check=True)
        shutil.copyfile(tokenizer, staging / "tokenizer.json")
        (staging / "irodori.json").write_text(json.dumps({
            "format": "irodori-coreai-v1", "modelID": "Aratako/Irodori-TTS-v4.1-Small-MF",
            "precision": "fp32", "sampleRate": 48000
        }, indent=2) + "\n")
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
    args = parser.parse_args()
    print(prepare(args.artifacts, args.tokenizer, args.output))
