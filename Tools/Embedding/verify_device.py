import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid


def run(command, timeout=300):
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, timeout=timeout)


def validate_results(precision, manifest_sha256, summary, vectors):
    if summary.get("precision") != precision or summary.get("manifestSHA256") != manifest_sha256:
        raise RuntimeError("The device did not verify the requested bundle; rebuild the app and copy the bundle again")
    if summary.get("passed") is not True or len(vectors) != 20 or not all(row.get("passed") is True for row in vectors):
        raise RuntimeError("Device validation failed; retained outputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    identifier = "com.coreai-samples.DemoApp.ios"
    manifest = json.loads((args.bundle / "embedding-manifest.json").read_text())
    manifest_sha256 = hashlib.sha256((args.bundle / "embedding-manifest.json").read_bytes()).hexdigest()
    quantized = manifest["precision"] == "int4-int8-ptq-s128-v70"
    if manifest["precision"] not in ("fp16-safe-s128-v70", "int4-int8-ptq-s128-v70"):
        raise ValueError("Unsupported embedding profile")
    folder = "embeddinggemma2-int4-int8" if quantized else "embeddinggemma2"
    run_tag = args.run + "-" + uuid.uuid4().hex
    (args.output / "invocation.json").write_text(json.dumps(
        {"run": run_tag, "precision": manifest["precision"], "manifestSHA256": manifest_sha256}) + "\n")
    device = ["--device", args.device]
    run(["xcrun", "devicectl", "device", "install", "app", *device, str(args.app),
         "--json-output", str(args.output / "install.json")])
    run(["xcrun", "devicectl", "device", "copy", "to", *device, "--source", str(args.bundle),
         "--destination", "Documents/" + folder, "--domain-type", "appDataContainer",
         "--domain-identifier", identifier], timeout=600)
    command = ["xcrun", "devicectl", "device", "process", "launch", *device,
               "--terminate-existing", "--console", identifier, "--image-search-eval", run_tag]
    if quantized:
        command += ["--embedding-profile", "int4_int8"]
    process = subprocess.Popen(command)
    try:
        result = process.wait(timeout=900)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=30)
        raise
    run(["xcrun", "devicectl", "device", "copy", "from", *device,
         "--source", "Documents/embeddinggemma2-eval-" + run_tag,
         "--destination", str(args.output / "report"), "--domain-type", "appDataContainer",
         "--domain-identifier", identifier])
    summary = json.loads((args.output / "report/summary.json").read_text())
    vectors = json.loads((args.output / "report/vectors.json").read_text())
    if (args.output / "report/failure.txt").exists() or result:
        raise RuntimeError("Device validation failed; retained outputs")
    validate_results(manifest["precision"], manifest_sha256, summary, vectors)
    print(json.dumps({"passed": True, "precision": manifest["precision"], "vectors": len(vectors),
                      "minimumCosine": min(row["cosine"] for row in vectors),
                      "maximumDifference": max(row["maxDifference"] for row in vectors)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
