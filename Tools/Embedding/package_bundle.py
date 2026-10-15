import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def package(source, text, vision, reference, output):
    if output.exists():
        raise FileExistsError(output)
    fixtures_data = (reference / "fixtures.json").read_bytes()
    if json.loads(fixtures_data).get("model_revision") != REVISION:
        raise ValueError("The reference must identify the pinned source revision")
    fixtures_hash = hashlib.sha256(fixtures_data).hexdigest()
    for kind, path in (("text", text), ("vision", vision)):
        contract = json.loads((path / "contract.json").read_text())
        if contract.get("kind") != kind or contract.get("precision") != "fp16_safe" or contract.get("complete") is not True:
            raise ValueError("Only completed safe FP16 text/vision exports can be packaged")
        if contract.get("fixtures_sha256") != fixtures_hash or contract.get("model_revision") != REVISION:
            raise ValueError("The encoder and reference identities differ")
    pin = json.loads((Path(__file__).parents[1] / "Conversion/recipes/embeddinggemma2/model.json").read_text())
    if source.name != REVISION or pin["revision"] != REVISION:
        raise ValueError("The source must be the pinned immutable snapshot")
    for name in ("tokenizer.json", "tokenizer_config.json", "README.md"):
        identity = pin["source_files"][name]
        path = source / name
        if path.stat().st_size != identity["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]:
            raise ValueError(f"Pinned packaging source differs: {name}")
    table = np.load(reference / "token_embeddings.npy", mmap_mode="r", allow_pickle=False)
    if table.shape != (262144, 512) or table.dtype != np.float32:
        raise ValueError("Unexpected embedding table contract")
    bits = table.view(np.uint32)
    if np.any(bits & 0xFFFF) or not np.isfinite(table).all():
        raise ValueError("Table is not an exact BF16 expansion")
    output.mkdir(parents=True)
    for kind, path in (("text", text), ("vision", vision)):
        shutil.copytree(path / f"{kind}.aimodel", output / f"{kind}.aimodel")
    for name in ("tokenizer.json", "tokenizer_config.json", "README.md"):
        shutil.copyfile(source / name, output / name)
    (bits >> 16).astype("<u2").tofile(output / "token_embeddings.bf16")
    shutil.copyfile(Path(__file__).parent / "Licenses/Apache-2.0.txt", output / "LICENSE")
    (output / "MODEL_LICENSE.txt").write_text(
        "Source model: google/embeddinggemma-2, revision " + REVISION + "\n"
        "The source model card identifies the Apache License 2.0. See LICENSE and the unmodified README.md.\n"
        "Source model card: https://huggingface.co/google/embeddinggemma-2\n"
        "Converted text/vision assets use scaled RMSNorm and final L2 normalization, S128 and vision70.\n"
        "The vocabulary table is an exact BF16 copy. This bundle does not include audio.\n")
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                files[str(path.relative_to(output))] = hashlib.file_digest(handle, "sha256").hexdigest()
    (output / "embedding-manifest.json").write_text(json.dumps(
        {"format": 1, "revision": REVISION, "precision": "fp16-safe-s128-v70", "files": files}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    package(args.source, args.text, args.vision, args.reference, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
