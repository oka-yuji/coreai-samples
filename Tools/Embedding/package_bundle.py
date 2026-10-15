import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for kind in ("text", "vision"):
        shutil.copytree(args.assets / f"{kind}-fp16-safe-v1" / f"{kind}.aimodel", args.output / f"{kind}.aimodel")
    for name in ("tokenizer.json", "tokenizer_config.json", "README.md"):
        shutil.copy2(args.assets / "source" / name, args.output / name)
    table = np.load(args.assets / "reference-v2/token_embeddings.npy", mmap_mode="r")
    if table.shape != (262144, 512) or table.dtype != np.float32:
        raise ValueError("Unexpected embedding table contract")
    bits = table.view(np.uint32)
    if np.any(bits & 0xFFFF) or not np.isfinite(table).all():
        raise ValueError("Table is not an exact BF16 expansion")
    (bits >> 16).astype("<u2").tofile(args.output / "token_embeddings.bf16")
    (args.output / "MODEL_LICENSE.txt").write_text(
        "EmbeddingGemma 2 is provided by Google under the Gemma Terms of Use.\n"
        "https://ai.google.dev/gemma/terms\n"
        "https://ai.google.dev/gemma/prohibited_use_policy\n"
        "The original model card is included as README.md.\n"
        "Source: google/embeddinggemma-2 revision 914f7f89142e33e77833254d9c9b90c3cef7303b\n"
        "Converted FP16 text/vision assets use scaled RMSNorm and final L2 normalization.\n"
        "Local validation bundle; no permission to redistribute is implied by packaging.\n")
    files = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                files[str(path.relative_to(args.output))] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {"format": 1, "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
                "precision": "fp16-safe-s128-v70", "files": files}
    (args.output / "embedding-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
