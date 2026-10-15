import argparse
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

from pinned import check_upstream, load_contract, pinned_hub, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    contract, snapshot = load_contract(args.contract, args.snapshot)
    upstream = args.upstream.resolve()
    revision = check_upstream(upstream, contract)
    destination = args.out / contract["output_name"]
    if destination.exists():
        raise FileExistsError(destination)
    sys.path.insert(0, str(upstream / "python/src"))
    with pinned_hub(snapshot, contract):
        import coreai_models
        if not Path(coreai_models.__file__).resolve().is_relative_to(upstream / "python/src"):
            raise ValueError("Exporter imported from an unexpected checkout")
        sys.argv = [
            "coreai.llm.export", contract["model_id"], "--platform", "macOS",
            "--compute-precision", contract["precision"], "--compression", contract["compression"],
            "--max-context-length", str(contract["context_length"]),
            "--output-name", contract["output_name"], "--output-dir", str(args.out),
        ]
        sys.argv.append("--dry-run")
        runpy.run_module("coreai_models.llm.export", run_name="__main__")
        sys.argv.pop()
        runpy.run_module("coreai_models.llm.export", run_name="__main__")
    metadata_path = destination / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["source"]["hf_model_id"] != contract["model_id"]:
        raise ValueError("Exported source identity changed")
    metadata["source"]["revision"] = contract["revision"]
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    tokenizer = destination / "tokenizer"
    if tokenizer.is_symlink() or not tokenizer.resolve().is_relative_to(destination.resolve()):
        raise ValueError("Unsafe tokenizer destination")
    shutil.rmtree(tokenizer)
    tokenizer.mkdir()
    assets = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
              "chat_template.jinja", "merges.txt", "vocab.json", "tokenizer.model", "spiece.model")
    hashes = {}
    for name in assets:
        source = snapshot / name
        if source.is_file():
            shutil.copyfile(source, tokenizer / name)
            hashes[name] = sha256(source)
            if sha256(tokenizer / name) != hashes[name]:
                raise ValueError(f"Tokenizer copy mismatch: {name}")
    (args.out / "provenance.json").write_text(json.dumps(
        {"contract": contract, "tokenizer_sha256": hashes, "upstream_revision": revision,
         "scope": "export only; numerical gate is a separate stage"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
