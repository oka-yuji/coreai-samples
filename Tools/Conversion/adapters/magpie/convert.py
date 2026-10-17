import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ...storage import contained, write_json
from .artifacts import compare_tensor
from .source import add_source_arguments, model_from_arguments
from .wrappers import dynamic_shapes, make_wrapper, names


def load_record(root, record, category):
    directory = contained(Path(root), record["directory"])
    return tuple(np.load(directory / f"{category}-{item['name']}.npy", allow_pickle=False)
                 for item in record[category])


def tensors(values):
    return tuple(torch.from_numpy(value.copy()) for value in values)


def as_outputs(value):
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


def main():
    parser = argparse.ArgumentParser()
    add_source_arguments(parser)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--kind", choices=["encoder", "decoder", "local", "codec"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = json.loads((args.reference / "reference.json").read_text())
    if not reference["complete"]:
        raise ValueError("The reference is incomplete")
    records = [item for item in reference["records"] if item["kind"] == args.kind]
    if not records:
        raise ValueError("No reference records for this component")
    model, contract = model_from_arguments(args)
    if contract != reference["contract"]:
        raise ValueError("The source and reference contracts differ")
    module = make_wrapper(model, args.kind)
    from nemo.core.classes.common import typecheck

    typecheck.set_typecheck_enabled(False)
    args.output.mkdir(parents=True)
    comparisons = []
    with torch.inference_mode():
        for record in records:
            inputs = tensors(load_record(args.reference, record, "inputs"))
            expected = load_record(args.reference, record, "outputs")
            actual = as_outputs(module(*inputs))
            measured = [compare_tensor(wanted, observed.detach().cpu().numpy(), contract["normalized_max_limit"])
                        for wanted, observed in zip(expected, actual, strict=True)]
            comparisons.append({"id": record["id"], "outputs": measured,
                                "passed": all(item["passed"] for item in measured)})
    write_json(args.output / "wrapper-comparison.json", comparisons)
    if not all(item["passed"] for item in comparisons):
        raise RuntimeError("The wrapper differs from the original NeMo reference")
    example_record = next((item for item in records if args.kind != "local" or item["inputs"][0]["shape"][1] == 4), records[0])
    example = tensors(load_record(args.reference, example_record, "inputs"))
    import coreai_torch
    from coreai.runtime import AIModelAssetMetadata

    started = time.perf_counter()
    with torch.no_grad():
        exported = torch.export.export(module, example, dynamic_shapes=dynamic_shapes(args.kind), strict=False)
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    input_names, output_names = names(args.kind)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(exported, input_names=input_names, output_names=output_names,
                                   state_names=[], entrypoint_name="main")
    program = converter.to_coreai()
    metadata = AIModelAssetMetadata()
    metadata.author = "NVIDIA; converted by Yuji Oka"
    metadata.license = "NVIDIA Open Model License"
    metadata.model_description = f"Magpie TTS v2607 {args.kind}; FP32 research profile"
    metadata.set_custom("source_revision", contract["model"]["revision"])
    metadata.set_custom("codec_revision", contract["codec"]["revision"])
    package = args.output / f"{args.kind}.aimodel"
    program.save_asset(package, metadata)
    record = {"complete": True, "kind": args.kind, "asset": package.name,
              "seconds": time.perf_counter() - started, "contract": contract,
              "input_names": input_names, "output_names": output_names,
              "example": example_record["id"], "state_names": [],
              "vocabulary_size": model.text_embedding.num_embeddings,
              "reference_sha256": hashlib.sha256((args.reference / "reference.json").read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("source.py", "wrappers.py", "codec.py", "convert.py")}}
    write_json(args.output / "export.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
