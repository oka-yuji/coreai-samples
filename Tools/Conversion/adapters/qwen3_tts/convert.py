import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ...storage import contained, write_json
from .inputs import prepare_inputs
from .source import add_source_arguments, compare_tensor, model_from_arguments
from .wrappers import dynamic_shapes, make_wrapper, names


def load_record(root, record, category):
    directory = contained(Path(root), record["directory"])
    return tuple(np.load(directory / f"{category}-{entry['name']}.npy", allow_pickle=False) for entry in record[category])


def outputs(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def main():
    parser = argparse.ArgumentParser()
    add_source_arguments(parser)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--kind", choices=["projection", "talker", "predictor", "codec"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = json.loads((args.reference / "reference.json").read_text())
    if not reference["complete"]:
        raise ValueError("The reference is incomplete")
    records = [item for item in reference["records"] if item["kind"] == args.kind]
    if not records:
        raise ValueError("The component has no reference records")
    wrapper, contract = model_from_arguments(args)
    if contract != reference["contract"]:
        raise ValueError("Reference and source identities differ")
    module = make_wrapper(wrapper.model, args.kind)
    args.output.mkdir(parents=True)
    comparisons = []
    with torch.inference_mode():
        for record in records:
            inputs = tuple(torch.from_numpy(value.copy()) for value in prepare_inputs(args.kind, load_record(args.reference, record, "inputs")))
            actual = outputs(module(*inputs))
            expected = load_record(args.reference, record, "outputs")
            measured = [compare_tensor(wanted, observed.detach().cpu().numpy(), contract["normalized_max_limit"])
                        for wanted, observed in zip(expected, actual, strict=True)]
            comparisons.append({"id": record["id"], "passed": all(item["passed"] for item in measured), "outputs": measured})
    write_json(args.output / "wrapper-comparison.json", comparisons)
    if not all(row["passed"] for row in comparisons):
        raise RuntimeError("The wrapper differs from the original model")
    record = next((item for item in records if item["inputs"][0]["shape"][-2 if args.kind != "codec" else -1] > 2), records[0])
    example = tuple(torch.from_numpy(value.copy()) for value in prepare_inputs(args.kind, load_record(args.reference, record, "inputs")))
    import coreai_torch
    from coreai.runtime import AIModelAssetMetadata

    started = time.perf_counter()
    with torch.no_grad():
        exported = torch.export.export(module, example, dynamic_shapes=dynamic_shapes(args.kind), strict=False)
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    input_names, output_names = names(args.kind, padded=True)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(exported, input_names=input_names, output_names=output_names,
                                   state_names=[], entrypoint_name="main")
    program = converter.to_coreai()
    metadata = AIModelAssetMetadata()
    metadata.author = "Qwen Team, Alibaba Group; converted by Yuji Oka"
    metadata.license = "Apache-2.0"
    metadata.model_description = f"Qwen3-TTS 0.6B CustomVoice {args.kind}; FP32 research profile"
    metadata.set_custom("source_revision", contract["revision"])
    asset = args.output / f"{args.kind}.aimodel"
    program.save_asset(asset, metadata)
    result = {"complete": True, "kind": args.kind, "asset": asset.name, "contract": contract,
              "seconds": time.perf_counter() - started, "input_names": input_names, "output_names": output_names,
              "state_names": [], "reference_sha256": hashlib.sha256((args.reference / "reference.json").read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("source.py", "wrappers.py", "codec.py", "inputs.py", "convert.py")}}
    write_json(args.output / "export.json", result)
    print(json.dumps({"kind": args.kind, "seconds": result["seconds"], "complete": True}), flush=True)


if __name__ == "__main__":
    main()
