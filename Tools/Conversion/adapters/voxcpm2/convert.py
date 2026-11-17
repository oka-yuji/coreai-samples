import argparse
import json
from pathlib import Path
import time

import torch

from ...storage import file_hash, write_json
from .source import add_source_arguments, load_contract, load_model, with_capacity
from .wrappers import ASSETS, functions, keep_fp32


def stored_bytes(rows):
    tensors = {}
    for _, module, _, _, _, states in rows:
        tensors.update({id(value): value for value in module.parameters()})
        tensors.update({id(value): value for name, value in module.named_buffers() if name not in states})
    return sum(value.numel() * value.element_size() for value in tensors.values())


def export(rows, precision, contract, asset, path):
    import coreai_torch
    from coreai.runtime import AIModelAssetMetadata
    from torch.export.graph_signature import InputKind

    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    report = []
    for name, module, example, inputs, outputs, states in rows:
        started = time.perf_counter()
        with torch.no_grad():
            program = torch.export.export(module.eval(), example, strict=False)
        program = program.run_decompositions(coreai_torch.get_decomp_table())
        signature = program.graph_signature
        mutated = set(signature.buffers_to_mutate.values())
        order = [spec.target for spec in signature.input_specs if spec.kind == InputKind.BUFFER and spec.target in mutated]
        if order != list(states):
            raise ValueError(f"State order differs from the contract: {asset}.{name} {order}")
        unused = [node.name for node in program.graph.nodes
                  if node.op == "placeholder" and node.name in signature.user_inputs and not node.users]
        if unused or len({id(value) for value in example}) != len(example):
            raise ValueError(f"The exported graph ignores or aliases inputs: {asset}.{name} {unused}")
        kept = 0
        if precision == "fp16":
            from coreai_opt.casting import cast_to_16_bit_precision

            kept = sum(node.op == "call_function" and keep_fp32(node) for node in program.graph.nodes)
            program = cast_to_16_bit_precision(program, ignored_ops=keep_fp32)
        converter.add_exported_program(program, input_names=inputs, output_names=outputs, state_names=states,
                                       entrypoint_name=name)
        report.append({"function": name, "export_seconds": time.perf_counter() - started, "kept_fp32_nodes": kept,
                       "aten_ops": sorted({str(node.target) for node in program.graph.nodes if node.op == "call_function"})})
    started = time.perf_counter()
    program = converter.to_coreai()
    metadata = AIModelAssetMetadata()
    metadata.author = "OpenBMB (VoxCPM2)"
    metadata.license = "Apache-2.0"
    metadata.model_description = f"{contract['model_id']} {asset} {precision}; macOS Core AI research conversion"
    metadata.set_custom("source_revision", contract["revision"])
    metadata.set_custom("upstream_revision", contract["upstream_revision"])
    metadata.set_custom("capacity", contract["execution"]["capacity"])
    program.save_asset(path, metadata)
    files = {str(item.relative_to(path)): item.stat().st_size for item in sorted(path.rglob("*")) if item.is_file()}
    return {"file": path.name, "functions": report, "to_coreai_and_save_seconds": time.perf_counter() - started,
            "bytes": sum(files.values()), "files": files, "main_mlirb_sha256": file_hash(path / "main.mlirb")["sha256"]}


def main():
    parser = argparse.ArgumentParser()
    add_source_arguments(parser)
    parser.add_argument("--asset", choices=ASSETS, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], action="append", required=True)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = load_contract(args.contract)
    contract = with_capacity(base, args.capacity) if args.capacity else base
    model = load_model(args.snapshot, args.upstream, base)
    args.output.mkdir(parents=True)
    report = {"complete": False, "asset": args.asset, "contract": contract}
    write_json(args.output / "export.json", report)
    specs = functions(model, args.asset, contract)
    report["parameter_bytes_fp32"] = stored_bytes(specs)
    report["precisions"] = {}
    for precision in args.precision:
        report["precisions"][precision] = export(specs, precision, contract, args.asset,
                                                 args.output / f"{args.asset}-{precision}.aimodel")
        write_json(args.output / "export.json", report)
    report["complete"] = True
    write_json(args.output / "export.json", report)
    print(json.dumps({"asset": args.asset, **{p: v["bytes"] for p, v in report["precisions"].items()}}), flush=True)


if __name__ == "__main__":
    main()
