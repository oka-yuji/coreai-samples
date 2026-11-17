import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from ...storage import read_json, write_json
from .checks import Checks, aggregate, fsq_summary, verdict
from .runtime import entries, validate_inputs
from .source import add_source_arguments, load_contract, load_model
from .wrappers import ASSETS, functions


class TorchComponents:
    def __init__(self, model, contract, asset):
        self.contract = contract
        self.modules = {(asset, row[0]): row[1:] for row in functions(model, asset, contract)}

    def state(self, asset, function):
        return {name: np.zeros(shape, np.float32) for name, _, shape in entries(self.contract, asset, function, "states", "fp32")}

    def call(self, asset, function, values, state=None):
        module, _, inputs, outputs, states = self.modules[asset, function]
        values = validate_inputs(self.contract, asset, function, values)
        with torch.no_grad():
            for name in states:
                getattr(module, name).copy_(torch.from_numpy(state[name]))
            result = module(*(torch.from_numpy(np.ascontiguousarray(values[name])) for name in inputs))
            for name in states:
                state[name][...] = getattr(module, name).numpy()
        result = result if isinstance(result, tuple) else (result,)
        return {name: value.numpy().copy() for name, value in zip(outputs, result, strict=True)}


def main():
    parser = argparse.ArgumentParser()
    add_source_arguments(parser)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = load_contract(args.contract)
    reference = read_json(args.reference / "reference.json")
    if not reference["complete"] or reference["contract"] != contract:
        raise ValueError("The reference is incomplete or was captured for another contract")
    model = load_model(args.snapshot, args.upstream, contract)
    args.output.mkdir(parents=True)
    report = {"complete": False, "passed": False}
    write_json(args.output / "wrapper-check.json", report)
    started = time.perf_counter()
    rows = []
    for asset in ASSETS:
        components = TorchComponents(model, contract, asset)
        for record in reference["records"]:
            rows.extend(Checks(components, contract, args.reference, record, np.float32).run([asset]))
    write_json(args.output / "rows.json", rows)
    report.update(complete=True, passed=verdict(rows, contract["criteria"]), seconds=time.perf_counter() - started,
                  summary=aggregate(rows), fsq=fsq_summary(rows, contract["criteria"]))
    write_json(args.output / "wrapper-check.json", report)
    print(json.dumps({"passed": report["passed"], "rows": len(rows)}), flush=True)
    if not report["passed"]:
        raise RuntimeError("The PyTorch wrappers differ from the original model")


if __name__ == "__main__":
    main()
