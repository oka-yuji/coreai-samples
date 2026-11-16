import asyncio
from pathlib import Path
import time

import numpy as np

from ...storage import read_json

FLOATS = {"fp32": "float32", "fp16": "float16"}


def entries(contract, asset, function, kind, precision):
    return [(name, FLOATS[precision] if dtype == "float32" else dtype, list(shape))
            for name, dtype, shape in contract["assets"][asset]["functions"][function][kind]]


def validate_inputs(contract, asset, function, values, precision="fp32"):
    expected = entries(contract, asset, function, "inputs", precision)
    if set(values) != {name for name, _, _ in expected}:
        raise ValueError(f"Inputs must be {[name for name, _, _ in expected]}")
    values = {name: np.asarray(values[name]) for name, _, _ in expected}
    for name, dtype, shape in expected:
        value = values[name]
        if value.dtype != np.dtype(dtype) or list(value.shape) != shape:
            raise ValueError(f"{asset}.{function}.{name} must be {dtype} {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{asset}.{function}.{name} must be finite")
    execution = contract["execution"]
    limits = {"length": (1, execution["prefill_length"]), "position": (0, execution["capacity"] - 1),
              "ids": (0, contract["architecture"]["vocab_size"] - 1)}
    for name, (low, high) in limits.items():
        if name in values and (np.any(values[name] < low) or np.any(values[name] > high)):
            raise ValueError(f"{asset}.{function}.{name} must be within {low}..{high}")
    return values


def describe(function):
    descriptor = function.desc

    def rows(names, lookup):
        return [(name, str(lookup(name).dtype), [int(size) for size in lookup(name).shape]) for name in names]

    return {"inputs": rows(descriptor.input_names, descriptor.input_descriptor),
            "outputs": rows(descriptor.output_names, descriptor.output_descriptor),
            "states": rows(descriptor.state_names, descriptor.state_descriptor)}


class Components:
    def __init__(self, directories, precision, contract):
        from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions

        if not SpecializationOptions.is_supported():
            raise RuntimeError("Use the macOS Core AI runtime")
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        self.runner = asyncio.Runner()
        self.contract, self.precision = contract, precision
        self.models, self.functions, self.exports, self.load_seconds = {}, {}, {}, {}
        try:
            for directory in map(Path, directories):
                export = read_json(directory / "export.json")
                asset = export["asset"]
                if not export["complete"] or export["contract"] != contract or asset in self.exports:
                    raise ValueError("Unexpected, duplicate or mismatched component")
                self.exports[asset] = export
                started = time.perf_counter()
                self.models[asset] = self.runner.run(AIModel.load(directory / export["precisions"][precision]["file"],
                                                                  specialization_options=options))
                for function in contract["assets"][asset]["functions"]:
                    loaded = self.models[asset].load_function(function)
                    observed = describe(loaded)
                    if observed != {kind: entries(contract, asset, function, kind, precision) for kind in observed}:
                        raise ValueError(f"{asset}.{function} differs from the contract: {observed}")
                    self.functions[asset, function] = loaded
                self.load_seconds[asset] = time.perf_counter() - started
        except BaseException:
            self.close()
            raise

    def state(self, asset, function):
        from coreai.runtime import NDArray

        return {name: NDArray(np.zeros(shape, dtype=np.dtype(dtype)))
                for name, dtype, shape in entries(self.contract, asset, function, "states", self.precision)}

    def call(self, asset, function, values, state=None):
        from coreai.runtime import NDArray

        values = validate_inputs(self.contract, asset, function, values, self.precision)
        inputs = {name: NDArray(np.ascontiguousarray(value)) for name, value in values.items()}
        result = self.runner.run(self.functions[asset, function](inputs=inputs, state=state or {}))
        return {name: np.asarray(result[name].numpy()).copy()
                for name, _, _ in entries(self.contract, asset, function, "outputs", self.precision)}

    def close(self):
        self.functions.clear()
        self.models.clear()
        self.runner.close()
