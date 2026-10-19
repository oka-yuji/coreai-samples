import asyncio
import json
from pathlib import Path
import time

import numpy as np

from .wrappers import names
from .inputs import EXECUTION_SHAPES, prepare_inputs
from .profiles import model_profile


def validate_inputs(kind, values, *, talker_width=1024):
    if talker_width not in (1024, 2048):
        raise ValueError("Unsupported CustomVoice input width")
    values = tuple(np.asarray(value) for value in values)
    fields, _ = names(kind)
    if len(values) != len(fields) or any(not np.isfinite(value).all() for value in values):
        raise ValueError("Inputs must be finite and match the component arity")
    if kind == "projection":
        hidden, = values
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != 2048 or not 1 <= hidden.shape[1] <= 512:
            raise ValueError("Projection input must be [1, sequence, 2048], length 1 to 512")
    elif kind == "talker":
        hidden, positions = values
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != talker_width or not 1 <= hidden.shape[1] <= 512:
            raise ValueError(f"Talker input must be [1, sequence, {talker_width}], length 1 to 512")
        if positions.shape != (3, 1, hidden.shape[1]) or positions.dtype != np.int32 or np.any(positions < 0) or np.any(positions >= 32768):
            raise ValueError("The three position axes must match the talker sequence and supported range")
    elif kind == "predictor":
        hidden, codebook = values
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != talker_width or not 2 <= hidden.shape[1] <= 16:
            raise ValueError(f"Predictor input must be [1, sequence, {talker_width}], length 2 to 16")
        if codebook.shape != (1,) or codebook.dtype != np.int32 or not 0 <= int(codebook[0]) < 15:
            raise ValueError("Predictor head index must be between 0 and 14")
        if int(codebook[0]) != hidden.shape[1] - 2:
            raise ValueError("Predictor prefix length and head index disagree")
    else:
        codes, = values
        if codes.ndim != 3 or codes.shape[:2] != (1, 16) or not 1 <= codes.shape[2] <= 256 or codes.dtype != np.int32:
            raise ValueError("Codec input must be int32 [1, 16, frames], length 1 to 256")
        if np.any(codes < 0) or np.any(codes >= 2048):
            raise ValueError("An audio code is outside the tokenizer vocabulary")
    if kind != "codec" and values[0].dtype != np.float32:
        raise ValueError("The research profile requires float32 hidden states")
    return values


class CoreAIComponents:
    def __init__(self, directories, contract):
        _, self.talker_width = model_profile(contract)
        if contract.get("execution_shapes") != EXECUTION_SHAPES:
            raise ValueError("This runtime requires the fixed decoder shape profile")
        self.runner = asyncio.Runner()
        self.models = {}
        self.functions = {}
        self.exports = {}
        self.metrics = {}
        try:
            for directory in directories:
                directory = Path(directory)
                export = json.loads((directory / "export.json").read_text())
                kind = export["kind"]
                if kind not in {"projection", "talker", "predictor", "codec"} or kind in self.exports:
                    raise ValueError("Unexpected or duplicate component")
                if not export["complete"] or export["contract"] != contract:
                    raise ValueError("Component source contracts do not match")
                self.exports[kind] = export
                self.models[kind] = self.runner.run(self._load(directory / export["asset"]))
                self.functions[kind] = self.models[kind].load_function("main")
                if self.functions[kind].desc.state_names:
                    raise ValueError("This profile requires stateless functions")
                self.metrics[kind] = {"calls": 0, "seconds": 0.0, "input_shapes": []}
            if set(self.exports) != {"projection", "talker", "predictor", "codec"}:
                raise ValueError("All four Core AI components are required")
        except BaseException:
            self.close()
            raise

    async def _load(self, path):
        from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions

        if not SpecializationOptions.is_supported():
            raise RuntimeError("Use the macOS Core AI runtime")
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        return await AIModel.load(path, specialization_options=options)

    def call(self, kind, values):
        from coreai.runtime import NDArray

        values = prepare_inputs(kind, validate_inputs(kind, values, talker_width=self.talker_width))
        shapes = [list(value.shape) for value in values]
        if shapes not in self.metrics[kind]["input_shapes"]:
            self.metrics[kind]["input_shapes"].append(shapes)
        export = self.exports[kind]
        inputs = {name: NDArray(np.ascontiguousarray(value)) for name, value in zip(export["input_names"], values, strict=True)}
        started = time.perf_counter()
        result = self.runner.run(self.functions[kind](inputs=inputs, state={}))
        self.metrics[kind]["calls"] += 1
        self.metrics[kind]["seconds"] += time.perf_counter() - started
        output = tuple(np.asarray(result[name].numpy()).copy() for name in export["output_names"])
        if any(not np.isfinite(value).all() for value in output):
            raise ValueError(f"Non-finite Core AI output: {kind}")
        return output

    def close(self):
        self.functions.clear()
        self.models.clear()
        self.runner.close()
