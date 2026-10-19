import asyncio
import json
from pathlib import Path
import time

import numpy as np

from .wrappers import names


def validate_inputs(kind, values, vocabulary_size=None):
    values = tuple(np.asarray(value) for value in values)
    input_names, _ = names(kind)
    if len(values) != len(input_names) or any(not np.isfinite(value).all() for value in values):
        raise ValueError("Inputs must have the expected count and contain finite values")
    integer_fields = {"encoder": {0}, "decoder": set(), "local": {1}, "codec": {1}}[kind]
    for index, value in enumerate(values):
        expected_dtype = np.dtype("int32" if index in integer_fields else "float32")
        if value.dtype != expected_dtype:
            raise ValueError(f"Unexpected dtype for {input_names[index]}: {value.dtype}")
    if kind == "encoder":
        tokens, mask = values
        if tokens.ndim != 2 or tokens.shape[0] != 1 or not 2 <= tokens.shape[1] <= 512 or mask.shape != tokens.shape:
            raise ValueError("Encoder inputs must be [1, text] with 2 to 512 tokens")
        if vocabulary_size is None or np.any(tokens < 0) or np.any(tokens >= vocabulary_size):
            raise ValueError("A text token is outside the pinned vocabulary")
        masks = [mask]
    elif kind == "decoder":
        hidden, mask, memory, memory_mask, prior = values
        if hidden.ndim != 3 or hidden.shape[0] != 2 or hidden.shape[2] != 768 or not 2 <= hidden.shape[1] <= 512:
            raise ValueError("Decoder input must be [2, sequence, 768] with 2 to 512 positions")
        if memory.ndim != 3 or memory.shape[0] != 2 or memory.shape[2] != 768 or not 2 <= memory.shape[1] <= 512:
            raise ValueError("Decoder memory must be [2, text, 768] with 2 to 512 positions")
        if mask.shape != hidden.shape[:2] or memory_mask.shape != memory.shape[:2] or prior.shape != (2, 1, memory.shape[1]):
            raise ValueError("Decoder masks and prior must match their sequence dimensions")
        if np.any(prior <= 0):
            raise ValueError("The attention prior must be positive")
        masks = [mask, memory_mask]
    elif kind == "local":
        hidden, codebook = values
        if hidden.ndim != 3 or hidden.shape[0] != 2 or hidden.shape[2] != 768 or not 1 <= hidden.shape[1] <= 16:
            raise ValueError("Local input must be [2, sequence, 768] with 1 to 16 positions")
        if codebook.shape != (1,) or not 0 <= int(codebook[0]) < 16:
            raise ValueError("The codebook index must be between 0 and 15")
        masks = []
    else:
        latent, lengths = values
        if latent.ndim != 3 or latent.shape[:2] != (1, 32) or not 2 <= latent.shape[2] <= 512:
            raise ValueError("Codec input must be [1, 32, frames] with 2 to 512 frames")
        if lengths.shape != (1,) or not 1 <= int(lengths[0]) <= latent.shape[2]:
            raise ValueError("Codec length is outside the provided latent sequence")
        masks = []
    if any(np.any((mask != 0) & (mask != 1)) or np.any(np.sum(mask, axis=1) == 0) for mask in masks):
        raise ValueError("Masks must be binary and include at least one position")
    return values


class CoreAIComponents:
    def __init__(self, directories, contract):
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
                if not export["complete"] or export["contract"] != contract or kind in self.exports:
                    raise ValueError("The component set has incompatible or duplicate exports")
                if kind not in ("encoder", "decoder", "local", "codec"):
                    raise ValueError("Unknown component")
                self.exports[kind] = export
                self.models[kind] = self.runner.run(self._load(directory / export["asset"]))
                self.functions[kind] = self.models[kind].load_function("main")
                if self.functions[kind].desc.state_names:
                    raise ValueError("The research profile requires stateless functions")
                self.metrics[kind] = {"calls": 0, "seconds": 0.0}
            if set(self.exports) != {"encoder", "decoder", "local", "codec"}:
                raise ValueError("All four components are required")
        except BaseException:
            self.close()
            raise

    async def _load(self, path):
        from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions

        if not SpecializationOptions.is_supported():
            raise RuntimeError("This profile requires the macOS Core AI runtime")
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        return await AIModel.load(path, specialization_options=options)

    def call(self, kind, values):
        from coreai.runtime import NDArray

        export = self.exports[kind]
        values = validate_inputs(kind, values, export.get("vocabulary_size"))
        inputs = {name: NDArray(np.ascontiguousarray(value))
                  for name, value in zip(export["input_names"], values, strict=True)}
        started = time.perf_counter()
        result = self.runner.run(self.functions[kind](inputs=inputs, state={}))
        self.metrics[kind]["calls"] += 1
        self.metrics[kind]["seconds"] += time.perf_counter() - started
        outputs = tuple(np.asarray(result[name].numpy()).copy() for name in export["output_names"])
        if any(not np.isfinite(value).all() for value in outputs):
            raise ValueError(f"Core AI returned non-finite {kind} output")
        return outputs

    def close(self):
        self.functions.clear()
        self.models.clear()
        self.runner.close()
