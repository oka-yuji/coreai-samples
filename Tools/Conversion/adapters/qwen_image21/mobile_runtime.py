import asyncio
import gc
import os
from pathlib import Path
import time

import numpy as np

from .contracts import require_heavy_job, validate_arrays
from .mobile_artifacts import read_component


class CandidateComponent:
    def __init__(self, directory, profile, kind):
        require_heavy_job()
        self.directory = Path(directory)
        self.contract, _ = read_component(self.directory, profile, kind)
        self.runner = asyncio.Runner()
        self.model = self.function = None
        self.calls = []
        started = time.perf_counter()
        try:
            self.model = self.runner.run(self.load())
            self.function = self.model.load_function("main")
            if self.function.desc.state_names:
                raise ValueError("A mobile candidate must be stateless")
        except BaseException:
            self.close()
            raise
        self.load_seconds = time.perf_counter() - started

    async def load(self):
        from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions

        if os.environ.get("USE_LOCAL_COREAI") or not SpecializationOptions.is_supported():
            raise RuntimeError("Use the macOS Core AI runtime for candidate validation")
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        return await AIModel.load(self.directory / self.contract["asset"], specialization_options=options)

    def call(self, values):
        from coreai.runtime import NDArray

        validate_arrays(values, self.contract["inputs"])
        inputs = {name: NDArray(np.ascontiguousarray(value)) for name, value in values.items()}
        started = time.perf_counter()
        outputs = self.runner.run(self.function(inputs=inputs, state={}))
        self.calls.append(time.perf_counter() - started)
        expected = self.contract["outputs"]
        if set(outputs) != {item["name"] for item in expected}:
            raise ValueError("Core AI returned different physical output names")
        arrays = tuple(np.asarray(outputs[item["name"]].numpy()).copy() for item in expected)
        for value, item in zip(arrays, expected, strict=True):
            if list(value.shape) != item["shape"] or value.dtype.name != item["dtype"]:
                raise ValueError("Core AI returned a different physical output shape or dtype")
        return arrays

    def close(self):
        self.function = self.model = None
        self.runner.close()
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
