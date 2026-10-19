import asyncio
import gc
from pathlib import Path
import time

import numpy as np

from .contracts import KINDS, check_profile, read_json, require_heavy_job, validate_arrays


class Component:
    def __init__(self, directory):
        require_heavy_job()
        self.directory = Path(directory)
        self.contract = read_json(self.directory / "component.json")
        check_profile(self.contract["profile"])
        kind = self.contract["kind"]
        if kind not in KINDS or self.contract["asset"] != f"{kind}.aimodel" or self.contract["passed"] is not True:
            raise ValueError("The component contract is invalid or unverified")
        self.runner = asyncio.Runner()
        self.model = self.function = None
        self.calls = []
        started = time.perf_counter()
        try:
            self.model = self.runner.run(self._load())
            self.function = self.model.load_function("main")
            if self.function.desc.state_names:
                raise ValueError("The component must be stateless")
        except BaseException:
            self.close()
            raise
        self.load_seconds = time.perf_counter() - started

    async def _load(self):
        import os
        from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions

        if os.environ.get("USE_LOCAL_COREAI") or not SpecializationOptions.is_supported():
            raise RuntimeError("Use the macOS Core AI runtime")
        options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
        return await AIModel.load(self.directory / self.contract["asset"], specialization_options=options)

    def call(self, values):
        from coreai.runtime import NDArray

        validate_arrays(values, self.contract["inputs"])
        inputs = {key: NDArray(np.ascontiguousarray(value)) for key, value in values.items()}
        started = time.perf_counter()
        output = self.runner.run(self.function(inputs=inputs, state={}))
        arrays = tuple(np.asarray(output[key].numpy()).copy() for key in self.contract["output_names"])
        self.calls.append(time.perf_counter() - started)
        if any(not np.isfinite(value).all() for value in arrays):
            raise ValueError("Core AI returned non-finite output")
        return arrays

    def close(self):
        self.function = self.model = None
        self.runner.close()
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
