import hashlib
import json
from pathlib import Path

from .evaluation import IMPLEMENTATION, MODEL, REVISION, SIGMAS


def boundaries(joint=2304, floating="float32"):
    def tensor(name, shape, dtype=floating):
        return {"name": name, "shape": shape, "dtype": dtype, "layout": "row-major"}
    return {
        "vision": {"inputs": [tensor("pixel_values", [1024, 1536])],
                   "outputs": [tensor(name, [256, 4096]) for name in
                               ("merged", "deepstack_0", "deepstack_1", "deepstack_2")]},
        "text": {"inputs": [
            tensor("input_ids", [1, 512], "int32"), tensor("vision", [1, 512, 4096]),
            tensor("deepstack", [3, 1, 512, 4096]), tensor("cosine", [1, 512, 128]),
            tensor("sine", [1, 512, 128]), tensor("mask", [1, 1, 512, 512], "bool"),
            tensor("visual_mask", [1, 512], "bool")], "outputs": [tensor("hidden", [1, 512, 4096])]},
        "denoiser": {"inputs": [
            tensor("latents", [1, 2048 if joint == 2304 else 1024, 64]), tensor("text", [1, 512, 4096]),
            tensor("timestep", [1]), tensor("text_index", [joint], "int32"),
            tensor("image_index", [joint], "int32"), tensor("image_mask", [joint], "bool"),
            tensor("cosine", [1, joint, 1, 64]), tensor("sine", [1, joint, 1, 64]),
            tensor("mask", [1, 1, joint, joint], "bool"), tensor("target", [joint], "bool")],
            "outputs": [tensor("noise", [1, joint, 64])]},
        "vae_encoder": {"inputs": [tensor("value", [1, 4, 1, 512, 512])],
                        "outputs": [tensor("result", [1, 64, 1, 32, 32])]},
        "vae_decoder": {"inputs": [tensor("value", [1, 64, 1, 32, 32])],
                        "outputs": [tensor("result", [1, 4, 1, 512, 512])]},
    }


def baseline_profile(bundle, dependency_lock):
    bundle = Path(bundle)
    components = {}
    converters = None
    for kind, specification in boundaries().items():
        contract = json.loads((bundle / kind / "component.json").read_text())
        if contract.get("profile", {}).get("revision") != REVISION or contract["profile"]["precision"] != "fp32":
            raise ValueError("The baseline must be the pinned existing FP32 bundle")
        if contract["inputs"] != [{key: value for key, value in item.items() if key != "layout"}
                                 for item in specification["inputs"]]:
            raise ValueError("Baseline physical input contract differs")
        if contract["output_names"] != [item["name"] for item in specification["outputs"]]:
            raise ValueError("Baseline output contract differs")
        if converters is not None and converters != contract["export_code_sha256"]:
            raise ValueError("Baseline components have different export sources")
        converters = contract["export_code_sha256"]
        components[kind] = {"asset": f"{kind}/{kind}.aimodel", "function": "main", "state_names": [],
                            "compute_dtype": "float32", "weight_encoding": {"kind": "dense", "dtype": "float32"},
                            **specification}
    return {
        "format": 2, "id": "qmi-fp32-reference-v2",
        "source": {"model": MODEL, "revision": REVISION, "implementation_revision": IMPLEMENTATION,
                   "dependency_lock_sha256": hashlib.sha256(Path(dependency_lock).read_bytes()).hexdigest(),
                   "converter_sha256": converters},
        "workload": {"resolution": [512, 512], "reference_resolution": [512, 512], "batch": 1,
                     "maximum_references": 1, "processed_text_capacity": 512, "joint_capacity": 2304,
                     "operation": "generate-and-edit", "cfg": 1, "sigmas": list(SIGMAS),
                     "terminal_sigma": 0.0, "scheduler_compute_dtype": "float32",
                     "scheduler_output_dtype": "float32", "cast_back": "every-step-to-model-output",
                     "scheduler_delta_dtype": "model-output", "scheduler_product_dtype": "model-output",
                     "model_timestep_rule": "sigma-f32-scale1000-cast-divide1000",
                     "rounding": "nearest-ties-to-even", "prefix_kv_cache": False,
                     "noise_shape": [1, 1024, 64], "noise_dtype": "float32",
                     "rng_version": "torch-2.9.0-cpu-randn-packed-v1",
                     "denoiser_fp16_clip": [-65504, 65504], "zero_centered_norm_dtype": "float32"},
        "components": components,
    }
