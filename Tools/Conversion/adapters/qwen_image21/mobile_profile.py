from copy import deepcopy

from .evaluation import IMPLEMENTATION, MODEL, REVISION, SIGMAS, canonical_hash, validate_profile
from .profile import boundaries


NATIVE_COMPLEX_CANDIDATE = "w4-dit-g16-native-complex-blocks16-23-f32"
CANDIDATES = ("fp16", "bf16", "mixed", "w8", "w4", "mixed-f32", "w4-f32", "w4-blocks20-21-f32", "w4-blocks16-23-f32",
              "w4-dit-g16-blocks16-23-f32", NATIVE_COMPLEX_CANDIDATE)


def comparison_candidate(profile):
    return "mixed-f32" if profile["id"] in {f"qmi-mobile-{name}-v1" for name in CANDIDATES if name.endswith("-f32")} else "mixed"


def component_settings(candidate):
    if candidate not in CANDIDATES:
        raise ValueError("Select a declared mobile candidate")
    full_precision = candidate.endswith("-f32")
    family = candidate.removesuffix("-f32")
    if candidate in ("fp16", "bf16"):
        precision = "float16" if candidate == "fp16" else "bfloat16"
        result = {kind: {"compute_dtype": precision, "weight_encoding": {"kind": "dense", "dtype": precision}}
                  for kind in boundaries()}
        result["denoiser"]["weight_encoding"]["time_frequency_dtype"] = "float32"
        return result
    result = {}
    for kind in boundaries():
        if kind.startswith("vae_"):
            result[kind] = {"compute_dtype": "float32", "weight_encoding": {"kind": "dense", "dtype": "float32"}}
        elif family == "mixed":
            result[kind] = {
                "compute_dtype": "float16" if kind == "denoiser" and not full_precision else "float32",
                "weight_encoding": {"kind": "dense" if kind == "denoiser" and not full_precision else "dense-linear", "dtype": "float16",
                                    "embedding_dtype": "float16" if kind == "text" else "float32",
                                    "other_parameters_dtype": "float16" if kind == "denoiser" and not full_precision else "float32"},
            }
        else:
            result[kind] = {
                "compute_dtype": "float16" if kind == "denoiser" and not full_precision else "float32",
                "weight_encoding": {"kind": "linear-integer", "bits": 8 if kind == "vision" or candidate == "w8" else 4,
                                    "group_size": 16 if kind == "vision" else 64, "axis": 1, "scheme": "symmetric_with_clipping",
                                    "storage_dtype": "float16", "effective_weight": "round16((q-zp)*scale16)",
                                    "embedding_dtype": "float16" if kind == "text" else "float32",
                                    "other_parameters_dtype": "float16" if kind == "denoiser" and not full_precision else "float32"},
            }
    result["denoiser"]["weight_encoding"]["time_frequency_dtype"] = "float32"
    if candidate in ("w4-dit-g16-blocks16-23-f32", NATIVE_COMPLEX_CANDIDATE):
        result["denoiser"]["weight_encoding"]["group_size"] = 16
    preserved = {"w4-blocks20-21-f32": (20, 21), "w4-blocks16-23-f32": tuple(range(16, 24)),
                 "w4-dit-g16-blocks16-23-f32": tuple(range(16, 24)),
                 NATIVE_COMPLEX_CANDIDATE: tuple(range(16, 24))}
    if candidate in preserved:
        suffixes = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.proj", "img_mlp.out", "img_mlp.gate_layer")
        result["denoiser"]["weight_encoding"]["dense_modules"] = sorted(
            f"transformer_blocks.{block}.{suffix}.linear" for block in preserved[candidate] for suffix in suffixes)
    return result


def make_profile(candidate, dependency_lock_sha256, converter_sha256):
    settings = component_settings(candidate)
    components = {
        kind: {"asset": f"{kind}/{kind}.aimodel", "function": "main", "state_names": [],
               **settings[kind], **specification}
        for kind, specification in boundaries().items()
    }
    profile = {
        "format": 2, "id": f"qmi-mobile-{candidate}-v1",
        "source": {"model": MODEL, "revision": REVISION, "implementation_revision": IMPLEMENTATION,
                   "dependency_lock_sha256": dependency_lock_sha256, "converter_sha256": deepcopy(converter_sha256)},
        "workload": {"resolution": [512, 512], "reference_resolution": [512, 512], "batch": 1,
                     "maximum_references": 1, "processed_text_capacity": 512, "joint_capacity": 2304,
                     "operation": "generate-and-edit", "cfg": 1, "sigmas": list(SIGMAS),
                     "terminal_sigma": 0.0, "scheduler_compute_dtype": "float32",
                     "scheduler_output_dtype": settings["denoiser"]["compute_dtype"],
                     "cast_back": "every-step-to-model-output", "scheduler_delta_dtype": "model-output",
                     "scheduler_product_dtype": "model-output",
                     "model_timestep_rule": "sigma-f32-scale1000-cast-divide1000",
                     "rounding": "nearest-ties-to-even", "prefix_kv_cache": False,
                     "noise_shape": [1, 1024, 64], "noise_dtype": "float32",
                     "rng_version": "torch-2.9.0-cpu-randn-packed-v1", "denoiser_fp16_clip": [-65504, 65504],
                     "zero_centered_norm_dtype": "float32"},
        "components": components,
    }
    validate_profile(profile)
    return profile


def validate_mobile_profile(profile):
    digest = validate_profile(profile)
    candidates = [name for name in CANDIDATES if profile["id"] == f"qmi-mobile-{name}-v1"]
    if len(candidates) != 1:
        raise ValueError("The mobile profile identity is unsupported")
    expected = make_profile(candidates[0], profile["source"]["dependency_lock_sha256"],
                            profile["source"]["converter_sha256"])
    if digest != canonical_hash(expected):
        raise ValueError("The candidate precision, weight encoding or physical contract changed")
    return digest
