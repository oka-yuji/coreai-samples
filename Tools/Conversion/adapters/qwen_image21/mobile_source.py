import gc

import torch

from .evaluation import SIGMAS, canonical_hash
from .mobile_models import DTYPES, compress_original, configure_original, parameter_precision
from .quantization import plain_reference


def independent_component(model, kind, profile, expected_compression):
    model = configure_original(model, kind, profile)
    model, compression = compress_original(model, kind, profile)
    if expected_compression is not None and canonical_hash(compression) != canonical_hash(expected_compression):
        raise ValueError("The source reference does not reproduce the exported weight identity: " + kind)
    if compression["kind"] == "linear-integer":
        if expected_compression is None:
            raise ValueError("Quantized references require the exported component weight records")
        model = plain_reference(model, compression)
    if any(torch.nn.utils.parametrize.is_parametrized(child) for child in model.modules()):
        raise ValueError("The source reference retains a custom quantization operator")
    return model.eval().requires_grad_(False), compression


def source_pipeline(snapshot, profile, compression_records, device):
    from diffusers import QwenImage21Pipeline

    pipeline = QwenImage21Pipeline.from_pretrained(snapshot, dtype=torch.float32, local_files_only=True)
    if list(pipeline.config.sample_sigmas) != list(SIGMAS):
        raise ValueError("The original source sampling schedule changed")
    observed = {}
    targets = (("vision", pipeline.text_encoder.model, "visual"),
               ("text", pipeline.text_encoder.model, "language_model"),
               ("denoiser", pipeline, "transformer"), ("vae_encoder", pipeline, "vae"))
    for kind, owner, attribute in targets:
        original = getattr(owner, attribute)
        reference, compression = independent_component(original, kind, profile, compression_records.get(kind))
        setattr(owner, attribute, reference)
        observed[kind] = compression
        del original, reference
        gc.collect()
    if profile["components"]["vae_decoder"]["compute_dtype"] != profile["components"]["vae_encoder"]["compute_dtype"]:
        raise ValueError("The shared source VAE must have one declared compute precision")
    observed["vae_decoder"] = observed["vae_encoder"]
    if "vae_decoder" in compression_records and canonical_hash(compression_records["vae_decoder"]) != canonical_hash(observed["vae_decoder"]):
        raise ValueError("The source VAE decoder differs from the exported weight identity")
    parameter_precision(pipeline.text_encoder.lm_head, DTYPES[profile["components"]["text"]["compute_dtype"]])
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, observed
