import json

import torch

from .denoiser import Denoiser, VAEPart
from .encoders import TextEncoder, VisionEncoder, load_encoder
from .mobile_profile import validate_mobile_profile
from .quantization import StoredEmbedding, quantize, store_linear_weights_in_half


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def parameter_precision(module, dtype):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.data = parameter.data.to(dtype=dtype)
    return module


def load_denoiser(snapshot):
    from diffusers import QwenImage21Transformer2DModel
    from safetensors import safe_open

    directory = snapshot / "transformer"
    config = json.loads((directory / "config.json").read_text())
    with torch.device("meta"):
        model = QwenImage21Transformer2DModel.from_config(config)
    index = json.loads((directory / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    needed = set(model.state_dict())
    weights = {}
    for filename in sorted({index[name] for name in needed}):
        with safe_open(directory / filename, framework="pt", device="cpu") as source:
            for name in sorted(needed):
                if index[name] == filename:
                    weights[name] = source.get_tensor(name).float()
    model.load_state_dict(weights, strict=True, assign=True)
    temporal = model.time_text_embed.time_proj
    model.time_text_embed.time_proj = type(temporal)(timestep_dim=temporal.timestep_dim, time_factor=temporal.time_factor)
    rotary = model.pos_embed
    model.pos_embed = type(rotary)(theta=rotary.theta, axes_dim=rotary.axes_dim)
    if any(value.is_meta for value in (*model.parameters(), *model.buffers(), *model.pos_embed.freqs)):
        raise ValueError("The denoiser has an unmaterialized parameter or buffer")
    return model.eval().requires_grad_(False)


def configure_original(model, kind, profile):
    validate_mobile_profile(profile)
    component = profile["components"][kind]
    encoding = component["weight_encoding"]
    if encoding["kind"] == "dense":
        return parameter_precision(model, DTYPES[component["compute_dtype"]])
    if component["compute_dtype"] == "float32":
        model = store_linear_weights_in_half(model)
        if kind == "text":
            model.embed_tokens = StoredEmbedding(model.embed_tokens)
    else:
        parameter_precision(model, DTYPES[component["compute_dtype"]])
    return model


def load_original(snapshot, kind, profile):
    from diffusers import AutoencoderKLQwenImage21

    validate_mobile_profile(profile)
    if kind not in profile["components"]:
        raise ValueError("Unknown component")
    if kind in ("text", "vision"):
        model = load_encoder(snapshot, kind)
    elif kind == "denoiser":
        model = load_denoiser(snapshot)
    else:
        model = AutoencoderKLQwenImage21.from_pretrained(snapshot / "vae", torch_dtype=torch.float32, local_files_only=True)
    return configure_original(model, kind, profile).eval().requires_grad_(False)


def compress_original(model, kind, profile):
    encoding = profile["components"][kind]["weight_encoding"]
    if encoding["kind"] in ("dense", "dense-linear"):
        return model, {"kind": "dense", "layers": [], "dtype": encoding["dtype"]}
    return quantize(model, encoding["bits"], encoding["group_size"], dense_modules=encoding.get("dense_modules", ()))


class TextBoundary(torch.nn.Module):
    def __init__(self, original, dtype):
        super().__init__()
        self.encoder = TextEncoder(original)
        self.dtype = dtype

    def forward(self, input_ids, vision, deepstack, cosine, sine, mask, visual_mask):
        return self.encoder(input_ids, vision.to(self.dtype), deepstack.to(self.dtype),
                            cosine.to(self.dtype), sine.to(self.dtype), mask, visual_mask).float()


class VisionBoundary(torch.nn.Module):
    def __init__(self, original, dtype):
        super().__init__()
        self.encoder = VisionEncoder(original)
        self.dtype = dtype
        self.encoder.position = self.encoder.position.to(dtype)
        self.encoder.cosine = self.encoder.cosine.float()
        self.encoder.sine = self.encoder.sine.float()

    def forward(self, pixel_values):
        return tuple(value.float() for value in self.encoder(pixel_values.to(self.dtype)))


class DenoiserBoundary(torch.nn.Module):
    def __init__(self, original, dtype, rotary=None):
        super().__init__()
        self.denoiser = Denoiser(original, rotary=rotary)
        self.dtype = dtype

    def forward(self, latents, text, timestep, text_index, image_index, image_mask, cosine, sine, mask, target):
        return self.denoiser(latents.to(self.dtype), text.to(self.dtype), timestep.to(self.dtype),
                             text_index, image_index, image_mask, cosine, sine, mask, target).float()


class VAEBoundary(torch.nn.Module):
    def __init__(self, original, kind, dtype):
        super().__init__()
        self.vae = VAEPart(original, kind)
        self.dtype = dtype

    def forward(self, value):
        return self.vae(value.to(self.dtype)).float()


def export_boundary(original, kind, profile, *, rotary=None):
    if rotary is not None and (kind != "denoiser" or profile["components"][kind]["compute_dtype"] != "float32"):
        raise ValueError("Explicit Metal rotary is only supported for the Float32 denoiser")
    dtype = DTYPES[profile["components"][kind]["compute_dtype"]]
    if kind == "text":
        result = TextBoundary(original, dtype)
    elif kind == "vision":
        result = VisionBoundary(original, dtype)
    elif kind == "denoiser":
        result = DenoiserBoundary(original, dtype, rotary=rotary)
    else:
        result = VAEBoundary(original, kind, dtype)
    return result.eval().requires_grad_(False)
