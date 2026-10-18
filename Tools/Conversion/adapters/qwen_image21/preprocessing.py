from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
from PIL import Image
import torch
from transformers import Qwen3VLConfig, Qwen3VLProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLTextRotaryEmbedding
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope

from .contracts import JOINT_LENGTH, TEXT_LENGTH, layout, pad


SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
TEXT_NAMES = ("input_ids", "vision", "deepstack", "cosine", "sine", "mask", "visual_mask")
DENOISER_NAMES = ("latents", "text", "timestep", "text_index", "image_index", "image_mask", "cosine", "sine", "mask", "target")


def config_path(root):
    return Path(root) / "text_encoder"


def prompt_inputs(root, prompt, image=None):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A nonempty prompt is required")
    processor = Qwen3VLProcessor.from_pretrained(Path(root) / "processor", local_files_only=True)
    prefix = "<image1><|vision_start|><|image_pad|><|vision_end|>" if image is not None else ""
    template = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prefix}{prompt}<|im_end|>\n<|im_start|>assistant\n"
    kwargs = {"text": [template], "padding": True, "padding_side": "left", "return_tensors": "pt"}
    if image is not None:
        if image.size != (512, 512):
            raise ValueError("The reference image must be exactly 512 by 512 pixels")
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        kwargs["images"] = [rgb]
    encoded = processor(**kwargs)
    if encoded.input_ids.shape[1] > TEXT_LENGTH:
        raise ValueError("The processed prompt exceeds 512 tokens including image slots and the system prefix")
    system = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    drop = len(processor.apply_chat_template(system, tokenize=True, return_dict=False)[0])
    image_id = processor.tokenizer.encode("<|image_pad|>")[0]
    return {key: value.numpy() for key, value in encoded.items()}, drop, image_id


def text_inputs(root, encoded, vision_outputs=None):
    config = Qwen3VLConfig.from_pretrained(config_path(root), local_files_only=True)
    ids = np.asarray(encoded["input_ids"])
    if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= TEXT_LENGTH:
        raise ValueError("The text profile requires one prompt with at most 512 tokens")
    if not np.all(np.asarray(encoded["attention_mask"]) == 1):
        raise ValueError("The single-prompt input must not contain source padding")
    if np.any(ids < 0) or np.any(ids >= config.text_config.vocab_size):
        raise ValueError("A token is outside the vocabulary")
    length = ids.shape[1]
    visual = ids == config.image_token_id
    dense = np.zeros((1, length, 4096), dtype=np.float32)
    deepstack = np.zeros((3, 1, length, 4096), dtype=np.float32)
    if visual.any():
        if visual.sum() != 256 or vision_outputs is None or len(vision_outputs) != 4:
            raise ValueError("One reference image requires exactly 256 merged vision tokens and three DeepStack taps")
        for value in vision_outputs:
            if value.shape != (256, 4096) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Unexpected vision encoder output")
        dense[visual] = vision_outputs[0]
        for index, value in enumerate(vision_outputs[1:]):
            deepstack[index][visual] = value
        if not np.array_equal(encoded.get("image_grid_thw"), [[1, 32, 32]]):
            raise ValueError("The vision grid must be one 512-square image")
    elif vision_outputs is not None:
        raise ValueError("Vision outputs were provided without image slots")
    helper = SimpleNamespace(config=config)
    helper.get_vision_position_ids = MethodType(Qwen3VLModel.get_vision_position_ids, helper)
    position_ids, _ = Qwen3VLModel.get_rope_index(
        helper, torch.from_numpy(ids),
        torch.from_numpy(encoded.get("mm_token_type_ids", np.zeros_like(ids))),
        image_grid_thw=torch.from_numpy(encoded["image_grid_thw"]) if visual.any() else None,
        attention_mask=torch.from_numpy(encoded["attention_mask"]))
    with torch.no_grad():
        cosine, sine = Qwen3VLTextRotaryEmbedding(config.text_config)(torch.empty(1, length, 1), position_ids)
    mask = np.zeros((1, 1, TEXT_LENGTH, TEXT_LENGTH), dtype=np.bool_)
    mask[0, 0, :length, :length] = np.tri(length, dtype=np.bool_)
    mask[0, 0, length:, :length] = True
    return {"input_ids": pad(ids.astype(np.int32), (1, TEXT_LENGTH)),
            "vision": pad(dense, (1, TEXT_LENGTH, 4096)),
            "deepstack": pad(deepstack, (3, 1, TEXT_LENGTH, 4096)),
            "cosine": pad(cosine.numpy(), (1, TEXT_LENGTH, 128)),
            "sine": pad(sine.numpy(), (1, TEXT_LENGTH, 128)), "mask": mask,
            "visual_mask": pad(visual, (1, TEXT_LENGTH))}


def denoiser_layout(image_slots):
    values = layout(image_slots)
    with torch.no_grad():
        frequencies = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])(
            [(1, 32, 32)] * (values["reference_count"] + 1),
            torch.from_numpy(values["compact_image_mask"]), torch.device("cpu"))
    cosine = np.ones((1, JOINT_LENGTH, 1, 64), dtype=np.float32)
    sine = np.zeros_like(cosine)
    cosine[0, :values["length"], 0] = frequencies.real.numpy()
    sine[0, :values["length"], 0] = frequencies.imag.numpy()
    values.update(cosine=cosine, sine=sine)
    return values


def denoiser_inputs(latents, text, timestep, structure):
    if latents.shape != (1, 1024 * (structure["reference_count"] + 1), 64):
        raise ValueError("Latent tokens do not match the reference-image layout")
    values = {key: structure[key] for key in ("text_index", "image_index", "image_mask", "cosine", "sine", "mask", "target")}
    values.update(latents=pad(np.asarray(latents, dtype=np.float32), (1, 2048, 64)),
                  text=pad(np.asarray(text, dtype=np.float32), (1, TEXT_LENGTH, 4096)),
                  timestep=np.asarray([timestep], dtype=np.float32))
    return {name: values[name] for name in DENOISER_NAMES}


def initial_noise(seed):
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("The seed must be an integer between 0 and 2**63 - 1")
    value = torch.randn(1, 1, 64, 32, 32, generator=torch.Generator("cpu").manual_seed(seed))
    return value.reshape(1, 64, 1024).transpose(1, 2).contiguous().numpy()
