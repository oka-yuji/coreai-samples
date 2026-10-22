import json
from pathlib import Path

import numpy as np

from .examples import arrays
from .preprocessing import denoiser_inputs, denoiser_layout, text_inputs


VISION_NAMES = ("merged", "deepstack_0", "deepstack_1", "deepstack_2")


def tensor(path, shape):
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or list(value.shape) != list(shape) or not np.isfinite(value).all():
        raise ValueError("Source tensor has a different comparison region, dtype or finite status: " + str(path))
    return value


def fixed_examples(kind, directory, snapshot, metadata):
    directory = Path(directory)
    editing = metadata["operation"] == "edit"
    if kind in ("vision", "vae_encoder") and not editing:
        return
    if kind == "vision":
        inputs = arrays(directory / "vision-inputs.npz")
        if not np.array_equal(inputs.get("grid_thw"), [[1, 32, 32]]):
            raise ValueError("The original vision grid differs from the declared fixed shape")
        expected = [tensor(directory / ("vision-" + name + ".npy"), (256, 4096)) for name in VISION_NAMES]
        yield {"id": "vision", "values": {"pixel_values": inputs["arg_0"]}, "expected": expected}
    elif kind == "text":
        encoded = arrays(directory / "encoder-inputs.npz")
        if encoded["input_ids"].shape != (1, metadata["processed_tokens"]):
            raise ValueError("The source text prefix differs from the frozen processed-token count")
        features = [tensor(directory / ("vision-" + name + ".npy"), (256, 4096)) for name in VISION_NAMES] if editing else None
        expected = tensor(directory / "language-reference.npy", (1, metadata["processed_tokens"], 4096))
        yield {"id": "text", "values": text_inputs(snapshot, encoded, features), "expected": [expected]}
    elif kind == "denoiser":
        encoded = arrays(directory / "encoder-inputs.npz")
        slots = (encoded["input_ids"][0] == metadata["image_token_id"])[metadata["drop"]:]
        structure = denoiser_layout(slots)
        if structure["length"] != metadata["expanded_tokens"]:
            raise ValueError("The source joint region differs from the frozen expanded-token count")
        for index in range(8):
            inputs = arrays(directory / f"transformer-{index}-inputs.npz")
            layout = json.loads((directory / f"transformer-{index}-layout.json").read_text())
            expected_mask = np.concatenate((slots, np.ones(256, dtype=np.bool_)))[None]
            if (not np.array_equal(inputs["img_mask"], expected_mask)
                    or inputs["encoder_hidden_states"].shape != (1, len(slots), 4096)
                    or layout != {"img_shapes": [[[1, 32, 32]] * (int(editing) + 1)], "prefix_kv_cache": False}):
                raise ValueError("The source DiT input layout changed")
            expected = tensor(directory / f"joint-noise-{index}.npy", (1, structure["length"], 64))
            values = denoiser_inputs(inputs["hidden_states"], inputs["encoder_hidden_states"], float(inputs["timestep"][0]), structure)
            yield {"id": "denoiser-" + str(index), "values": values, "expected": [expected],
                   "target_indices": structure["target_indices"]}
    elif kind in ("vae_encoder", "vae_decoder"):
        encode = kind == "vae_encoder"
        operation = "encode" if encode else "decode"
        shape = (1, 4, 1, 512, 512) if encode else (1, 64, 1, 32, 32)
        output = "vae-encode-output.npy" if encode else "rgba.npy"
        result_shape = (1, 64, 1, 32, 32) if encode else (1, 4, 1, 512, 512)
        yield {"id": kind, "values": {"value": tensor(directory / f"vae-{operation}-input.npy", shape)},
               "expected": [tensor(directory / output, result_shape)]}
    else:
        raise ValueError("Unknown comparison component")
