from pathlib import Path

import numpy as np

from .contracts import read_json
from .preprocessing import DENOISER_NAMES, TEXT_NAMES, denoiser_inputs, denoiser_layout, text_inputs


def arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name].copy() for name in values.files}


def load_examples(kind, reference, snapshot):
    identity = read_json(Path(reference) / "identity.json")
    if identity["dtype"] != "torch.float32" or identity["use_kv_cache"] is not False:
        raise ValueError("Conversion requires the independent FP32 cache-off reference")
    records = []
    for case in ("generate", "edit"):
        directory = Path(reference) / case
        if kind == "vision":
            if case != "edit":
                continue
            captured = arrays(directory / "vision-inputs.npz")
            if not np.array_equal(captured["grid_thw"], [[1, 32, 32]]):
                raise ValueError("Reference vision grid does not match the export profile")
            values = {"pixel_values": captured["arg_0"]}
            expected = arrays(directory / "vision-reference.npz")
            names = ("merged", "deepstack_0", "deepstack_1", "deepstack_2")
            outputs = tuple(expected[name] for name in names)
        elif kind == "text":
            encoded = arrays(directory / "encoder-inputs.npz")
            features = None
            if case == "edit":
                vision = arrays(directory / "vision-reference.npz")
                features = tuple(vision[name] for name in ("merged", "deepstack_0", "deepstack_1", "deepstack_2"))
            values = text_inputs(snapshot, encoded, features)
            values = {name: values[name] for name in TEXT_NAMES}
            names, outputs = ("hidden",), (np.load(directory / "language-reference.npy", allow_pickle=False),)
        elif kind == "denoiser":
            inputs = arrays(directory / "transformer-inputs.npz")
            text = inputs["encoder_hidden_states"]
            structure = denoiser_layout(inputs["img_mask"][0, :text.shape[1]])
            values = denoiser_inputs(inputs["hidden_states"], text, float(inputs["timestep"][0]), structure)
            names, outputs = ("noise",), (np.load(directory / "noise-0.npy", allow_pickle=False),)
        else:
            operation = "encode" if kind == "vae_encoder" else "decode"
            if operation == "encode" and case != "edit":
                continue
            values = {"value": np.load(directory / f"vae-{operation}-input.npy", allow_pickle=False)}
            names, outputs = ("result",), (np.load(directory / f"vae-{operation}-output.npy", allow_pickle=False),)
        records.append({"case": case, "values": values, "output_names": names, "expected": outputs})
    return records


def comparable(kind, value, expected):
    return value[:, :expected.shape[1]] if kind in ("text", "denoiser") else value
