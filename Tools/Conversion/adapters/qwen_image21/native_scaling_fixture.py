import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def fixture():
    inputs = np.array([0.1, -0.1001, 10.05, -3.14159, 0.000061, 1.234567, 256.0625, -0.77], dtype=np.float32)
    means = np.array([0.17, -0.1001, 2.71828, -0.0101, 0.00001, 1.23456, -2.001, 0.3], dtype=np.float32)
    scales = np.array([0.19, 0.013, 1.001, 3.14159, 0.007, 0.7777, 2.001, 0.3], dtype=np.float32)
    values = np.resize(inputs, 65_536)
    rows = []
    for name in ("float32", "float16", "bfloat16"):
        dtype = getattr(torch, name)
        x = torch.from_numpy(values.copy()).to(dtype)
        mean = torch.from_numpy(np.resize(means, 64)).to(dtype).reshape(64, 1)
        std = torch.from_numpy(np.resize(scales, 64)).to(dtype).reshape(64, 1)
        normalize = ((x.reshape(64, 1024) - mean) / std).transpose(0, 1).contiguous()
        denormalize = x.reshape(1024, 64).transpose(0, 1).contiguous() * std + mean
        rows.append({"precision": name, **{operation: hashlib.sha256(tensor.float().numpy().tobytes()).hexdigest()
                      for operation, tensor in (("normalize", normalize), ("denormalize", denormalize))}})
    return {"source": "Diffusers b800f828f3951c4d206e154411bcbfa693cd322e VAE scaling formulas",
            "torch": torch.__version__, "device": "cpu", "input": inputs.tolist(), "mean": means.tolist(),
            "standardDeviation": scales.tolist(), "cases": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with args.output.open("x") as output:
        json.dump(fixture(), output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
