import numpy as np

EXECUTION_SHAPES = {"talker": 512, "predictor": 16}


def prepare_inputs(kind, values):
    if kind == "talker":
        hidden, positions = values
        length = hidden.shape[1]
        return (np.pad(hidden, ((0, 0), (0, EXECUTION_SHAPES[kind] - length), (0, 0))),
                np.pad(positions, ((0, 0), (0, 0), (0, EXECUTION_SHAPES[kind] - length))),
                np.array([length - 1], dtype=np.int32))
    if kind == "predictor":
        hidden, codebook = values
        return np.pad(hidden, ((0, 0), (0, EXECUTION_SHAPES[kind] - hidden.shape[1]), (0, 0))), codebook
    return values
