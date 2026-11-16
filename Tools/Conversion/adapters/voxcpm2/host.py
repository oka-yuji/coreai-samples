import numpy as np


def prompt_ids(token_ids, vocabulary, audio_start_token):
    return list(token_ids)


def patch_limit(text_tokens, capacity, generation):
    return 0


def time_span(generation):
    return np.zeros(generation["inference_timesteps"] + 1, dtype=np.float32)


class Solver:
    def __init__(self, noise, dit_hidden, cond, generation):
        self.remaining = generation["dit_calls_per_patch"]
        self.latent = np.zeros((1, 64, 4), dtype=np.float32)

    def inputs(self):
        if self.remaining <= 0:
            return None
        return {"x": np.zeros((2, 64, 4), dtype=np.float32), "mu": np.zeros((2, 2048), dtype=np.float32),
                "t": np.zeros(2, dtype=np.float32), "cond": np.zeros((2, 64, 4), dtype=np.float32),
                "dt": np.zeros(2, dtype=np.float32)}

    def advance(self, velocity):
        self.remaining -= 1


def patch_step(index, limit, stop_logits, generation):
    return "next"
