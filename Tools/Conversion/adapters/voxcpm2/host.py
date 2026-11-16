import math
import operator

import numpy as np

WORD_BOUNDARY = chr(0x2581)
PATCH = (1, 64, 4)
PAIR = (2, 64, 4)
HIDDEN = (1, 2048)
NORM_EPSILON = np.float32(1e-8)


def is_kanji(text):
    return all(0x4E00 <= ord(character) <= 0x9FFF for character in text)


def prompt_ids(token_ids, vocabulary, audio_start_token):
    kanji = {}
    for identifier, token in vocabulary.items():
        if token and is_kanji(token):
            kanji[token] = min(identifier, kanji.get(token, identifier))
    ids = []
    for identifier in map(operator.index, token_ids):
        if identifier not in vocabulary:
            raise ValueError(f"Token ID {identifier} is not in the vocabulary")
        clean = vocabulary[identifier].replace(WORD_BOUNDARY, "")
        if len(clean) < 2 or clean not in kanji:
            ids.append(identifier)
            continue
        for character in clean:
            if character not in kanji:
                raise ValueError(f"U+{ord(character):04X} of token ID {identifier} is not in the vocabulary")
            ids.append(operator.index(kanji[character]))
    return ids + [operator.index(audio_start_token)]


def patch_limit(text_tokens, capacity, generation):
    text_tokens, capacity = operator.index(text_tokens), operator.index(capacity)
    if text_tokens < 0:
        raise ValueError(f"The text token count must not be negative: {text_tokens}")
    limit = min(int(generation["max_patches_ratio"] * text_tokens + generation["max_patches_offset"]),
                operator.index(generation["max_len"]), capacity - text_tokens - 1)
    if limit < 1:
        raise ValueError(f"{text_tokens} text tokens leave no patch in capacity {capacity}")
    return limit


def time_span(generation):
    steps = operator.index(generation["inference_timesteps"])
    if steps < 1:
        raise ValueError(f"inference_timesteps must be at least 1: {steps}")
    index = np.arange(steps + 1)
    step = np.float32(-1) / np.float32(steps)
    linear = np.where(index < (steps + 1) // 2, np.float32(1) + step * index.astype(np.float32),
                      np.float32(0) - step * (steps - index).astype(np.float32))
    cosine = np.cos((np.float32(math.pi / 2) * linear).astype(np.float64)).astype(np.float32)
    return linear + np.float32(generation["sway_sampling_coef"]) * ((cosine - np.float32(1)) + linear)


def float_array(value, shape, name):
    value = np.array(value, dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {list(shape)}, not {list(value.shape)}")
    return value


class Solver:
    def __init__(self, noise, dit_hidden, cond, generation):
        noise = float_array(noise, PATCH, "noise")
        dit_hidden = float_array(dit_hidden, HIDDEN, "dit_hidden")
        cond = float_array(cond, PATCH, "cond")
        self.times = time_span(generation)
        self.steps = len(self.times) - 1
        self.zero_star = bool(generation["cfg_zero_star"])
        zero_steps = operator.index(generation["zero_init_steps"]) if self.zero_star else 0
        calls = operator.index(generation["dit_calls_per_patch"])
        if not 0 <= zero_steps <= self.steps or calls != self.steps - zero_steps:
            raise ValueError(f"{self.steps} steps with {zero_steps} zero-velocity steps do not make {calls} calls")
        self.guidance = np.float32(generation["cfg_value"])
        self.x = noise
        self.mu = np.concatenate([dit_hidden, np.zeros_like(dit_hidden)])
        self.cond = np.concatenate([cond, cond])
        self.t = self.times[0]
        self.dt = self.times[0] - self.times[1]
        self.step = 1
        while self.step <= zero_steps:
            self.next_time()

    @property
    def latent(self):
        return self.x.copy()

    def inputs(self):
        if self.step > self.steps:
            return None
        return {"x": np.concatenate([self.x, self.x]), "mu": self.mu.copy(),
                "t": np.array([self.t, self.t], dtype=np.float32), "cond": self.cond.copy(),
                "dt": np.zeros(2, dtype=np.float32)}

    def advance(self, velocity):
        if self.step > self.steps:
            raise ValueError("The solver has no estimator call left")
        velocity = float_array(velocity, PAIR, "velocity")
        conditional, unconditional = velocity[:1], velocity[1:]
        scale = np.float32(1)
        if self.zero_star:
            scale = (np.sum(conditional * unconditional, dtype=np.float32)
                     / (np.sum(unconditional * unconditional, dtype=np.float32) + NORM_EPSILON))
        scaled = unconditional * scale
        self.x = self.x - self.dt * (scaled + self.guidance * (conditional - scaled))
        self.next_time()

    def next_time(self):
        self.t = self.t - self.dt
        if self.step < self.steps:
            self.dt = self.t - self.times[self.step + 1]
        self.step += 1


def patch_step(index, limit, stop_logits, generation):
    index, limit = operator.index(index), operator.index(limit)
    logits = np.asarray(stop_logits).reshape(-1)
    if logits.size != 2:
        raise ValueError(f"stop_logits must hold 2 values, not {logits.size}")
    if not 0 <= index < limit:
        raise ValueError(f"The patch index {index} is outside 0..{limit - 1}")
    if index > operator.index(generation["min_len"]) and logits[1] > logits[0]:
        return "stopped"
    return "limit" if index == limit - 1 else "next"
