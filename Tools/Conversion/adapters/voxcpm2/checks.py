from pathlib import Path

import numpy as np

from .source import compare, summarize


def read(value):
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value).astype(np.float32)


class Checks:
    def __init__(self, components, contract, reference, record, dtype):
        self.components, self.contract, self.dtype = components, contract, dtype
        self.limit = contract["criteria"]["normalized_max_limit"]
        self.directory = Path(reference) / record["directory"]
        self.record = record
        self.rows = []

    def load(self, name):
        return np.load(self.directory / f"{name}.npy", allow_pickle=False)

    def floats(self, value):
        return np.asarray(value, dtype=self.dtype)

    def add(self, asset, case, reference, actual, gate=True, **extra):
        self.rows.append({"asset": asset, "case": case, "fixture": self.record["id"], "gate": gate,
                          **compare(reference, actual, self.limit), **extra})

    def flag(self, asset, case, passed, **extra):
        self.rows.append({"asset": asset, "case": case, "fixture": self.record["id"], "gate": True,
                          "passed": bool(passed), "nonfinite": 0, **extra})

    def padded(self, value, fill):
        return self.floats(np.concatenate((value, fill), axis=1))

    def lm(self, asset):
        prefix, size, length = asset, self.contract["execution"]["prefill_length"], self.record["length"]
        source = self.load("embeds" if asset == "tslm" else "prefill_fuse_out")
        blank = np.zeros((1, size - length, source.shape[-1]), np.float32)
        state = self.components.state(asset, "prefill")
        first = self.components.call(asset, "prefill", {"embeds": self.padded(source, blank),
                                                       "length": np.array([length], np.int32)}, state)["hidden"]
        self.add(asset, "prefill.hidden", self.load(f"{prefix}_prefill_out"), first[:, :length])
        for side in ("k", "v"):
            cache = read(state[f"{side}_cache"])
            self.add(asset, f"prefill.{side}_cache", self.load(f"{prefix}_prefill_{side}"), cache[:, :, :length])
            self.flag(asset, f"prefill.{side}_cache_padding_zero", not np.any(cache[:, :, length:size]))
        inputs, outputs = self.load(f"{prefix}_step_in"), self.load(f"{prefix}_step_out")
        self.flag(asset, "step.count", len(inputs) >= self.contract["criteria"]["min_kv_steps"], steps=len(inputs))
        for index, (value, expected) in enumerate(zip(inputs, outputs, strict=True)):
            actual = self.components.call(asset, "step", {"embed": self.floats(value),
                                                          "position": np.array([length + index], np.int32)}, state)
            self.add(asset, "step.hidden", expected, actual["hidden"], index=index, position=length + index)
        other = self.components.state(asset, "prefill")
        noise = np.random.default_rng(length).standard_normal(blank.shape).astype(np.float32)
        second = self.components.call(asset, "prefill", {"embeds": self.padded(source, noise),
                                                        "length": np.array([length], np.int32)}, other)["hidden"]
        self.add(asset, "prefill.noise_padding_hidden", first[:, :length], second[:, :length],
                 bitwise=bool(np.array_equal(first[:, :length], second[:, :length])))
        for side in ("k", "v"):
            cache, baseline = read(other[f"{side}_cache"]), read(state[f"{side}_cache"])
            self.add(asset, f"prefill.noise_padding_{side}_cache", baseline[:, :, :length], cache[:, :, :length],
                     bitwise=bool(np.array_equal(baseline[:, :, :length], cache[:, :, :length])))
            self.flag(asset, f"prefill.noise_padding_{side}_cache_zero", not np.any(cache[:, :, length:size]))

    def tslm(self):
        size, length = self.contract["execution"]["prefill_length"], self.record["length"]
        ids = np.concatenate((self.load("ids"), np.full((1, size - length), self.contract["execution"]["pad_token"],
                                                        np.int32)), axis=1)
        embeds = self.components.call("tslm", "embed", {"ids": ids})["embeds"]
        self.add("tslm", "embed.embeds", self.load("embeds"), embeds[:, :length])
        self.lm("tslm")

    def ralm(self):
        self.lm("ralm")

    def proj(self):
        size, length = self.contract["execution"]["prefill_length"], self.record["length"]
        lm_hidden, residual, mu, stop = (self.load(name) for name in ("lm_hidden_in", "residual_hidden_in", "mu", "stop_logits"))
        for index in range(len(mu)):
            out = self.components.call("proj", "heads", {"lm_hidden": self.floats(lm_hidden[index]),
                                                        "residual_hidden": self.floats(residual[index])})
            self.add("proj", "heads.dit_hidden", mu[index], out["dit_hidden"], index=index)
            self.add("proj", "heads.stop_logits", stop[index], out["stop_logits"], index=index)
            self.flag("proj", "heads.stop_argmax", np.argmax(stop[index]) == np.argmax(out["stop_logits"]), index=index)
        raw, embed, fsq, fused, ties = (self.load(name) for name in ("tslm_step_out", "tslm_step_in", "fsq_out", "fuse_out",
                                                                     "fsq_tie_distance"))
        candidates = self.candidates()
        for index in range(len(raw)):
            out = self.components.call("proj", "fuse", {"lm_raw": self.floats(raw[index]),
                                                       "curr_embed": self.floats(embed[index])})
            outcome = classify_fuse((fsq[index], fused[index]), (out["lm_hidden"], out["residual_input"]),
                                    candidates.get(index, []), self.limit)
            measured = [outcome[name] for name in ("lm_hidden", "residual_input") if "normalized_max" in outcome[name]]
            errors = {key: max(row[key] for row in measured) for key in ("normalized_max", "max_abs")} if measured else {}
            self.rows.append({"asset": "proj", "case": "fuse", "fixture": self.record["id"], "gate": True, "index": index,
                              "passed": outcome["category"] != "fail",
                              "nonfinite": sum(outcome[name]["nonfinite"] for name in ("lm_hidden", "residual_input")),
                              **errors, "fsq_tie_distance": float(ties[index]),
                              "candidates": len(candidates.get(index, [])), **outcome})
        hidden = self.load("tslm_prefill_out")
        blank = np.zeros((1, size - length, hidden.shape[-1]), np.float32)
        out = self.components.call("proj", "prefill_fuse", {"hidden": self.padded(hidden, blank)})
        self.add("proj", "prefill_fuse.residual_inputs", self.load("prefill_fuse_out"), out["residual_inputs"][:, :length])

    def locenc(self):
        patches, embeds = self.load("patch"), self.load("curr_embed")
        for index in range(len(patches)):
            out = self.components.call("locenc", "main", {"patch": self.floats(patches[index])})
            self.add("locenc", "main.curr_embed", embeds[index][:, 0], out["curr_embed"], index=index)

    def dit(self):
        values = {name: self.load(f"dit_{name}") for name in ("x", "mu", "t", "cond", "dt", "velocity")}
        patches, calls = values["x"].shape[:2]
        for patch in range(patches):
            for call in range(calls):
                inputs = {name: self.floats(values[name][patch, call]) for name in ("x", "mu", "t", "cond", "dt")}
                out = self.components.call("dit", "main", inputs)
                self.add("dit", "main.velocity", values["velocity"][patch, call], out["velocity"], index=patch, call=call)

    def vae(self):
        latent, expected, full = self.load("latent"), self.load("stream_audio"), self.load("full_audio")
        state = self.components.state("vae", "main")
        chunks = []
        frames = self.contract["architecture"]["patch_size"]
        for index in range(latent.shape[-1] // frames):
            out = self.components.call("vae", "main", {"latent": self.floats(latent[..., frames * index:frames * (index + 1)])}, state)
            chunks.append(out["audio"])
            self.add("vae", "main.chunk_audio", expected[index], out["audio"], gate=False, index=index,
                     reference_rms=float(np.sqrt(np.mean(np.square(expected[index], dtype=np.float64)))))
        joined = np.concatenate(chunks, axis=-1)
        self.add("vae", "main.joined_audio", np.concatenate(list(expected), axis=-1), joined)
        self.add("vae", "main.joined_audio_vs_full_decode", full, joined, gate=False)

    def candidates(self):
        steps, dims, distances, codes, lm_hidden, residual = (
            self.load(f"fsq_candidate_{name}") for name in ("step", "dim", "distance", "code", "lm_hidden", "residual_input"))
        result = {}
        for row in range(len(steps)):
            result.setdefault(int(steps[row]), []).append(
                {"dim": int(dims[row]), "distance": float(distances[row]), "code": float(codes[row, 0]),
                 "candidate_code": float(codes[row, 1]), "outputs": (lm_hidden[row], residual[row])})
        return result

    def run(self, assets):
        for asset in assets:
            getattr(self, asset)()
        return self.rows


def tie_distance(scaled):
    scaled = np.asarray(scaled, dtype=np.float64)
    return np.abs(np.abs(scaled - np.floor(scaled)) - 0.5)


def fuse_candidates(scaled, codes, distance):
    scaled = np.asarray(scaled, dtype=np.float64).reshape(-1)
    codes = np.asarray(codes, dtype=np.float64).reshape(-1)
    distances = tie_distance(scaled)
    rows = []
    for dim in np.flatnonzero(distances <= distance):
        lower = float(np.floor(scaled[dim]))
        if codes[dim] not in (lower, lower + 1):
            raise ValueError("A quantizer code is not next to its rounding boundary")
        rows.append({"dim": int(dim), "distance": float(distances[dim]), "code": float(codes[dim]),
                     "candidate_code": lower + 1 if codes[dim] == lower else lower})
    return rows


def classify_fuse(expected, actual, candidates, limit):
    normal = [compare(reference, value, limit) for reference, value in zip(expected, actual, strict=True)]
    result = {"lm_hidden": normal[0], "residual_input": normal[1]}
    if any(row["nonfinite"] for row in normal):
        return {"category": "fail", **result}
    if all(row["passed"] for row in normal):
        return {"category": "match", **result}
    peaks = [max(float(np.max(np.abs(np.asarray(reference, dtype=np.float64)))), 1e-8) for reference in expected]
    matched = []
    for candidate in candidates:
        if any(np.shape(alternative) != np.shape(value) for alternative, value in zip(candidate["outputs"], actual, strict=True)):
            raise ValueError("A boundary candidate does not have the output shape")
        errors = [float(np.max(np.abs(np.asarray(alternative, dtype=np.float64) - np.asarray(value, dtype=np.float64)))) / peak
                  for alternative, value, peak in zip(candidate["outputs"], actual, peaks, strict=True)]
        if all(error <= limit for error in errors):
            matched.append({key: candidate[key] for key in ("dim", "distance", "code", "candidate_code")}
                           | {"lm_hidden_error": errors[0], "residual_input_error": errors[1]})
    if len(matched) == 1:
        return {"category": "boundary", "boundary": matched[0], **result}
    return {"category": "fail", "matched_candidates": len(matched), **result}


def fsq_summary(rows, criteria):
    fuse = [row for row in rows if row["case"] == "fuse"]
    counts = {category: sum(row["category"] == category for row in fuse) for category in ("match", "boundary", "fail")}
    fixtures = {}
    for row in fuse:
        entry = fixtures.setdefault(row["fixture"], {"calls": 0, "match": 0, "boundary": 0, "fail": 0})
        entry["calls"] += 1
        entry[row["category"]] += 1
    limit = criteria["fsq_boundary_max_fraction"] * len(fuse)
    return {"calls": len(fuse), **counts, "boundary_limit": limit, "within_limit": counts["boundary"] <= limit,
            "fixtures": fixtures,
            "boundary_rows": [{"fixture": row["fixture"], "index": row["index"], **row["boundary"]}
                              for row in fuse if row["category"] == "boundary"]}


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault(f"{row['asset']}.{row['case']}", []).append(row)
    return {key: {**summarize(items), "gate": items[0]["gate"]} for key, items in sorted(groups.items())}


def verdict(rows, criteria):
    if criteria["fsq_boundary_max_dims"] != 1:
        raise ValueError("Only one changed quantizer dimension per call is supported")
    gated = [row for row in rows if row["gate"]]
    return (bool(gated) and all(row["passed"] and row["nonfinite"] == 0 for row in gated)
            and fsq_summary(rows, criteria)["within_limit"])
