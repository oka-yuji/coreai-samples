import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from ...storage import write_json
from .source import add_source_arguments, configure_inference, model_from_arguments, seed_everything, text_batch
from .wrappers import names


def array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if np.issubdtype(value.dtype, np.floating) or value.dtype == np.bool_:
        return value.astype(np.float32)
    return value.astype(np.int32)


class Capture:
    def __init__(self, root, fixture):
        self.root = root
        self.fixture = fixture
        self.records = []
        self.decoder_count = 0
        self.local_count = 0
        self.local_input = None
        self.local_step = None

    def save(self, kind, label, inputs, outputs):
        key = f"{kind}-{self.fixture}-{label}"
        target = self.root / "tensors" / key
        target.mkdir(parents=True)
        input_names, output_names = names(kind)
        inventory = {}
        for category, fields, values in [("inputs", input_names, inputs), ("outputs", output_names, outputs)]:
            inventory[category] = []
            for name, value in zip(fields, values, strict=True):
                value = array(value)
                if not np.isfinite(value).all():
                    raise ValueError(f"Non-finite reference: {key}/{name}")
                np.save(target / f"{category}-{name}.npy", value, allow_pickle=False)
                inventory[category].append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)})
        record = {"id": key, "kind": kind, "directory": f"tensors/{key}", **inventory}
        self.records.append(record)
        write_json(target / "record.json", record)

    def install(self, model, stack):
        original = model.forward

        def decoder(*args, **kwargs):
            result = original(*args, **kwargs)
            step = self.decoder_count
            self.decoder_count += 1
            if step in (0, 1, 8, 32):
                if args:
                    values = dict(zip(("dec_input_embedded", "dec_input_mask", "cond", "cond_mask", "attn_prior", "multi_encoder_mapping"), args))
                    values.update(kwargs)
                else:
                    values = kwargs
                prior = values["attn_prior"]
                if isinstance(prior, list):
                    prior = next((item for item in prior if item is not None), None)
                memory = values["cond"]
                if prior is None:
                    prior = torch.ones(memory.shape[0], 1, memory.shape[1])
                attention = torch.stack([
                    layer["cross_attn_probabilities"][0][:, 0, -1, :] for layer in result[1]
                ], dim=1)
                inputs = (values["dec_input_embedded"], values["dec_input_mask"], memory, values["cond_mask"], prior)
                outputs = (result[0][:, -1, :], result[2][:, -1, :], attention)
                self.save("decoder", str(step), inputs, outputs)
            return result

        stack.enter_context(patch.object(model, "forward", side_effect=decoder))

        def before_local(module, args):
            self.local_step = self.local_count
            self.local_count += 1
            self.local_input = args[0].detach().clone() if self.local_step in (0, 3, 15, 16, 31, 64, 255) else None

        stack.callback(model.local_transformer.register_forward_pre_hook(before_local).remove)
        for index, projection in enumerate(model.local_transformer_out_projections):
            def after_projection(module, args, output, index=index):
                if self.local_input is not None:
                    self.save("local", str(self.local_step),
                              (self.local_input, torch.tensor([index], dtype=torch.int32)), (output,))
                    self.local_input = None
            stack.callback(projection.register_forward_hook(after_projection).remove)

        def after_codec(module, args, kwargs, output):
            latent = kwargs.get("inputs", args[0] if args else None)
            lengths = kwargs.get("input_len", args[1] if len(args) > 1 else None)
            self.save("codec", "full", (latent, lengths), (output[0],))

        stack.callback(model._codec_model.audio_decoder.register_forward_hook(after_codec, with_kwargs=True).remove)


def main():
    parser = argparse.ArgumentParser()
    add_source_arguments(parser)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    specification = json.loads(args.fixtures.read_text())
    fixtures = [item for item in specification["fixtures"] if args.only is None or item["id"] == args.only]
    if not fixtures:
        raise ValueError("No matching reference fixture")
    model, contract = model_from_arguments(args)
    configure_inference(model, specification)
    report = {"complete": False, "profile": "nemo_fp32_cpu_no_kv_cache", "contract": contract,
              "specification": specification, "codec_weights": model._codec_model._coreai_weight_audit,
              "records": [], "utterances": []}
    write_json(args.output / "reference.json", report)
    for fixture in fixtures:
        seed_everything(specification["seed"])
        capture = Capture(args.output, fixture["id"])
        started = time.perf_counter()
        batch = text_batch(model, fixture, specification["speaker"])
        with torch.inference_mode():
            tokens, lengths, mask, embedded, encoded = model._encode_text_input(batch)
            capture.save("encoder", "full", (tokens, mask), (encoded,))
            with ExitStack() as stack:
                capture.install(model, stack)
                generated = model.infer_batch(batch, use_cfg=True, use_local_transformer_for_inference=True)
        count = int(generated.predicted_audio_lens[0])
        audio = generated.predicted_audio[0, :count].detach().cpu().numpy()
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("The source failed to produce finite, nonempty audio")
        sf.write(args.output / f"{fixture['id']}.wav", audio, contract["sample_rate"], subtype="FLOAT")
        np.save(args.output / f"{fixture['id']}-codes.npy", array(generated.predicted_codes), allow_pickle=False)
        write_json(args.output / f"{fixture['id']}-tokens.json", tokens.tolist())
        utterance = {"id": fixture["id"], "seconds": time.perf_counter() - started,
                     "audio_seconds": count / contract["sample_rate"],
                     "frames": int(generated.predicted_codes_lens[0]), "decoder_calls": capture.decoder_count,
                     "stop_reason": "eos" if int(generated.predicted_codes_lens[0]) < specification["max_decoder_frames"] else "frame_limit"}
        report["records"].extend(capture.records)
        report["utterances"].append(utterance)
        write_json(args.output / "reference.json", report)
        print(json.dumps(utterance), flush=True)
    report["complete"] = True
    write_json(args.output / "reference.json", report)


if __name__ == "__main__":
    main()
