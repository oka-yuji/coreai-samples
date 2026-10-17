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
from .source import add_source_arguments, array, canonical_positions, generation_settings, model_from_arguments
from .wrappers import names


class Capture:
    def __init__(self, root, fixture, model):
        self.root = root
        self.fixture = fixture
        self.model = model
        self.records = []
        self.counts = {"projection": 0, "talker": 0, "predictor": 0, "codec": 0}
        self.prefix = None
        self.positions = None
        self.predictor_prefix = None
        self.predictor_index = None
        self.sequence = None
        self.first_prefix = None
        self.codec_frames = None

    def save(self, kind, label, inputs, outputs):
        identifier = f"{kind}-{self.fixture}-{label}"
        directory = self.root / "tensors" / identifier
        directory.mkdir(parents=True)
        record = {"id": identifier, "kind": kind, "directory": f"tensors/{identifier}"}
        input_names, output_names = names(kind)
        for category, fields, values in [("inputs", input_names, inputs), ("outputs", output_names, outputs)]:
            record[category] = []
            for name, value in zip(fields, values, strict=True):
                value = canonical_positions(value) if category == "inputs" and name == "positions" else array(value)
                if not np.isfinite(value).all():
                    raise ValueError(f"Non-finite reference: {identifier}/{name}")
                np.save(directory / f"{category}-{name}.npy", value, allow_pickle=False)
                record[category].append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)})
        self.records.append(record)
        write_json(directory / "record.json", record)

    def install(self, stack):
        def projection(module, args, result):
            index = self.counts["projection"]
            self.counts["projection"] += 1
            self.save("projection", str(index), (args[0],), (result,))

        stack.callback(self.model.talker.text_projection.register_forward_hook(projection).remove)

        def before_talker(module, args, kwargs):
            hidden = kwargs["inputs_embeds"]
            positions = kwargs["position_ids"]
            if self.prefix is None:
                self.prefix = hidden.detach().clone()
                self.positions = positions.detach().clone()
                self.first_prefix = self.prefix.clone()
            else:
                self.prefix = torch.cat([self.prefix, hidden], dim=1)
                self.positions = torch.cat([self.positions, positions], dim=2)

        def after_talker(module, args, kwargs, result):
            index = self.counts["talker"]
            self.counts["talker"] += 1
            if index in (0, 1, 8, 32):
                last = result.last_hidden_state[:, -1, :]
                self.save("talker", str(index), (self.prefix, self.positions),
                          (self.model.talker.codec_head(last), last))

        stack.callback(self.model.talker.model.register_forward_pre_hook(before_talker, with_kwargs=True).remove)
        stack.callback(self.model.talker.model.register_forward_hook(after_talker, with_kwargs=True).remove)

        def before_predictor(module, args, kwargs):
            hidden = kwargs.get("inputs_embeds")
            if hidden is not None and hidden.shape[1] > 1:
                self.predictor_prefix = hidden.detach().clone()
                self.predictor_index = hidden.shape[1] - 2
            else:
                self.predictor_index = int(kwargs["generation_steps"])
                hidden = module.model.get_input_embeddings()[self.predictor_index - 1](kwargs["input_ids"])
                self.predictor_prefix = torch.cat([self.predictor_prefix, hidden], dim=1)

        def after_predictor(module, args, kwargs, result):
            index = self.counts["predictor"]
            self.counts["predictor"] += 1
            if index in (0, 1, 14, 15, 29, 63, 255):
                self.save("predictor", str(index),
                          (self.predictor_prefix, torch.tensor([self.predictor_index], dtype=torch.int32)),
                          (result.logits[:, -1, :],))

        predictor = self.model.talker.code_predictor
        stack.callback(predictor.register_forward_pre_hook(before_predictor, with_kwargs=True).remove)
        stack.callback(predictor.register_forward_hook(after_predictor, with_kwargs=True).remove)

        def codec(module, args, result):
            index = self.counts["codec"]
            self.counts["codec"] += 1
            self.codec_frames = args[0].shape[-1]
            self.save("codec", str(index), (args[0],), (result,))

        stack.callback(self.model.speech_tokenizer.model.decoder.register_forward_hook(codec).remove)
        native_generate = self.model.talker.generate

        def generate(*args, **kwargs):
            directory = self.root / f"{self.fixture}-prepared"
            directory.mkdir()
            for name in ("inputs_embeds", "attention_mask", "trailing_text_hidden", "tts_pad_embed"):
                np.save(directory / f"{name}.npy", array(kwargs[name]), allow_pickle=False)
            result = native_generate(*args, **kwargs)
            self.sequence = result.sequences.detach().clone()
            return result

        stack.enter_context(patch.object(self.model.talker, "generate", side_effect=generate))

    def positional_checks(self):
        for position in (0, 255, 256, 511):
            length = position + 1
            repeats = (length + self.first_prefix.shape[1] - 1) // self.first_prefix.shape[1]
            hidden = self.first_prefix.repeat(1, repeats, 1)[:, :length]
            positions = torch.arange(length).view(1, 1, -1).expand(3, 1, -1)
            result = self.model.talker.model(inputs_embeds=hidden, position_ids=positions,
                                             attention_mask=torch.ones(1, length, dtype=torch.long),
                                             use_cache=False, past_key_values=None)
            last = result.last_hidden_state[:, -1, :]
            self.save("talker", f"position-{position}", (hidden, positions),
                      (self.model.talker.codec_head(last), last))


def main():
    parser = argparse.ArgumentParser()
    add_source_arguments(parser)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    specification = json.loads(args.fixtures.read_text())
    wrapper, contract = model_from_arguments(args)
    report = {"complete": False, "contract": contract, "specification": specification,
              "profile": "original_fp32_cpu_eager", "records": [], "utterances": []}
    write_json(args.output / "reference.json", report)
    for index, fixture in enumerate(specification["fixtures"]):
        torch.manual_seed(specification["seed"])
        capture = Capture(args.output, fixture["id"], wrapper.model)
        started = time.perf_counter()
        with torch.inference_mode():
            with ExitStack() as stack:
                capture.install(stack)
                waves, rate = wrapper.generate_custom_voice(text=fixture["text"], **generation_settings(specification))
            if index == 0:
                capture.positional_checks()
        audio = np.asarray(waves[0])
        if rate != contract["sample_rate"] or not audio.size or not np.isfinite(audio).all():
            raise ValueError("The original model produced invalid audio")
        sf.write(args.output / f"{fixture['id']}.wav", audio, rate, subtype="FLOAT")
        np.save(args.output / f"{fixture['id']}-sequence.npy", array(capture.sequence), allow_pickle=False)
        token_ids = wrapper._tokenize_texts([wrapper._build_assistant_text(fixture["text"])])[0]
        write_json(args.output / f"{fixture['id']}-tokens.json", token_ids.tolist())
        row = {"id": fixture["id"], "audio_seconds": len(audio) / rate, "seconds": time.perf_counter() - started,
               "frames": capture.codec_frames, "calls": capture.counts,
               "stop_reason": "eos" if int(capture.sequence[0, -1]) == wrapper.model.config.talker_config.codec_eos_token_id else "token_limit"}
        report["records"].extend(capture.records)
        report["utterances"].append(row)
        write_json(args.output / "reference.json", report)
        print(json.dumps(row), flush=True)
    report["complete"] = True
    write_json(args.output / "reference.json", report)


if __name__ == "__main__":
    main()
