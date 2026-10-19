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
from .reference import array
from .runtime import CoreAIComponents
from .source import add_source_arguments, configure_inference, model_from_arguments, seed_everything, text_batch


class CoreAIBridge:
    def __init__(self, model, components):
        self.model = model
        self.components = components
        for module in (model.local_transformer_in_projection, model.local_transformer_audio_out_projection,
                       model.audio_in_projection):
            if not isinstance(module, torch.nn.Identity):
                raise ValueError("Unexpected learned projection outside the converted components")

    def run(self, kind, *inputs):
        return tuple(torch.from_numpy(value) for value in self.components.call(kind, tuple(array(value) for value in inputs)))

    def encode(self, batch):
        tokens, lengths = batch["text"], batch["text_lens"]
        mask = torch.arange(tokens.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        encoded, = self.run("encoder", tokens, mask)
        embedded = self.model.text_embedding(tokens) * mask.unsqueeze(-1)
        return tokens, lengths, mask, embedded, encoded

    def forward(self, dec_input_embedded, dec_input_mask, cond, cond_mask, attn_prior, multi_encoder_mapping):
        if multi_encoder_mapping is not None or isinstance(cond, list):
            raise ValueError("Multiple context encoders are outside this profile")
        if isinstance(attn_prior, list):
            prior = next((item for item in attn_prior if item is not None), None)
        else:
            prior = attn_prior
        if prior is None:
            prior = torch.ones(cond.shape[0], 1, cond.shape[1])
        logits, last, attention = self.run("decoder", dec_input_embedded, dec_input_mask, cond, cond_mask, prior)
        layers = [{"cross_attn_probabilities": [attention[:, index, None, None, :], None]}
                  for index in range(attention.shape[1])]
        return logits.unsqueeze(1), layers, last.unsqueeze(1), None

    def local_sample(self, dec_output, temperature=0.7, topk=80, unfinished_items=None, finished_items=None,
                     use_cfg=False, cfg_scale=1.0, use_kv_cache=False, forbid_audio_eos=False, sanitize_logits=False):
        from nemo.collections.tts.modules.magpietts_modules import clear_forbidden_logits

        if not use_cfg or use_kv_cache or dec_output.shape[0] != 2:
            raise ValueError("This research profile requires one CFG pair without KV caching")
        hidden = dec_output.unsqueeze(1)
        sampled = []
        for index in range(16):
            logits, = self.run("local", hidden, torch.tensor([index], dtype=torch.int32))
            logits[0] = cfg_scale * logits[0] + (1.0 - cfg_scale) * logits[1]
            if sanitize_logits:
                logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
            for item in unfinished_items or {}:
                logits[item, self.model.audio_eos_id] = float("-inf")
            for item in finished_items or {}:
                logits[item] = float("-inf")
                logits[item, self.model.audio_eos_id] = 0.0
            logits = clear_forbidden_logits(logits.unsqueeze(1), self.model.codebook_size,
                                            forbid_audio_eos=forbid_audio_eos).squeeze(1)
            threshold = torch.topk(logits, topk, dim=-1).values[:, -1, None]
            logits[logits < threshold] = float("-inf")
            if temperature <= 0:
                token = logits.argmax(dim=-1, keepdim=True)
            else:
                token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)
            token[1] = token[0]
            sampled.append(token)
            embedded = self.model.audio_embeddings[index](token.squeeze(-1)).unsqueeze(1)
            hidden = torch.cat([hidden, embedded], dim=1)
        return torch.cat(sampled, dim=1).reshape(2, 2, 8).permute(0, 2, 1)[:1]

    def decode(self, inputs, input_len):
        audio, = self.run("codec", inputs, input_len)
        return audio, input_len * self.model.codec_model_samples_per_frame

    def install(self, stack):
        stack.enter_context(patch.object(self.model, "_encode_text_input", side_effect=self.encode))
        stack.enter_context(patch.object(self.model, "forward", side_effect=self.forward))
        stack.enter_context(patch.object(self.model._lt_helper, "sample_autoregressive", side_effect=self.local_sample))
        stack.enter_context(patch.object(self.model._codec_model.audio_decoder, "forward", side_effect=self.decode))


def main():
    parser = argparse.ArgumentParser()
    add_source_arguments(parser)
    parser.add_argument("--export", type=Path, action="append", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    specification = json.loads(args.fixtures.read_text())
    model, contract = model_from_arguments(args)
    configure_inference(model, specification)
    report = {"complete": False, "passed": False, "contract": contract, "utterances": []}
    write_json(args.output / "generation.json", report)
    components = CoreAIComponents(args.export, contract)
    try:
        bridge = CoreAIBridge(model, components)
        fixtures = [item for item in specification["fixtures"] if args.only is None or item["id"] == args.only]
        if not fixtures:
            raise ValueError("No matching generation fixture")
        with ExitStack() as stack, torch.inference_mode():
            bridge.install(stack)
            for fixture in fixtures:
                seed_everything(specification["seed"])
                batch = text_batch(model, fixture, specification["speaker"])
                started = time.perf_counter()
                result = model.infer_batch(batch, use_cfg=True, use_local_transformer_for_inference=True)
                seconds = time.perf_counter() - started
                length = int(result.predicted_audio_lens[0])
                audio = result.predicted_audio[0, :length].numpy()
                if not len(audio) or not np.isfinite(audio).all():
                    raise ValueError("Core AI synthesis did not produce finite, nonempty audio")
                sf.write(args.output / f"{fixture['id']}.wav", audio, contract["sample_rate"], subtype="FLOAT")
                frames = int(result.predicted_codes_lens[0])
                row = {"id": fixture["id"], "seconds": seconds, "audio_seconds": length / contract["sample_rate"],
                       "frames": frames, "stop_reason": "eos" if frames < specification["max_decoder_frames"] else "frame_limit",
                       "finite": True, "output": f"{fixture['id']}.wav"}
                report["utterances"].append(row)
                write_json(args.output / "generation.json", report)
                print(json.dumps(row), flush=True)
        report["complete"] = True
        report["passed"] = bool(report["utterances"])
        report["components"] = components.metrics
        write_json(args.output / "generation.json", report)
    finally:
        components.close()


if __name__ == "__main__":
    main()
