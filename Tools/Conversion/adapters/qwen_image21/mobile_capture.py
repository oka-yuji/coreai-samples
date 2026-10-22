from pathlib import Path

import numpy as np
import torch

from .fixtures import file_record, save_json
from .mobile_models import DTYPES


class NonFiniteSourceError(ValueError):
    pass


class SourceTrace:
    def __init__(self, pipeline, profile):
        self.pipeline = pipeline
        self.dit_dtype = DTYPES[profile["components"]["denoiser"]["compute_dtype"]]
        self.vae_dtype = DTYPES[profile["components"]["vae_encoder"]["compute_dtype"]]
        self.handles = []
        self.originals = {"prepare": pipeline.prepare_latents, "transformer": pipeline.transformer.forward,
                          "encode": pipeline.vae.encode, "decode": pipeline.vae.decode, "step": pipeline.scheduler.step}
        self.directory = None

    def begin(self, directory, operation, expected_inputs=None):
        self.directory = None
        if operation not in ("generate", "edit"):
            raise ValueError("Unknown source operation")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=False)
        self.directory = destination
        self.operation = operation
        self.expected_inputs = expected_inputs
        self.counts = {name: 0 for name in ("encoder_inputs", "language_inputs", "vision_inputs", "vision", "text",
                                          "denoiser", "vae_encoder", "vae_decoder", "scheduler", "callback")}
        self.tensors = {}
        self.steps = []

    def array(self, value):
        return (value.detach().float() if value.is_floating_point() else value.detach()).cpu().numpy()

    def output(self, name, value):
        if not value.is_floating_point():
            raise ValueError("Source outputs must be floating-point tensors: " + name)
        array = self.array(value)
        path = self.directory / (name + ".npy")
        np.save(path, array, allow_pickle=False)
        nonfinite = int(np.count_nonzero(~np.isfinite(array)))
        self.tensors[path.name] = {"original_dtype": str(value.dtype), "shape": list(value.shape),
                                   "nonfinite_elements": nonfinite, **file_record(path)}
        if nonfinite:
            raise NonFiniteSourceError("Non-finite source output: " + name)

    def inputs(self, name, args, kwargs):
        if name in ("encoder", "language", "vision"):
            self.counts[name + "_inputs"] += 1
        values = {"arg_" + str(index): value for index, value in enumerate(args) if isinstance(value, torch.Tensor)}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                values[key] = value
            elif isinstance(value, (list, tuple)) and all(isinstance(item, torch.Tensor) for item in value):
                values.update({key + "_" + str(index): item for index, item in enumerate(value)})
        path = self.directory / (name + "-inputs.npz")
        np.savez(path, **{key: self.array(value) for key, value in values.items()})
        self.tensors[path.name] = {"original_dtypes": {key: str(value.dtype) for key, value in values.items()},
                                   "shapes": {key: list(value.shape) for key, value in values.items()}, **file_record(path)}
        if name == "encoder" and self.expected_inputs is not None:
            if not set(self.expected_inputs) <= set(values):
                raise ValueError("The original source encoder omitted a frozen input")
            for key, expected in self.expected_inputs.items():
                expected = torch.from_numpy(expected).to(dtype=values[key].dtype)
                if not np.array_equal(self.array(expected), self.array(values[key])):
                    raise ValueError("The original source encoder input differs from its frozen fixture: " + key)

    def text(self, module, args):
        self.counts["text"] += 1
        self.output("language-reference", args[0])

    def vision(self, module, args, result):
        self.counts["vision"] += 1
        if len(result.deepstack_features) != 3:
            raise ValueError("The source vision model must return all three deepstack features")
        for name, value in zip(("merged", "deepstack_0", "deepstack_1", "deepstack_2"),
                               (result.pooler_output, *result.deepstack_features), strict=True):
            self.output("vision-" + name, value)

    def prepare(self, images, batch, channels, height, width, requested_dtype, device, generator, latents=None):
        if batch != 1 or channels != 64 or height != 512 or width != 512 or (images is not None and len(images) != 1):
            raise ValueError("Source precision boundaries require the fixed one-image profile")
        target, _ = self.originals["prepare"](None, batch, channels, height, width, self.dit_dtype, device, generator, latents)
        condition = None
        if images is not None:
            encoded = self.pipeline._encode_vae_image(images[0].to(device=device, dtype=self.vae_dtype), generator)
            condition = self.pipeline._pack_latents(encoded, batch, channels, encoded.shape[3], encoded.shape[4]).to(self.dit_dtype)
        if target.dtype != self.dit_dtype or list(target.shape) != [1, 1024, 64]:
            raise ValueError("The source initial latent did not use the declared denoiser precision")
        return target, condition

    def transformer(self, *args, **kwargs):
        index = self.counts["denoiser"]
        if args or kwargs.get("kv_cache") is not None or kwargs.get("kv_cache_mode") is not None:
            raise ValueError("Source capture requires the fixed stateless keyword invocation")
        values = dict(kwargs)
        for name in ("hidden_states", "encoder_hidden_states", "timestep"):
            values[name] = values[name].to(self.dit_dtype)
        self.inputs("transformer-" + str(index), (), values)
        save_json(self.directory / f"transformer-{index}-layout.json", {"img_shapes": values["img_shapes"], "prefix_kv_cache": False})
        result = self.originals["transformer"](**values)
        self.output("joint-noise-" + str(index), result[0])
        self.counts["denoiser"] += 1
        return result

    def encode(self, value, *args, **kwargs):
        self.counts["vae_encoder"] += 1
        if value.dtype != self.vae_dtype:
            raise ValueError("The source VAE image has the wrong precision")
        self.output("vae-encode-input", value)
        result = self.originals["encode"](value, *args, **kwargs)
        self.output("vae-encode-output", result.latent_dist.mode())
        return result

    def decode(self, value, *args, **kwargs):
        self.counts["vae_decoder"] += 1
        if value.dtype != self.vae_dtype:
            raise ValueError("The source VAE latent has the wrong precision")
        self.output("vae-decode-input", value)
        result = self.originals["decode"](value, *args, **kwargs)
        self.output("rgba", result[0])
        return result

    def step(self, model_output, timestep, sample, *args, **kwargs):
        index = self.counts["scheduler"]
        if sample.dtype != self.dit_dtype or model_output.dtype != self.dit_dtype or self.counts["denoiser"] != index + 1:
            raise ValueError("The source scheduler input precision or step order changed")
        self.output("step-latent-input-" + str(index), sample)
        self.output("step-noise-input-" + str(index), model_output)
        result = self.originals["step"](model_output, timestep, sample, *args, **kwargs)
        self.output("step-raw-output-" + str(index), result[0])
        self.steps.append({"index": index, "timestep": float(timestep), "before_dtype": str(sample.dtype),
                           "noise_dtype": str(model_output.dtype), "scheduler_result_dtype": str(result[0].dtype)})
        self.counts["scheduler"] += 1
        return result

    def callback(self, pipeline, index, timestep, values):
        if index != self.counts["callback"] or self.counts["scheduler"] != index + 1 or values["latents"].dtype != self.dit_dtype:
            raise ValueError("The source callback precision or step order changed")
        self.output("latent-" + str(index), values["latents"])
        self.steps[index]["callback_dtype"] = str(values["latents"].dtype)
        self.counts["callback"] += 1
        return values

    def finish(self):
        editing = int(self.operation == "edit")
        expected = {"vision": editing, "text": 1, "denoiser": 8, "vae_encoder": editing,
                    "vae_decoder": 1, "scheduler": 8, "callback": 8,
                    "encoder_inputs": 1, "language_inputs": 1, "vision_inputs": editing}
        if self.counts != expected:
            raise ValueError("The source trace does not contain every required component and all eight steps")
        return self.snapshot(complete=True)

    def snapshot(self, *, complete=False):
        layouts = {f"transformer-{index}-layout.json": file_record(self.directory / f"transformer-{index}-layout.json")
                   for index in range(8) if (self.directory / f"transformer-{index}-layout.json").is_file()}
        record = {"complete": complete, "counts": self.counts, "steps": self.steps, "tensors": self.tensors, "layouts": layouts,
                  "not_applicable": [] if self.operation == "edit" else [{"component": name, "reason": "generation has no reference image"}
                                                        for name in ("vision", "vae_encoder")]}
        save_json(self.directory / "trace.json", record)
        return record

    def __enter__(self):
        pipeline = self.pipeline
        for name, module in (("encoder", pipeline.text_encoder), ("language", pipeline.text_encoder.model.language_model),
                             ("vision", pipeline.text_encoder.model.visual)):
            self.handles.append(module.register_forward_pre_hook(
                lambda module, args, kwargs, name=name: self.inputs(name, args, kwargs), with_kwargs=True))
        self.handles.append(pipeline.text_encoder.model.language_model.norm.register_forward_pre_hook(self.text))
        self.handles.append(pipeline.text_encoder.model.visual.register_forward_hook(self.vision))
        pipeline.prepare_latents, pipeline.transformer.forward = self.prepare, self.transformer
        pipeline.vae.encode, pipeline.vae.decode, pipeline.scheduler.step = self.encode, self.decode, self.step
        return self

    def __exit__(self, *error):
        for handle in self.handles:
            handle.remove()
        pipeline = self.pipeline
        pipeline.prepare_latents, pipeline.transformer.forward = self.originals["prepare"], self.originals["transformer"]
        pipeline.vae.encode, pipeline.vae.decode, pipeline.scheduler.step = self.originals["encode"], self.originals["decode"], self.originals["step"]
        self.handles.clear()
        self.originals.clear()
        self.pipeline = None
