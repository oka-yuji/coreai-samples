# Image Research 512

Convert Qwen-Image-2.1-Turbo into five Core AI components and run text-to-image
and one-reference-image editing on macOS. This is a fixed research profile with
a Python host, not a Swift app or an iOS deployment.

The model assets retain the **Qwen Research License**: non-commercial research
and evaluation only. Independently authored sample code remains MIT-licensed.
The bundled LICENSE and NOTICE.txt explain the model terms and conversion.
Do not infer commercial permission from the sample code license.

The profile uses batch one, 512-square RGBA images, CFG one, the checkpoint's
eight saved sigmas, FP32 computation, and a stateless denoiser. It recomputes the
prefix instead of retaining a denoising KV cache. Prompts may use at most 512
processed tokens, including the system template and image slots. Reference
images must already be 512 by 512 pixels. The command rejects other shapes.

## Environment and inputs

Use macOS 27, Apple Silicon, Xcode 27 with Metal Toolchain, Python 3.11, and uv.
The investigation machine is an M4 Max with 128 GiB. Smaller-memory Macs are
untested. Original weights occupy about 32.44 GB; FP32 assets, specialization
caches, and working outputs require substantial additional disk space. Allow
at least 200 GB free for a fresh full workflow. Model jobs use the machine-wide
heavy-job lock and must run serially.

Run from the repository root. Download the pinned source once into the ignored
workspace. Setup never downloads model weights itself.

```sh
export QI_WORKSPACE="$PWD/.local/conversion"
export UV_PROJECT_ENVIRONMENT="$QI_WORKSPACE/qwen_image21/.venv"
uv sync --frozen --no-dev --project Tools/Conversion/recipes/qwen_image21
"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
workspace = Path(os.environ["QI_WORKSPACE"])
print(snapshot_download(
    "Qwen/Qwen-Image-2.1-Turbo",
    revision="fc13ba1986fcb905854dc8de2f52eee395f49b5b",
    cache_dir=workspace / "qwen_image21/hf-cache",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE", "README.md"],
    max_workers=4,
))
PY
"$UV_PROJECT_ENVIRONMENT/bin/python" -m Tools.Conversion.local setup \
  --recipe qwen_image21 --workspace "$QI_WORKSPACE" \
  --snapshot "$QI_WORKSPACE/qwen_image21/hf-cache/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba1986fcb905854dc8de2f52eee395f49b5b"
"$UV_PROJECT_ENVIRONMENT/bin/python" -m Tools.Conversion.local plan \
  --recipe qwen_image21 --workspace "$QI_WORKSPACE"
"$UV_PROJECT_ENVIRONMENT/bin/python" -m Tools.Conversion.local run \
  --recipe qwen_image21 --workspace "$QI_WORKSPACE"
```

`--precision` on the generic maintainer command belongs to the existing
EmbeddingGemma profiles; this recipe always uses FP32. The exact dependency
versions and Diffusers commit are frozen in `uv.lock`. Source files are checked
against `model.json`, including the four weight files' published SHA256 values.
The setup and run commands use a separate environment from the TTS conversions.

`local run` coordinates the same lock path with the recipe's child processes.
If an agreed alternative machine lock is necessary, supply the same `--lock-dir`
to setup, plan/run, and standalone generation. All other model jobs on the Mac
must use that lock too. An occupied lock is never stolen. Adapter modules are
internal stages and reject model execution without an owning runner lock.

## Outputs and validation

The eleven stages verify source identity, capture an independent original FP32
cache-off reference and separate FP32/BF16 cache-on conditions, export
vision/text/denoiser/VAE encoder/VAE decoder assets,
run the complete Core AI pipeline, and package the verified results. The original
BF16/cache-on path is a distinct comparison condition, not the FP32 conversion
reference. Failed attempts remain separate from completed reusable objects.
Use the final report under `work/qwen_image21/runs/` to locate the bundle stage's
immutable artifact directory. Repeating the command verifies and reuses intact
completed stages; `--refresh` deliberately executes new attempts.

Each component is checked against its source outputs before and after conversion.
The fixed gates are normalized maximum error at most 1e-3 and normalized RMSE
at most 1e-4. Full eight-step trajectories use normalized latent RMSE at most
5e-3 and absolute RGBA RMSE at most 1e-2 on float values in [-1, 1]. These are
fidelity checks on two fixed examples; human inspection and broader image-quality
assessment remain separate. An attractive image alone does not prove fidelity.

The host uses Core AI for every learned network. Python handles the original
processor/tokenizer, position/layout arithmetic, Euler scheduler, and image I/O.
The text asset returns pre-final-RMSNorm hidden states and omits the LM head.
The vision asset includes all three DeepStack outputs. Model components are
loaded and released in sequence, with only the denoiser retained across its
eight calls. Inputs use fixed shapes to bound specialization variants.

Reports distinguish conversion time, load time, first call after load, subsequent
calls, and cache reuse. `/usr/bin/time -l` records peak RSS and footprint in each
model stage's process log. Reference events also record MPS allocator counters.
Tensor capture is enabled, so these timings are diagnostic measurements rather
than a clean performance benchmark. Peak RSS, footprint, MPS allocation, model
file size, and exclusive GPU memory are different quantities.

## Generate from a verified bundle

Set `QI_BUNDLE` to the bundle artifact path reported by the successful run.
The CLI checks the complete bundle inventory before loading models.

```sh
"$UV_PROJECT_ENVIRONMENT/bin/python" -m Tools.Conversion.adapters.qwen_image21.generate \
  --bundle "$QI_BUNDLE" --output "$QI_WORKSPACE/image-output/new-run" \
  --prompt "A blue ceramic teapot on a white table, soft daylight." --seed 20261018
```

Add `--image input-512.png` for editing. `--noise saved-noise.npy` supplies an
exact FP32 [1, 1024, 64] noise tensor; the seed is used only when noise is omitted.
A nonempty output directory from a previous generation is not silently replaced.
The output includes an RGBA PNG, final latents, diagnostic tensors, and timings.
The bundle uses an independent primary name, Image Research 512, with its Qwen
origin retained in the description and notices. Weights are not committed to Git
or automatically uploaded by this workflow.

Run the required conversion tests in the frozen environment:

```sh
"$UV_PROJECT_ENVIRONMENT/bin/python" -m unittest discover -s Tools/Conversion/tests -v
```

For mobile rewriting boundaries, memory arithmetic, and unmeasured conditions,
see [the mobile investigation](../../../../docs/qwen-image21-mobile.md).
