# Image Research 512: macOS validation

The fixed FP32 profile completed text-to-image generation and one-image editing
with all five learned networks running through Core AI. Both eight-step outputs
passed the predeclared comparison against the independent original FP32 MPS
pipeline. A packaged-bundle CLI run reproduced the generated latent, RGBA array,
and PNG exactly. These results cover two fixed inputs on an M4 Max with 128 GiB.

The [numeric record](results.json) contains component comparisons, source-condition
comparisons, timings, and memory counters. The [pipeline summary](pipeline-summary.json)
was produced by `Tools.Conversion.summary`; raw logs, commands, prompts, and local
paths are not published here. The bundle also records source file hashes, code
hashes, dependency versions, fixture/noise hashes, and the component inventories
used by its verification.

## Profile and source

- macOS 27.0 (26A428), Xcode 27.0, M4 Max 128 GiB.
- Qwen-Image-2.1-Turbo revision `fc13ba1986fcb905854dc8de2f52eee395f49b5b`.
- Diffusers `b800f828f3951c4d206e154411bcbfa693cd322e`, Transformers 5.18.0,
  PyTorch 2.9.0, coreai-torch 0.4.3, coreai-core 1.0.0b3.
- FP32, batch one, 512 by 512 pixels, CFG one, eight saved sigmas, one optional
  512-square reference image, stateless denoiser with prefix recomputation.
- Text input capacity 512 tokens; joint denoiser capacity 2304 tokens. Runtime
  masks exclude padding keys and preserve each image block's attention boundary.
- Original reference runs capture inputs, intermediate values, and output tensors.
  Original FP32 cache-on/off and BF16 cache-on are separate conditions.

The full bundle contains 41 inventoried files totaling **62,370,841,163 bytes**
(62.37 GB, excluding `bundle.json`). This is a logical file total, not a RAM or
exclusive GPU memory requirement. The FP32 text asset omits the unused LM head.
The source files total approximately 32.44 GB at BF16.

## Fidelity

All 11 component output comparisons passed both the source-to-wrapper and
source-to-Core-AI checks: 22 passing comparisons across vision, text, denoiser,
VAE encoder, and VAE decoder. The component limits were normalized maximum
error 1e-3 and normalized RMSE 1e-4, with exact shapes and finite outputs.

| Full eight-step case | Latent normalized RMSE | RGBA absolute RMSE |
| --- | ---: | ---: |
| Generate | 1.732044e-6 | 1.335857e-6 |
| Edit | 6.764548e-7 | 3.954765e-7 |
| Predeclared limit | 5e-3 | 1e-2 |

Original FP32 cache-on and cache-off produced identical final latents and RGBA
arrays in both cases. With cache-on in both runs, BF16 versus FP32 had latent
normalized RMSE 0.013741/0.006965 and RGBA normalized RMSE 0.017764/0.008333
for generation/editing. Those precision-profile differences are descriptive;
they are not Core AI conversion-error gates.

The input and converted outputs were inspected for the blue teapot generation
and red-to-blue circle edit. User acceptance of image appearance remains pending.
This is not a general image-quality benchmark or a test of Japanese typography,
all resolutions, multiple reference images, or transparent-reference edge cases.

## Time and memory

The fresh eleven-stage workflow took **1131.57 seconds**, including identity
hashing, three reference conditions, conversions, validation, and packaging.
This excludes the earlier source download, implementation, and prototype work.

| Core AI case | Diagnostic pipeline time | Model loads within that time |
| --- | ---: | ---: |
| Generate | 163.63 s | 123.42 s |
| Edit | 207.24 s | 148.77 s |

These case times include component loading, preprocessing, inference, and tensor
capture. Models are released between components, with the denoiser retained for
its eight calls. The observed load cost dominates; this implementation prioritizes
fidelity and reproducibility, and is not yet an interactive-performance profile.
First calls after a load and subsequent calls are recorded separately. The
specialization cache may already have been populated by component validation.

The full verification process recorded maximum RSS **110,662,795,264 bytes**
and peak footprint **7,751,065,968 bytes** using `/usr/bin/time -l`. These counters
measure different things and must not be added together or read as exclusive
GPU memory. In particular, the lower footprint does not establish iPhone fit.

Reference and converted timings are not a controlled speed comparison: precision,
padding, caching, model residency, and diagnostic work differ. The standalone
bundle check also overlapped a disk-hashing cache audit, so its time is not a
clean benchmark.

## Reuse and scope

The next workflow run reused all eleven stages from the original run and completed
with verification passed. Reused results retain their original run identity;
this audit did not perform new model measurements. Partial or failed artifacts
are not accepted as completed stages.

The packaged CLI checks file inventories and fingerprints before loading models.
CPU regression tests cover changed same-size model/config files, incomplete
inventories, nonempty output preservation, and SIGTERM lock cleanup using a
stubbed generator. Real GPU cancellation, sustained memory behavior, thermals,
and device-specific recovery remain unmeasured.

Use the [recipe instructions](../../../Tools/Conversion/recipes/qwen_image21/README.md)
to reproduce the workflow. [The mobile investigation](../../qwen-image21-mobile.md)
separates native-host work, compression and memory candidates, and physical-device
gates. iOS/iPadOS execution, Swift integration, and quantization are not validated
by these Mac results.

The sample code and model assets retain separate license terms. The model and its
derived assets use the Qwen Research License for non-commercial research and
evaluation. The workflow creates local bundles and does not publish model weights.
