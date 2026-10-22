# Qwen Image 2.1 Turbo: mobile feasibility

A native iOS/iPadOS port using the same checkpoint is worth investigating, but **mobile feasibility is unproven**. Replacing the Python host with Swift does not by itself resolve memory use. The first candidate should combine a fixed 512 × 512 workload, one image per request, staged model loading, and separately evaluated weight compression.

As of October 22, 2026, the draft [native runtime and diagnostics](https://github.com/oka-yuji/coreai-samples/pull/284), [local import](https://github.com/oka-yuji/coreai-samples/pull/282), and [generation UI](https://github.com/oka-yuji/coreai-samples/pull/285) have implementation and CPU/build evidence. Real-model acceptance, device calibration, normal iPhone operation, and human image-quality review remain incomplete. The [parent issue](https://github.com/oka-yuji/coreai-samples/issues/222) tracks those separate boundaries.

The W4/F32 candidate completed all five exports and sixty source runs, but failed 32 of 1,230 fixed component comparisons. All 480 target-region comparisons passed; the frozen full-joint failures still prevent acceptance. Attention segmentation and RoPE arithmetic controls did not resolve those failures. The [numerical diagnostic record](research/qwen-image21-w4-f32-diagnostic-20261022.json) preserves the failed candidate, partial rollout, exact comparison scope, and next checks. The earlier Mac evidence below retains its original scope.

Research date: **2026-10-18**. Target: [Qwen/Qwen-Image-2.1-Turbo at `fc13ba1`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/tree/fc13ba1986fcb905854dc8de2f52eee395f49b5b). There has been **no iPhone or iPad inference, memory, thermal, or quality validation** for this port.

The 2026-10-20 plan in [#222](https://github.com/oka-yuji/coreai-samples/issues/222) starts with a small real-weight graph on the target iPhone 17 Pro, alongside [evaluation contracts](qwen-image21-evaluation.md) and shared model residency. The older investigation sequence below is historical. Device execution and full-component fit remain unverified.

## Current evidence and implementation boundary

| Area | Evidence at the research cutoff | What remains unverified |
| --- | --- | --- |
| macOS DiT | Full source weights, FP32, fixed 512 × 512 generation/editing inputs: original → wrapper → Core AI comparison passed | Broader image quality and mobile execution |
| macOS VAE | Source-weight encoder and decoder comparisons passed at 512 × 512 | Mobile memory and tiled execution |
| Complete pipeline | All five Core AI components completed generation and editing; both final outputs passed independent same-FP32 comparisons; packaged CLI reproduced one result exactly | User image acceptance, broader quality, mobile execution |
| Native/mobile host | Responsibilities identified below | Swift implementation and every physical-device check |

[macOS implementation #186](https://github.com/oka-yuji/coreai-samples/issues/186) uses a **Python host and a stateless, cache-off DiT**. The [Mac validation record](research/qwen-image21-20261018/README.md) fixes its inputs, thresholds, and measured scope. A Swift host and mobile optimization are subsequent work; Mac results do not establish iOS support.

The source-reading baseline is [Diffusers `b800f828`](https://github.com/huggingface/diffusers/tree/b800f828f3951c4d206e154411bcbfa693cd322e), Transformers **5.18.0**, Apple [`coreai-models` `5de95c6`](https://github.com/apple/coreai-models/tree/5de95c6cd29e5e390fb36d87e13b2396668f7974), and the installed **iPhoneOS 27.0 SDK**. These are source and API checks, not device measurements.

## Memory: weights and prefix KV are separate

The pinned model contains these source-weight files. Their sizes are **not runtime RAM requirements**; component loading, activations, caches, and specialization add separate costs. Components need not remain resident together. [Pinned distribution](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/tree/fc13ba1986fcb905854dc8de2f52eee395f49b5b)

| Component | Source file bytes | Decimal GB |
| --- | ---: | ---: |
| Qwen3-VL text/vision encoder | 17,534,340,584 | 17.53 |
| Image transformer | 14,230,284,408 | 14.23 |
| VAE | 675,508,656 | 0.68 |
| Total | 32,440,133,648 | 32.44 |

The completed FP32 Mac verification recorded maximum RSS 110,662,795,264 bytes and peak footprint 7,751,065,968 bytes. Its bundle is 62.37 GB in logical files. The counters differ, and the lower footprint does not establish mobile fit. Component release and memory recovery still require device-specific measurement.

An idealized 16-to-4-bit division gives about **8.11 GB** for all weights, or **8.62 GB** if the VAE stays at 16 bits. These are arithmetic comparisons, not achieved artifact sizes or RAM guarantees. Scales, lookup tables, alignment, uncompressed layers, activations, and KV are excluded. Quantization quality has not been evaluated.

The optional DiT prefix cache stores text and reference-image K/V after RoPE. For batch 1, one guidance branch, and 16-bit values, its tensor payload is:

```text
32 layers × 2 (K, V) × prefix tokens × 32 heads × 128 dimensions × 2 bytes
= 524,288 bytes per prefix token
```

| Actual reference-image size | Image latent tokens at 16× spatial compression | Image-only prefix KV payload |
| --- | ---: | ---: |
| 512 × 512 | 1,024 | 512 MiB |
| 1,024 × 1,024 | 4,096 | 2 GiB |
| 2,048 × 2,048 | 16,384 | 8 GiB |

These are **theoretical tensor payloads**, not measured peak RAM. Text tokens, temporary Q/K/V, concatenation, and other buffers are additional. Multiple references add tokens; an additional CFG branch has its own cache. Weight-only quantization does not compress this cache. [Cache layout](https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/models/transformers/transformer_qwenimage21.py#L56-L84), [Q/K/V storage](https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/models/transformers/transformer_qwenimage21.py#L330-L368), [model configuration](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/transformer/config.json)

**Set the reference resolution explicitly.** In this pipeline, `output_resolution` defaults to **1024** and determines the reference resize area separately from output `width`/`height`. Setting only `width=512, height=512` does not reduce the reference to 512. Record all three settings, actual resized dimensions, image grids, and latent-token counts. [Signature and preprocessing](https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L511-L670)

## Native host responsibilities

The host should have explicit tensor boundaries so the same fixtures can compare Python, native Mac, and physical-device execution.

| Boundary | Contract to preserve |
| --- | --- |
| Tokenization | Exact raw prompt template, system-token removal, image-token positions, left padding, and masks; verify token IDs before inference. |
| Reference processing | Use matching resize dimensions. Composite RGBA over white for the vision encoder; retain four channels for the VAE. |
| Encoder output | Take the last decoder hidden state **before final RMSNorm**. Preserve Qwen3-VL DeepStack injection and multimodal positions. |
| DiT inputs | Preserve the interleaved text/image sequence, four latent tokens per vision-language image slot, block-causal attention, and RoPE. |
| Optional prefix cache | Separate initial extraction from later reuse; retain each layer's post-RoPE K/V, and reset on every new request. |
| Latents/VAE | Preserve spatial flattening, posterior mode for reference encoding, per-channel latent normalization and its inverse, and RGBA output. |

The [pinned pipeline](https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L205-L450) and [DiT input construction](https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/models/transformers/transformer_qwenimage21.py#L922-L995) define these boundaries. DeepStack and multimodal positions belong to the encoder graph; they cannot be replaced with a plain text-only Qwen encoder for editing.

Use the checkpoint's **eight saved sigmas**, append the scheduler's terminal zero, and retain CFG=1 for the initial comparison. Its scheduler configuration has shift=1, dynamic shifting off, and stochastic sampling off. Euler updates run in FP32 before casting back to model precision. A native RNG with the same seed is not sufficient evidence of identical noise: initially supply the exact reference noise tensor. [Saved schedule](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/model_index.json), [scheduler configuration](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/scheduler/scheduler_config.json), [scheduler implementation](https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L493-L526)

The current compared [`TextEncoder` wrapper](../Tools/Conversion/adapters/qwen_image21/encoders.py) already selects language-model embeddings/layers and omits the vocabulary head and text-generation KV. The source's 151,936 × 4,096 untied head is 622,329,856 parameters, about 1.245 GB at BF16; removing it again is not a new mobile saving. The historical proposal to omit it was superseded by the existing FP32 wrapper and its comparisons. [Encoder configuration](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/text_encoder/config.json), [hidden-state consumption](https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L303-L337)

## Changes to investigate, in order

| Order | Candidate | Required evidence |
| --- | --- | --- |
| 1 | Hidden-state-only encoder; load and release encoder, VAE encoder, DiT, and VAE decoder by phase | Matching intermediate outputs and measured memory recovery after release |
| 2 | Batch 1, CFG=1, 512 output and reference, one reference image, bounded prompt length | Fidelity against the original at the same settings; separate quality evaluation of the lower resolution |
| 3 | Evaluate 16-bit execution, then mixed 8/4-bit weights | Separate precision/quantization loss from conversion error; inspect text rendering, editing adherence, details, and alpha |
| 4 | Compare prefix cache off/on | Whole-request latency and peak memory; off recomputes the prefix each step, on retains it |
| 5 | Spatially tiled VAE | Full/tiled comparison including seams, color, alpha, overlap, and edge-tile shapes |
| 6 | Stream small groups of text/DiT blocks if still necessary | Actual weight residency, repeated loading/specialization, I/O, energy, and total time across all eight steps |
| 7 | Explore Neural Engine placement and architecture-specific AOT | Device output checks and verified placement; successful execution alone is insufficient |

FP16 is not merely a smaller storage choice: the source DiT clips block outputs to ±65,504 only on its FP16 path. Its zero-centered RMSNorm computes `weight + 1` in FP32. Preserve those semantics when building the precision reference. [Norm](https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/models/transformers/transformer_qwenimage21.py#L170-L185), [FP16 branch](https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/models/transformers/transformer_qwenimage21.py#L646-L652)

The upstream VAE already has spatial tiled encoding/decoding, disabled by default, with tile size 256 and stride 192. Batch slicing does not reduce spatial memory for batch 1. Porting tiling still requires native buffer-lifetime and boundary checks. [VAE implementation](https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py#L1177-L1235)

Apple's FLUX.2 example provides a packaging precedent: iOS defaults to 512, a single-function transformer, and a smaller reference grid. Its multi-function asset shares disk weights but has a higher runtime peak; the Swift host also supports staged resource release. This is evidence for experiments, not proof that Qwen fits or that the existing FLUX host accepts it. [Apple packaging](https://github.com/apple/coreai-models/blob/5de95c6cd29e5e390fb36d87e13b2396668f7974/models/flux2/README.md#L29-L58), [resource release](https://github.com/apple/coreai-models/blob/5de95c6cd29e5e390fb36d87e13b2396668f7974/swift/Sources/CoreAIDiffusionPipeline/Components/CoreAIDiffusionModelFunction.swift#L27-L49)

Start with GPU correctness. Do not generalize the official LLM exporter's static-shape rules to every diffusion asset. The iOS 27 SDK declares BF16 arrays, but that declaration does not establish whole-model or Neural Engine support. Moving tensors to the CPU also does not automatically reduce total use on unified-memory hardware; measure what is actually released.

## Physical-device sequence and decision gates

First reproduce the Python tensor contracts in a native Mac host. Then test text-to-image followed by one-reference editing on an **Apple Silicon iPad with a larger measured app memory budget**, before moving to the target iPhone. Device family or total RAM alone is not an acceptance criterion.

For each device, record model identifier, OS build, asset hashes, precision, shape, token counts, cache mode, and power/thermal state. Separate specialization, loading, encoder, each DiT step, VAE, and total time. Measure uncached cold, cached cold, warm, repeated requests, cancellation/retry, and background/foreground transitions. Record RSS/residency, kernel peak physical footprint, and available app memory; a periodically sampled maximum can miss a short peak.

`os_proc_available_memory()` reports the current app limit minus its footprint, not total device RAM, and its result must not be cached. The increased-memory entitlement only helps on supported devices and does not guarantee extra capacity. [Apple memory API](https://developer.apple.com/documentation/os/os_proc_available_memory), [entitlement](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit)

| Decision | Gate |
| --- | --- |
| Proceed for a named device/configuration | Generation and editing pass declared numerical and visual criteria; cold/warm/repeated execution fits the measured budget with headroom; latency, heat, and cancellation behavior meet the intended use. |
| Limit the claim | If only 512 text-to-image succeeds, publish evidence for that workload. Editing, 2K output, and other devices remain unverified. |
| Hold the port | Minimum graphs are incorrect; compression fails quality criteria; phase peaks exceed the device budget; or streaming/loading makes time or thermal behavior unacceptable. |

Freeze numerical, quality, latency, and headroom thresholds before device testing, using the Mac measurements and intended workload. This research does not invent device limits or predict generation times from weight sizes.

## Same-checkpoint port versus a different model

Staged loading, equivalent graph rewrites, and native host code can preserve the checkpoint's computation. Quantization and lower resolution require their own fidelity and quality records. A smaller replacement encoder is not interchangeable merely because its output shape matches; distillation, pruning, or retraining is a separate model-development task. Choosing FLUX or another generator is a model substitution. Bundled prompt embeddings restrict prompt choice, and a remote encoder would make the system partly remote.

Native implementation does not remove the model's distribution conditions. Keep repository code and model/derived-asset licensing distinct, and preserve the [pinned Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/LICENSE) requirements when preparing any research distribution.
