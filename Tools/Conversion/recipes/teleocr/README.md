# TeleOCR macOS assets

This maintainer recipe converts [XingChen-AGI/TeleOCR](https://huggingface.co/XingChen-AGI/TeleOCR) at revision `70a19ec6a3b036311af19ae5e762e650f41a5980` into three Core AI assets for macOS and checks the parts that the Python runtime can verify. It does not run the Swift OCR path, generate text, build iOS shapes or publish weights.

## Setup and run

Use Apple Silicon, macOS 27, Xcode 27, uv and Python 3.11. Download the pinned snapshot once, then:

```sh
hf download XingChen-AGI/TeleOCR --revision 70a19ec6a3b036311af19ae5e762e650f41a5980
python3 -m Tools.Conversion.local setup --recipe teleocr
python3 -m Tools.Conversion.local plan --recipe teleocr
python3 -m Tools.Conversion.local run --recipe teleocr
```

Setup creates `.local/conversion/teleocr/.venv` from the committed `uv.lock` (Torch 2.9.0, Transformers 4.57.3, torchao 0.15.0, coreai-torch 0.4.3, coreai-opt 0.3.0, coreai-core 1.0.0b3) and checks fourteen snapshot files by size and SHA256. With this Torch version an unpinned torchao 0.18.0 breaks the Transformers import, so torchao stays at 0.15.0. `--snapshot` selects another copy of the same revision. Model stages take the shared `~/Library/Caches/coreai-samples/heavy-job` lock.

The run prints `.local/conversion/work/teleocr/runs/<run>/report.json`. Require `status: completed` and `verification: passed`. The `bundle` stage's `bundle` directory contains `vision.aimodel`, `embed.aimodel`, `main.aimodel`, `tokenizer/`, `metadata.json`, `LICENSE` and `NOTICE`. Use `python3 -m Tools.Conversion.summary <report.json>` for shareable numbers; raw reports, comparisons and assets stay under `.local`.

## Asset contract

`metadata.json` records the image normalization, patch and window sizes, special tokens, stop tokens, repetition penalty, prompts and the inputs, outputs and states of every function. The verify stage compares that table with `coreai-build inspect`.

- `vision.aimodel` `main`: `pixel_values` [N, 1176], `key_valid` [N], `cos` and `sin` [N, 80], all float16, N = 64 × windows. The host orders merge units by 4 × 4 windows, keeps empty unit slots at the image edges, sets padding to zero pixels, `key_valid` 0, `cos` 1 and `sin` 0, and omits windows without a valid unit. The graph turns `key_valid` into a 0 / −∞ key mask for both window and full attention layers and returns `image_features` [N / 4, 1024] in input order; the host drops padded rows and restores the source order.
- `embed.aimodel` `main`: `input_ids` [1, S] int32 to `inputs_embeds` [1, S, 1024] float32. The table keeps the original bfloat16 values; float16 would change 46,205 table values.
- `main.aimodel` `prefill` (`inputs_embeds`) and `main` (`input_ids`) share `cos` and `sin` [1, S, 128] with merged mRoPE sections, an additive `mask` [1, 1, S, P + S] and the float16 states `keyCache` and `valueCache` [28, 1, 8, capacity, 128]. Both write keys and values at [P, P + S) with P = mask length − S and return the last-position `logits` [1, 151936]. Rotary positions are inputs, so the KV write position and the mRoPE position stay independent.

RMS normalization is computed after dividing each row by its largest magnitude, which is algebraically equal to the original and keeps squares within the float16 range.

## Verification scope

The reference stage runs the original FP32 model on the synthetic page `F1.png` (from the #178 fixtures) at six vision sizes: 56², 448², 896², 1036² (100 windows, N 6,400), 364 × 1204 and 1204 × 364. Its padded window order must equal the original window index.

- Vision: the FP32 export of the same dynamic-window graph must match the original vision encoder with normalized RMSE ≤ 1e-4 on the five required inputs. The bundled FP16 asset must also stay finite with cosine ≥ 0.999 and normalized RMSE ≤ 0.05 against the original on those inputs; this loose gate only catches broken FP16 conversion. The float16 difference (PyTorch FP32 vs FP16) and the conversion difference (PyTorch FP16 vs Core AI) are recorded separately.
- Embedding: the F1 prompt tokens and a single token must match the original `embed_tokens` with maximum absolute difference 0.
- Decoder: a stateless FP16 prefill of the 448² F1 prompt (S = 286) must keep the original last-position argmax; the maximum difference and the float16 difference are recorded. The Python runtime does not execute the stateful functions; the Swift runtime does. `coreai-verify ocr-block` reads one block and `coreai-verify ocr` reads a page (layout, crops, blocks and OTSL tables) with the bundle, and their outputs are compared with the Python reference (see `Tools/OCREval`).

```sh
.local/conversion/teleocr/.venv/bin/python -m unittest discover -s Tools/Conversion/tests -v
```

TeleOCR is distributed under Apache-2.0. The bundle carries the license text and a notice describing the changes. This repository's code is MIT-licensed.

## Decoder INT8 / INT4 and vision INT8

The quantized recipe keeps the FP16 asset ABI and creates separate decoder INT8,
decoder INT4 and vision INT8 bundles:

```sh
python3 -m Tools.Conversion.local plan --recipe teleocr --precision int4_int8
python3 -m Tools.Conversion.local run --recipe teleocr --precision int4_int8
```

It uses the same environment and pinned snapshot as FP16. Its separate work directory
is `.local/conversion/work/teleocr-quantized`. The bundles are in `bundle_int8`,
`bundle_int4` and `bundle_vision_int8`. Require all three verify stages to pass.

Decoder compression covers exactly the seven attention/MLP Linear weights per layer,
using symmetric-with-clipping integer quantization in blocks of 64 input channels.
The embedding, tied LM head, normalization, biases, activations and KV stay
unquantized. Vision INT8 is a separate experiment with per-output-channel Linear
weights; its Conv3d patch projection stays unquantized, and its bundle keeps the FP16
decoder. This is linear integer compression for Mac, not iOS palettization.

Each compressed stage saves the selected and retained parameter identities in
`compression.json` and independently dequantized weights in `reference.safetensors`.
The reference contains only compressed weights and is applied to the pinned source
wrapper for evaluation; it is not a complete checkpoint. Its SHA-256 is in the
compression record. The bundle metadata contains the compression record and each
component's precision. The verifier checks that the component records match the
bundle and that inspect reports enough integer storage for the selected weights.
Before serialization, every exported function must use exactly the declared
compressed modules with the same integer values, zero points and FP16 scales.
This per-function identity audit is required for both prefill and decode; aggregate
storage counts alone cannot prove that the target set is correct.

`comparison.json` separates the original-model-to-quantized-PyTorch quality difference
from the quantized-PyTorch-to-Core-AI conversion difference. Decoder checks use the F1
448-square prefill; vision checks use the existing six inputs. An argmax divergence
is classified as numerical only within fixed conversion bounds: normalized RMSE
≤ 0.05 and cosine ≥ 0.999 against the independent quantized reference. Within
those bounds the top-two gap is compared with twice the maximum absolute difference
for reporting; that margin inequality alone is not an accuracy gate. Non-finite
outputs fail verification.
Full document quality, generation, performance and memory comparisons use the fixed
OCREval set in issue #333; passing these component checks does not complete that
evaluation or establish iPhone support.

Vision INT8 is verified through a host-only specialization made with
`coreai-build compile --preferred-compute gpu --expect-frequent-reshapes`.
The host architecture comes from the SDK. Python's public specialization options
do not expose the reshape hint; without it this dynamic-window INT8 graph aborts
in ANE region formation even with GPU preferred. The compiled copy stays in
scratch and is not shipped. The bundle retains the source `.aimodel`, and the
Swift OCR runtime already uses the same reshape hint. The comparison record
includes the specialization options, architecture and time.

The [2026-10-27 validation record](../../../../docs/results/teleocr-quantization/2026-10-27.json)
contains the conversion metrics, asset hashes, export audits and Swift checks.
All three candidates finish the F1 448-square block with 212 generated tokens.
INT4 produces different text from INT8; this component check makes no full-document
quality or adoption claim.
