# TeleOCR macOS FP16 assets

This maintainer recipe converts [XingChen-AGI/TeleOCR](https://huggingface.co/XingChen-AGI/TeleOCR) at revision `70a19ec6a3b036311af19ae5e762e650f41a5980` into three Core AI assets for macOS and checks the parts that the Python runtime can verify. It does not run the Swift OCR path, generate text, quantize, build iOS shapes or publish weights.

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

- Vision: the FP32 export of the same dynamic-window graph must match the original vision encoder with normalized RMSE ≤ 1e-4 on the five required inputs. The FP16 asset records the float16 difference (PyTorch FP32 vs FP16) and the conversion difference (PyTorch FP16 vs Core AI) separately.
- Embedding: the F1 prompt tokens and a single token must match the original `embed_tokens` with maximum absolute difference 0.
- Decoder: a stateless FP16 prefill of the 448² F1 prompt (S = 286) must keep the original last-position argmax; the maximum difference and the float16 difference are recorded. The Python runtime is not used for the stateful functions; their execution is verified from Swift with `expectFrequentReshapes`.

```sh
.local/conversion/teleocr/.venv/bin/python -m unittest discover -s Tools/Conversion/tests -v
```

TeleOCR is distributed under Apache-2.0. The bundle carries the license text and a notice describing the changes. This repository's code is MIT-licensed.
