# JEV-27B typed decisions (draft)

**Status: draft, not published.** This page describes the typed-decision bundle that the
[`jev27b` conversion recipe](../Tools/Conversion/recipes/jev27b/README.md) builds from
[autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B). The bundle is not distributed. The output check of the
merged checkpoint (stage 1 against stage 1', largest probability difference at most 0.01) has not passed and awaits a
decision; the bundle and the quantized references below are a limited exploration that uses that merged checkpoint. No
quality, fidelity or calibration result is claimed here.

JEV-27B answers typed questions (`noul`, `choice` with 2 to 16 options, `score` from 0 to 5) with one prefill of a bare
template. The final hidden state of the last token goes through a 24-row decision head. The bundle keeps that split:
the Core AI asset returns hidden states, and the head is applied in float32 outside the asset.

## Bundle layout

```text
jev27b_s1_qwen3_5_cfg_dynamic/
  jev27b_s1_qwen3_5_cfg_dynamic.aimodel/   main only: hidden states after the final RMSNorm
  metadata.json                            written by the exporter; source.revision is the merged revision
  tokenizer/                               tokenizer.json, tokenizer_config.json, chat_template.jinja
  head.safetensors                         proj.weight [24, 5120] and proj.bias [24], float32
  decision.json                            the typed-decision-v1 contract below
  LICENSE                                  Apache License 2.0
  NOTICE.md                                attribution and the changes made
  SHA256SUMS                               SHA-256 of every other file
```

- The asset has one function, `main`. Inputs: `input_ids` and `position_ids` (Int32, 1 x ?). Output: `hidden_states`
  (Float16, 1 x ? x 5120). States: `keyCache` and `valueCache` (BFloat16, 16 x 1 x 4 x ? x 256), `convStates`
  (BFloat16, 48 x 1 x 4 x 10240) and `recurrentStates` (BFloat16, 48 x 1 x 48 x 128 x 128). There is no `prefill`
  function and no `logits` output; the language-model head is not part of the asset.
- `metadata.json` still says `"kind": "llm"` because the exporter writes it for every language-model export. Tools that
  open the bundle identify it by `decision.json` and by the output name `hidden_states`.
- `head.safetensors` and the tokenizer files are byte-for-byte copies of the files distributed with JEV-27B. The
  tokenizer is never saved again.

Check a copy of the bundle from inside its folder:

~~~sh
shasum -a 256 -c SHA256SUMS
python3 -m Tools.Conversion.adapters.jev27b.decision <bundle>/decision.json --check-files
~~~

The second command runs from the repository root, needs no third-party package, and checks the schema below and the
size and SHA-256 of every listed file.

## decision.json (typed-decision-v1)

| Field | Content |
|---|---|
| `format` | `typed-decision-v1` |
| `backbone` | `asset` (folder name of the `.aimodel`), `function` (`main`), `inputs` (`input_ids`, `position_ids`), `output` (`hidden_states`), `hidden` (`dtype` `float16`, `size` 5120), `states` (the four state names in order), `context_length` (262144, the exported limit) |
| `head` | `file` (`head.safetensors`), `weight` (`proj.weight`), `bias` (`proj.bias`), `weight_shape` [24, 5120], `bias_shape` [24], `dtype` `float32`, `sha256` |
| `slots` | Head rows per kind as `[start, end)`: `noul` [0, 2), `score` [2, 8), `choice` [8, 24). The ranges are disjoint and cover all 24 rows. A choice with n options uses the first n rows of its range. |
| `verbalizers`, `verbalizer_ids` | Copied unchanged from the distributed `judge_config.json`: the 24 verbalizer strings and their token IDs. |
| `template` | `version` `bare-v1`; `text` with the placeholders `{kind}`, `{state}`, `{question}` and `{options}`; `option_separator` (a line feed); `choice_line` `{label}) {option}` with `labels` A to P; `fixed_options` for `noul` (`false`, `true`) and `score` (`0` to `5`); `add_special_tokens` false; `last_token` `]:` |
| `temperatures` | `per_kind` (the `per_kind` values of the distributed `calibration.json`, unchanged, all above 0), `calibration_id` (`<model>@<revision>/calibration.json`), `calibration_sha256`, `source` (`card`) |
| `limits` | `options` 16 and `input_tokens` 8192 (the whole rendered input, template included) |
| `ties` | `lowest-index`: equal probabilities select the earliest option |
| `source` | `model` and `base_model` (ID and revision), `adapter_sha256`, `adapter_config_sha256`, `head_sha256`, `merged_model_id`, `merged_revision`, `upstream_revision` (apple/coreai-models), `compression`, `compute_precision` |
| `files` | `path`, `bytes` and `sha256` of every bundle file except `decision.json` and `SHA256SUMS`, sorted by path |

A decision is computed as follows: render the template, tokenize it with the bundled tokenizer without special tokens,
run `main` once from zero states, read the hidden state of the last position (the token `]:`), compute
`z = proj.weight @ h + proj.bias` in float32, take the slot range of the kind, and apply a softmax to `z / T` over the
options of the question. Temperature 1 gives the uncalibrated probabilities; `per_kind` gives the shipping ones.

## Reference stages

| Stage | Model | Precision | Where |
|---|---|---|---|
| 0 | Distributed backbone with the decision head, adapter disabled | bfloat16 | `reference` |
| 0q | Stage 0 quantized with the export recipe, computed in float32 | float32 | `probe_base` |
| 1 | Backbone with the unmerged LoRA adapter (PEFT) | bfloat16 | `reference` |
| 1' | Merged checkpoint | bfloat16 | `reference_merged` |
| 2 | Merged checkpoint quantized with the export recipe, computed in float32 | float32 | `probe_merged` |
| 3 | Core AI asset run from Swift | asset | not part of this recipe |

Stages 2 and 0q are built in their own processes through the exporter's own loading and quantization path. Every tensor
of the quantized model (quantized data, scales, offsets and unquantized weights) is hashed before any float32 cast.
Stage 2 must reproduce the hashes recorded during the export; stage 0q has no exported counterpart and keeps its own
hash record. The quantized models then run in float32 with float32 states.

`verify_quantization` writes `reference_q.json`: one row per input of F (stages 0, 1, 1', 2 and 0q) and Q (stages 2 and
0q) with `id`, `kind`, `options`, `band`, `row_sha256`, `text_sha256`, `token_count`, `tokens`, `rejected`, and for each
stage `logits<stage>` (24 values before the temperature), `p<stage>` (temperature 1), `p<stage>_T` (shipping
temperature) and `elapsed<stage>_s`; its header gives the device and dtype of each stage (`stages`). Rows longer than
8,192 tokens are rejected without a forward pass. Hidden states
of F are kept in `hidden2.safetensors` and `hidden0q.safetensors` (`hidden` [302, 5120] float32 in the order of F,
`mask` [302] uint8 for rows with a forward pass). `quantization.json` (`typed-decision-quantization/1`) records the
differences of stage 2 from stage 1' and of stage 0q from stage 0. Its `passed` covers hash agreement, complete rows
and finite probabilities only; no numeric threshold is applied at this step.

## Terms

Qwen3.8-27B (Qwen) and JEV-27B (AutoTrust AI) are Apache-2.0. JEV-27B was trained on
`SargeDev/jev-distill-corpus-v3` (Apache-2.0), most of whose labels are output distributions of TypeSafe Jev 1.13. The
fixed sets are built from that corpus and from JGLUE (CC BY-SA 4.0) and are not redistributed.
