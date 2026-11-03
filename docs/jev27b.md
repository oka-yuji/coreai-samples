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

Inputs follow the reference renderer exactly. A `choice` has 2 to 16 distinct options, and an option must not contain a
line break (U+000A, U+000B, U+000C, U+000D, U+001C, U+001D, U+001E, U+0085, U+2028 or U+2029); its lines use
`choice_line` with the labels in order. `noul` and `score` take only their `fixed_options`, written one per line without
labels. Each placeholder of `text` is replaced once, with the value inserted as it is; text that looks like a placeholder
inside a value is not replaced again. Inputs that break these rules are rejected rather than rendered differently.

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
hash record. The quantized models then run in float32 with float32 zero states; the key and value caches hold
max(T, 2048) positions for an input of T tokens.

`verify_quantization` writes `reference_q.json`: one row per input of F (stages 0, 1, 1', 2 and 0q) and Q (stages 2 and
0q) with `set` (`F` or `Q`), `id`, `kind`, `options`, `band`, `row_sha256`, `text_sha256`, `token_count`, `tokens`,
`rejected`, and for each stage `logits<stage>` (24 values before the temperature), `p<stage>` (temperature 1),
`p<stage>_T` (shipping temperature) and `elapsed<stage>_s`, where `<stage>` is `0`, `1`, `1m` (stage 1'), `2` or `0q`; its header gives the device and dtype of each stage (`stages`). Rows longer than
8,192 tokens are rejected without a forward pass. Hidden states
of F are kept in `hidden2.safetensors` and `hidden0q.safetensors` (`hidden` [302, 5120] float32 in the order of F,
`mask` [302] uint8 for rows with a forward pass). `quantization.json` (`typed-decision-quantization/1`) records the
differences of stage 2 from stage 1' and of stage 0q from stage 0. Its `passed` covers hash agreement, complete rows
and finite probabilities only; no numeric threshold is applied at this step.

## Swift contract

`CoreAIBackend` opens the bundle, renders the template and applies the head without loading the Core AI asset. These
types are separate from the Clef-Flash decision types.

| API | Behavior |
|---|---|
| `TypedDecisionBundle(url:)` | Reads `decision.json` and checks it against the schema above, with the slots exactly as shown and a float16 hidden state; a template that differs from bare-v1 in any key or value is refused. `SHA256SUMS` must list every entry of `files` and `decision.json` with the same digests, the folder must hold exactly the listed files and no links, and the size and SHA-256 of every listed file are computed. Exposes `url`, `name`, `version` (SHA-256 of `decision.json`), `hiddenSize`, `limits` (`options`, `inputTokens`), `temperatures`, `verbalizerIDs`, `headURL`, `tokenizerURL`, `assetURL`, `backboneFunction`, `backboneInputs`, `backboneOutput`, `stateNames` and `contextLength`. Failures throw `TypedDecisionError.invalidBundle`; cancellation passes through. |
| `TypedDecisionRequest` | `state`, `question`, `kind` (`TypedDecisionKind`: `noul`, `choice`, `score`) and `options`. |
| `TypedDecisionRenderer.render(_:)` | The text of the reference renderer. Options that break the input rules throw `invalidOptions`. Options are compared as UTF-8 bytes, so canonically equivalent options with different scalars stay distinct in the text; the bundled tokenizer normalizes to NFC, so their tokens are the same. |
| `TypedDecisionRenderer.tokenIDs(for:tokenizer:limit:)` | Encodes the rendered text without special tokens. More than `limit` tokens throw `tooManyTokens`. |
| `TypedDecisionHead(url:width:)` | Maps `head.safetensors` and checks that it holds `proj.weight` [24, width] and `proj.bias` [24] in float32 inside the file. |
| `logits(hidden:)` | `proj.weight @ h + proj.bias` in float32. A hidden state of another length throws `invalidHidden`. |
| `probabilities(logits:kind:count:temperature:)` | The softmax over the options of the kind, computed in double precision and returned as float32, at the given temperature (`shipping`) and at 1 (`plain`). Non-finite logits and a temperature that is not positive and finite throw `nonFinite`; an option count outside the slots throws `invalidOptions`. |
| `TypedDecisionHead.top(of:)` | The index of the largest value; ties select the earliest. |
| `TypedDecisionAnswer` | `kind`, `options`, `logits`, `probabilities`, `probabilitiesT1`, `top` and `tokenCount` of one decision. |

`swift test --package-path CoreAISamplesKit --filter TypedDecision` runs the rendering, rejection, synthetic head and
bundle tests. With `JEV_BUNDLE_DIR` set to the bundle folder, the tests also compare the token IDs of every fixture row
with the bundled tokenizer, compare the bundled head with stage 1' (logits within 1e-4, probabilities within 1e-5) and
open the bundle, which reads every listed file, including the 16 GB asset, for its SHA-256. None of these tests loads or
runs the Core AI asset.

~~~sh
JEV_BUNDLE_DIR=/path/to/jev27b_s1_qwen3_5_cfg_dynamic swift test --package-path CoreAISamplesKit --filter TypedDecision
~~~

The fixtures are in `CoreAISamplesKit/Tests/CoreAIBackendTests/Fixtures/TypedDecision/`: `render.json` (15 synthetic
English rows with the input boundaries, the two card examples of the model card and 8 JGLUE rows of F with their text,
rendered and tokenized in Python), `reject.json` (19 option lists that the reference renderer refuses), `head.json` (a
head of width 4 with six cases computed with `decide`) and `real-head.json` (four rows of F with their stage 1' hidden
state, logits and probabilities, and token IDs but no text). Regenerate them in the reference environment:

~~~sh
python -m Tools.Conversion.adapters.jev27b.swift_fixture --snapshot <JEV-27B folder> --corpus <corpus folder> \
  --jglue <JGLUE folder> --fixtures Tools/Conversion/recipes/jev27b/fixtures --reference <reference.json> \
  --merged-reference <reference-merged.json> --hidden <hidden.safetensors> \
  --output CoreAISamplesKit/Tests/CoreAIBackendTests/Fixtures/TypedDecision
~~~

The Swift tokenizer (swift-transformers 1.3.4) matches the Split pre-tokenizer pattern on whole grapheme clusters, so
its token IDs differ from those of the Python tokenizer where a pattern boundary falls inside a cluster. Measured
examples are consecutive CRLF line breaks and Hindi and Thai words with combining marks. The fixtures contain no such
text.

## Terms

Qwen3.8-27B (Qwen) and JEV-27B (AutoTrust AI) are Apache-2.0. JEV-27B was trained on
`SargeDev/jev-distill-corpus-v3` (Apache-2.0), most of whose labels are output distributions of TypeSafe Jev 1.13. The
fixed sets are built from that corpus and from JGLUE (CC BY-SA 4.0) and are not redistributed. The Swift test
fixtures contain the text of eight JGLUE rows under CC BY-SA 4.0; their notice is in `render.json`.
