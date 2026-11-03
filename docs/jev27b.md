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

### Variant with a float32 gated delta update

The recipe also exports `jev27b_s1_gdnfp32_qwen3_5_cfg_dynamic` (step `export_gdnfp32`), and the Core AI steps run on
this bundle. It has the same layout, states and outputs. In each of the 48 linear-attention layers, the six inputs of
the composite `gated_delta_update` operation (query, key, value, g, beta and the recurrent state) are cast to float32
before the operation, and its two outputs are cast back to bfloat16. In the first bundle, the operation receives
bfloat16 values. The derived class `TypedDecisionBackboneGdnFp32` keeps the original module as a child of the wrapper,
so the exporter still externalizes it as the same composite operation. Quantization and the quantized weights do not
change: the step requires the per-tensor SHA-256 of the quantized model to equal the record of `export`.

- `variant.json` records the hash comparison, the number of wrapped layers, the input types of every
  `gated_delta_update` call in the IR of both assets (read without the weights), the number of `cast` operations and
  the inspection of the asset. Its `passed` is the check of the step.
- `decision.json` differs from the first bundle only in `backbone.asset` and `files`. `NOTICE.md` adds a line for the
  float32 update, and `provenance.json` records `variant: gdn-fp32` and the key of the `export` object it was checked
  against.

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
| 3 | Core AI asset of the float32 gated delta update variant, run from Swift (`TypedDecisionModel`) | asset | `typed_decide_f` |

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

`swift test --package-path CoreAISamplesKit --filter TypedDecision` runs the rendering, rejection, synthetic head,
bundle and model tests; the model tests use a fake backbone. With `JEV_BUNDLE_DIR` set to the bundle folder, the tests
also compare the token IDs of every fixture row with the bundled tokenizer, compare the bundled head with stage 1'
(logits within 1e-4, probabilities within 1e-5) and open the bundle, which reads every listed file, including the 16 GB
asset, for its SHA-256. None of these tests loads or runs the Core AI asset; `TypedDecisionInferenceTests` (below) does,
only with `COREAI_TYPED_DECISION_BUNDLE`.

~~~sh
JEV_BUNDLE_DIR=/path/to/jev27b_s1_qwen3_5_cfg_dynamic swift test --package-path CoreAISamplesKit --filter TypedDecision
~~~

The fixtures are in `CoreAISamplesKit/Tests/CoreAIBackendTests/Fixtures/TypedDecision/`: `render.json` (15 synthetic
English rows with the input boundaries, the two card examples of the model card and 8 JGLUE rows of F with their text,
rendered and tokenized in Python), `reject.json` (19 option lists that the reference renderer refuses), `head.json` (a
head of width 4 with six cases computed with `decide`), `real-head.json` (four rows of F with their stage 1' hidden
state, logits and probabilities, and token IDs but no text) and `inference.json` (the card examples and JGLUE rows of
`render.json` with their token count, token SHA-256, band and the stage 2 and 1' values of `reference_q.json`).
Regenerate them in the reference environment:

~~~sh
python -m Tools.Conversion.adapters.jev27b.swift_fixture --snapshot <JEV-27B folder> --corpus <corpus folder> \
  --jglue <JGLUE folder> --fixtures Tools/Conversion/recipes/jev27b/fixtures --reference <reference.json> \
  --merged-reference <reference-merged.json> --hidden <hidden.safetensors> \
  --quantized-reference <reference_q.json> --output CoreAISamplesKit/Tests/CoreAIBackendTests/Fixtures/TypedDecision
~~~

The Swift tokenizer (swift-transformers 1.3.4) matches the Split pre-tokenizer pattern on whole grapheme clusters, so
its token IDs differ from those of the Python tokenizer where a pattern boundary falls inside a cluster. Measured
examples are consecutive CRLF line breaks and Hindi and Thai words with combining marks. The fixtures contain no such
text.

## Running on Mac

`TypedDecisionModel` (an actor in `CoreAIBackend`) runs the bundle on Core AI. These types are also separate from the
Clef-Flash decision types; `DecisionRuntime` and `DecisionModel` are unchanged.

| API | Behavior |
|---|---|
| `TypedDecisionModel.load(bundle:)` | Opens the bundle with `TypedDecisionBundle(url:)`, so a bundle that breaks the contract fails before any asset is read. Then it loads the bundled tokenizer, maps the head and specializes `main` with the GPU preferred and `expectFrequentReshapes = true`, reusing the specialization cache entry when there is one. Before specializing it checks that the asset summary lists the `hidden_states` output; afterwards it checks the descriptors: `input_ids` and `position_ids` Int32 1 x ?, `hidden_states` Float16 1 x ? x 5120, the four states of `decision.json`, no interleaved layout and at most one dynamic dimension per state. `loadStats` holds the seconds of each part and whether the specialization came from the cache. |
| `decide(_:)` | Renders and tokenizes the request (rejections throw before the model runs), runs `main` once from zero states and applies the head. Each request allocates its own four states, with the dynamic dimension set to `capacity(forTokens:)`, the token count rounded up to a multiple of 256 and at most `input_tokens`. `position_ids` are 0 to T - 1 without padding, and only the hidden state of the last position is read. Returns a `TypedDecisionAnswer`. |
| One request at a time | A request that arrives while another is running throws `TypedDecisionError.busy`; requests are not queued. The running request keeps the model busy until its forward pass returns, also after its task is cancelled. Cancellation is checked before and after the forward pass and ends the request with `CancellationError`. |
| `tokenIDs(for:)`, `tokenCount(for:)` | The token IDs and their count for a request, with the same rejections as `decide(_:)`. |

`coreai-verify typed-decide` (in `Tools/Verify`) loads the model once and answers the rows of a JSON file one at a time:

~~~sh
swift build -c release --package-path Tools/Verify --product coreai-verify
Tools/Verify/.build/release/coreai-verify typed-decide --bundle <bundle> --input <in.json> --output <out.json> [--limit N]
~~~

- Input: `{"rows": [{"id", "kind", "state", "question", "options"}]}`; other keys are ignored. `--limit N` runs the
  first N rows.
- Output (`typed-decide/1`), written once at the end to a file that must not exist yet:
  `bundle` (`name`, `version` (the SHA-256 of `decision.json`), `asset_sha256` (the `main.mlirb` digest in
  `SHA256SUMS`), `specialization` (`cached` or `fresh`), `load_seconds` (`bundle`, `tokenizer`, `head`, `backbone`)),
  `input_sha256`, `started` (ISO 8601, UTC), `host` (`os_build`, `machine`) and `rows`. A row holds `id`, `kind`,
  `token_count`, `tokens_sha256` (the SHA-256 of the token IDs written in decimal and joined by commas), `capacity`,
  `logits` (24 values before the temperature), `probabilities` (shipping temperature), `probabilities_t1`, `top`
  and `elapsed_seconds`. A rejected row holds `id`, `kind`, `error` (`tooManyTokens` or `invalidOptions`) and, for
  `tooManyTokens`, `token_count`. Progress goes to standard error. Any other failure stops the run without an output.
- Running the model is a heavy job: hold the shared machine lock.

The recipe adds three steps after `verify_quantization`, still before `verify_merge`:

- `typed_input` restores the text of F from the corpus and JGLUE, checks that the token counts and bands equal the
  fixture, and writes `input-F.json` and `check.json`. `--sets` takes several sets (`F`, `Q`, `Cal`, `H`, `J`) and
  writes one `input-<set>.json` for each.
- `typed_decide_f` runs `coreai-verify typed-decide` on `input-F.json` with the variant bundle of `export_gdnfp32`
  and writes `swift-F.json`. The executable is the variable `verify`, which must be passed with `--set`
  ([recipe README](../Tools/Conversion/recipes/jev27b/README.md)).
- `compare_f` scores `swift-F.json` against stage 2 in `reference_q.json` and writes `comparison.json`
  (`typed-decision-fidelity/1`) and `report.md`. `passed` requires the rows of F in order with the reference token
  IDs (by `tokens_sha256`) and a `tooManyTokens` rejection exactly for the rejected row, finite probabilities between
  0 and 1 that sum to 1 within 1e-3, and at temperature 1, for every input band with rows, a largest
  `e_j = max_i |p3_i - p2_i|` of at most 0.03 and a mean of at most 0.01. Bands without rows are listed as
  unevaluated. Top-1 changes are reported with whether the new option is within 0.06 of the reference top; they are
  not a criterion. The shipping-temperature statistics are recomputed from the 24 logits of both sides and reported
  separately; `--temperatures <json>` (a `per_kind` object) scores them at other temperatures. The difference from
  stage 1' is shown for reference only.

With `COREAI_TYPED_DECISION_BUNDLE` set to the absolute path of a bundle folder (the variant, as in the recipe, or
the first bundle), `TypedDecisionInferenceTests` runs ten rows of F (the two card examples and the eight JGLUE rows of
`render.json`, all at most 512 tokens) on Core AI: the band limits against stage 2 at temperature 1 with the token IDs
of the reference, bit-identical results for the same row run twice, a request cancelled at the forward checkpoint and
a request cancelled while its forward pass runs (through a backbone that does not pass the cancellation on), each
followed by a row that still matches, and the difference between the rounded capacity and 8,192 positions. The fixture is `Fixtures/TypedDecision/inference.json`,
written by `swift_fixture` with `--quantized-reference <reference_q.json>`. The tests load the model, so hold the
shared machine lock; without the variable they are skipped. `DECIDE_METRICS_DIR` saves their measurements as
`typed-*.json`.

~~~sh
COREAI_TYPED_DECISION_BUNDLE=/path/to/jev27b_s1_gdnfp32_qwen3_5_cfg_dynamic swift test --package-path CoreAISamplesKit \
  --filter TypedDecisionInferenceTests
~~~

## Terms

Qwen3.8-27B (Qwen) and JEV-27B (AutoTrust AI) are Apache-2.0. JEV-27B was trained on
`SargeDev/jev-distill-corpus-v3` (Apache-2.0), most of whose labels are output distributions of TypeSafe Jev 1.13. The
fixed sets are built from that corpus and from JGLUE (CC BY-SA 4.0) and are not redistributed. The Swift test
fixtures contain the text of eight JGLUE rows under CC BY-SA 4.0; their notice is in `render.json`.
