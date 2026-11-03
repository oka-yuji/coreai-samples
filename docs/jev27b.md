# JEV-27B typed decisions (draft)

**Status: draft, not published.** This page describes the typed-decision bundle that the
[`jev27b` conversion recipe](../Tools/Conversion/recipes/jev27b/README.md) builds from
[autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B). The bundle is not distributed. The output check of the
merged checkpoint (stage 1 against stage 1', largest probability difference at most 0.01) has not passed and awaits a
decision; the bundle and the quantized references below are a limited exploration that uses that merged checkpoint. The
quality and calibration results under [Quality and calibration](#quality-and-calibration) measure agreement with the
teacher distribution of the corpus on fixed sets; they are not a claim of real-world correctness. The Japanese results
under [Japanese (JGLUE)](#japanese-jglue) are descriptive and have no pass line.

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
| `temperatures` | `per_kind` (the `per_kind` values of the distributed `calibration.json`, unchanged, all above 0), `calibration_id` (`<model>@<revision>/calibration.json`), `calibration_sha256`, `source` (`card`). The calibrated copy written by `calibrate` changes these four ([Calibrated bundle](#calibrated-bundle)). |
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
| 0 | Distributed backbone with the decision head, adapter disabled | bfloat16 | `reference` (F), `reference_q` (Q) |
| 0q | Stage 0 quantized with the export recipe, computed in float32 | float32 | `probe_base` |
| 1 | Backbone with the unmerged LoRA adapter (PEFT) | bfloat16 | `reference` (F), `reference_q` (Q) |
| 1' | Merged checkpoint | bfloat16 | `reference_merged` (F), `reference_merged_q` (Q) |
| 2 | Merged checkpoint quantized with the export recipe, computed in float32 | float32 | `probe_merged` |
| 3 | Core AI asset of the float32 gated delta update variant, run from Swift (`TypedDecisionModel`) | asset | `typed_decide_f` (F), `typed_decide_q`, `typed_decide_cal`, `typed_decide_h` |

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

## Quality and calibration

The recipe measures stage 3 (the Core AI asset of the variant bundle) against the other stages on Q and refits the
temperatures from stage 3 on Cal, with the conditions fixed before any measurement. Every number is agreement with the
teacher distribution of the corpus (`target`, output distributions of Jev 1.13), not calibration against real-world
correctness.

These steps also run before `verify_merge`:

- `reference_q` and `reference_merged_q` run Q through stages 1 and 0 (PEFT, `disable_adapter()`) and through stage 1'
  with `reference_sets.py`, which repeats the model loading and checks of `reference.py` and `reference_merged.py`,
  imports their head, forward and header functions, and leaves both modules unchanged. They write `reference-Q.json` and
  `reference-merged-Q.json` (rows in the form of `reference_q.json` with `set`, without hidden states). Stage 1 is
  computed but not used.
- `typed_input_eval` writes `input-Q.json`, `input-Cal.json` and `input-H.json`, and `typed_decide_q`,
  `typed_decide_cal` and `typed_decide_h` run them through `coreai-verify typed-decide` with the variant bundle.
- `score_q` (`score.py`) writes `quality.json` (`typed-decision-quality/1`) and `report.md`. `recorded` is its check:
  missing rows, rows that differ in ID, order, token IDs or rejection, non-finite values and invalid targets fail the
  step, while a quality condition that is not met is recorded and does not.
- `calibrate` (`calibrate.py`) writes `calibration.json` (`typed-decision-calibration/1`), `report.md`, a calibrated
  copy of the bundle and `comparison-refit.json`. Its checks are `recorded` and the `passed` of `comparison-refit.json`.

Definitions:

- Probabilities of every stage are recomputed from the 24 head logits before the temperature, with the slot range, the
  division by T and the softmax of `typed_head.decide`, at temperature 1 and at the card temperatures. Log-probabilities
  use log-sum-exp in the same order and are not clipped.
- Targets are restored with `load_rows` after the corpus integrity check. A target needs one non-negative value per
  option with a sum within 1e-3 of 1 and is divided by its sum. The teacher top is the first largest target value.
- choice accuracy: the argmax of p equals the teacher top, in points. noul Brier (soft): mean of (P(true) - t(true))^2;
  the hard form uses 1 when the teacher top is `true`. Multiclass Brier: the sum of (p_i - t_i)^2 over the options.
  score MAE / 5: mean of |sum k p(k) - sum k t(k)| / 5. KL(t || p) sums the terms with t_i > 0. noul AUROC ranks
  P(true) with the teacher top as the label, leaves out rows with a 0.5 target, counts tied predictions one half and is
  undefined with one class. ECE uses the confidence max p in 15 equal bins (1.0 falls in the last), against the mean
  target of the predicted option (soft) or the accuracy (hard); empty bins weigh 0.
- Intervals: 2,000 bootstrap replicates with seed 20261102 and the same draws for every stage; within a kind, rows that
  share a state are drawn together. The 95% interval takes the 2.5 and 97.5 percentiles with linear interpolation.

Conditions at temperature 1 on Q:

| Condition | Compared | Passes when |
|---|---|---|
| Quantization, choice | accuracy of stage 3 minus stage 1' | at least -2 points |
| Quantization, noul | Brier (soft) of stage 3 minus stage 1' | at most +0.01 |
| Quantization, score | MAE / 5 of stage 3 minus stage 1' | at most +0.02 |
| Adaptation | gain(bf16) = stage 1' against stage 0, gain(4bit) = stage 3 against stage 0q, for choice accuracy, noul Brier, score MAE / 5 and KL | gain(4bit) - 0.8 gain(bf16) >= 0 for each metric |

The point estimates decide the quantization conditions; the 95% interval of each paired difference is reported with
whether it contains the limit. Accuracy gains are adapted minus base, the other gains base minus adapted. A metric with
gain(bf16) at most 0 is undefined, which is not a pass. Stage 2 against stage 0q is reported beside the gains to
separate the contribution of the adapter at 4 bits from the difference between stage 3 and stage 2.

Temperatures: for each kind, beta = 1 / T in [0.25, 4] minimizes the mean cross-entropy to the target of the stage 3
logits on Cal with a golden-section search until the bracket is narrower than 1e-7. Reaching an end of the range is
recorded; the range is not widened. The candidate is rounded to float32, the value that the Swift runtime uses, and
written as the shortest decimal that rounds back to it. A kind adopts its candidate only when neither the NLL nor the
multiclass Brier on H' is larger than at the card temperature, both at float32; H' is H without the 38 rows whose
state also occurs in Cal. All of H is recorded too, but H selects the temperature and is not a final evaluation. ECE
intervals are conditional on the chosen temperature and leave out the uncertainty of the fit and of the selection.

Results (run `20261103T154310-65e7b5bb` for Q, Cal and H; `calibrate` in run `20261103T175311-463e881b`). The
quantization conditions and every adaptation gain pass:

| Condition (temperature 1, Q) | Stage 1' | Stage 3 | Difference [95% interval] | Limit | Result |
|---|---:|---:|---|---|---|
| choice accuracy (points) | 87.33 | 86.67 | -0.67 [-3.00, +1.67] | >= -2 | pass; the interval contains the limit |
| noul Brier (soft) | 0.00141 | 0.00165 | +0.00024 [+0.00009, +0.00042] | <= +0.01 | pass |
| score MAE / 5 | 0.02120 | 0.02171 | +0.00051 [-0.00029, +0.00129] | <= +0.02 | pass |

| Metric | gain(bf16) | gain(4bit) | gain(4bit) - 0.8 gain(bf16) | Stage 2 over 0q | Result |
|---|---:|---:|---:|---:|---|
| choice accuracy (points) | +25.33 | +26.33 | +6.07 | +26.67 | pass |
| noul Brier (soft) | +0.0648 | +0.0723 | +0.0205 | +0.0723 | pass |
| score MAE / 5 | +0.0937 | +0.0901 | +0.0151 | +0.0901 | pass |
| KL | +0.3378 | +0.3555 | +0.0852 | +0.3555 | pass |

| Stage (temperature 1, Q) | KL | Top-1 | Choice accuracy | noul AUROC | noul Brier (soft / hard) | score MAE |
|---|---:|---:|---:|---:|---|---:|
| 0 | 0.3577 | 0.5844 | 62.00 | 0.8999 | 0.0662 / 0.1934 | 0.5746 |
| 0q | 0.3757 | 0.5900 | 60.33 | 0.8925 | 0.0740 / 0.2008 | 0.5591 |
| 1' | 0.0198 | 0.9089 | 87.33 | 0.9941 | 0.0014 / 0.0843 | 0.1060 |
| 2 | 0.0203 | 0.9044 | 87.00 | 0.9933 | 0.0017 / 0.0859 | 0.1087 |
| 3 | 0.0202 | 0.9033 | 86.67 | 0.9934 | 0.0017 / 0.0857 | 0.1085 |

`quality.json` and `report.md` also hold the same table at the card temperatures, the values per kind and the card's
own figures, which cover all 29,955 rows of `test_set_30k` (placeholder rows included) at the card temperatures and are
therefore shown for reference only. One noul row of Q has a 0.5 target and is left out of the AUROC.

| Kind | Card T | Refitted T (no end reached) | H' NLL, card then refitted | H' multiclass Brier, card then refitted | Adopted |
|---|---:|---:|---|---|---|
| noul | 1.0143135 | 1.0056266 | 0.384497, 0.384488 | 0.003832, 0.003823 | yes |
| choice | 1.0161281 | 1.0053791 | 0.732314, 0.732290 | 0.020188, 0.020145 | yes |
| score | 1.0036213 | 1.010558 | 1.062951, 1.062910 | 0.009226, 0.009215 | yes |

ECE on H' at the card and the adopted temperatures, with 95% intervals:

| Kind | Soft, card T | Soft, adopted T | Hard, card T | Hard, adopted T |
|---|---|---|---|---|
| noul | 0.0041 [0.0030, 0.0091] | 0.0035 [0.0027, 0.0086] | 0.1619 [0.1446, 0.1795] | 0.1609 [0.1437, 0.1785] |
| choice | 0.0122 [0.0093, 0.0225] | 0.0092 [0.0079, 0.0199] | 0.2185 [0.1949, 0.2430] | 0.2163 [0.1932, 0.2420] |
| score | 0.0043 [0.0048, 0.0122] | 0.0042 [0.0045, 0.0120] | 0.3349 [0.3077, 0.3609] | 0.3365 [0.3093, 0.3626] |

ECE sums absolute bin gaps, so resampling biases it upward; for score the soft value lies below its percentile
interval. The hard ECE compares the confidence with the agreement of the top option with the teacher top; for score
the mean confidence on H' (about 0.56) lies far below that agreement (about 0.89) because the teacher distributions
are spread out. The fidelity criteria of F, re-evaluated from the saved logits at the adopted temperatures, still pass
(`comparison-refit.json`; at the adopted temperatures, maximum 0.0065, mean 0.0011, no top-1 change).

At stage 3 the `elapsed_seconds` of the rows (from the call of `decide` to its return) add up to 458.7 s for Q,
1,866.3 s for Cal and 919.6 s for H (about 180 tokens per second over all rows, median 0.48 s per row). Stages 1 and
0 of Q took 679.4 s and 644.7 s and stage 1' took 628.9 s on MPS.

### Calibrated bundle

`calibrate` writes `jev27b_s1_gdnfp32_qwen3_5_cfg_dynamic/` into its own output as an APFS clone (`cp -c`) of the
variant bundle and then checks that every file of the variant still matches its `SHA256SUMS` and that the copy holds
the same files. The variant itself is not modified.

- `calibration-refit.json` (`typed-decision-calibration-refit/1`) records, for each kind, the card temperature, the fit
  (beta, bracket, iterations, end reached, cross-entropy), the candidate, the adopted value and the reason, the
  statistics of Cal, H' and H at both temperatures, the excluded rows of H, the settings, the SHA-256 of `main.mlirb`
  and of the Cal and H fixtures, and the input digests of the stage 3 runs.
- `decision.json` differs from the variant in `files`, which lists `calibration-refit.json`, and in `temperatures`:
  `per_kind` holds the adopted values (the refitted float32 decimals, or the card value for a kind that keeps it),
  `source` is `refit` when any kind adopts its candidate, `calibration_sha256` is the SHA-256 of
  `calibration-refit.json` and `calibration_id` is `jev27b-refit/<its first 16 hexadecimal digits>`. `SHA256SUMS` is
  written after `decision.json` and lists both files.
- The asset, head and tokenizer are byte-identical to the variant, so the specialization cache entry of the variant
  applies. `TypedDecisionBundle` reads the temperatures as `Float`; with `JEV_BUNDLE_DIR` set to the calibrated bundle,
  the bundle test compares them with the `temperatures` of `calibration-refit.json`.

~~~sh
python3 -m Tools.Conversion.adapters.jev27b.decision <calibrated bundle>/decision.json --check-files
JEV_BUNDLE_DIR=/path/to/calibrated/jev27b_s1_gdnfp32_qwen3_5_cfg_dynamic swift test --package-path CoreAISamplesKit \
  --filter TypedDecision
~~~

## Japanese (JGLUE)

The recipe also runs the Japanese set J (200 rows each of the JNLI, JCommonsenseQA and JSTS validation files of
JGLUE v1.2.0) through stage 1' and stage 3 with two prompt variants, after fixing which variant to adopt on a separate
selection set. No pass line is set. The numbers below are descriptive results on 200 rows per task without intervals;
they are not a claim of Japanese quality or of calibration.

These steps run after `calibrate`, still before `verify_merge`:

- `jdev_fixture` (`japanese.py jdev`) writes `jdev.json`: 100 rows per task of the pinned JGLUE validation files. A
  row is eligible when it is not in J or F, when neither its `sentence1` nor its `sentence2` (JNLI, JSTS), or its
  question (JCommonsenseQA), equals any `sentence1`, `sentence2` or question of a JGLUE row of J or F (compared across
  the three tasks), and when its options render (no row was left out for this). The native IDs (`sentence_pair_id`,
  and `q_id` turned into a string) of the eligible rows of each task are sorted as strings, and one
  `random.Random("20261102:Jdev")` draws `rng.sample(ids, 100)` for JNLI, JCommonsenseQA and JSTS in this order.
  `jdev.json` records the ID and the SHA-256 of each row and the size of every exclusion; it holds no text.
- `typed_input_jdev` and `typed_input_j` (`japanese.py input`) wrap the rows with both variants and write the CLI input
  (`input-<set>.json`, with text, kept in the work directory) and `rows-<set>.json`: for each row the key, the SHA-256
  of the JGLUE row (the fixture value), the SHA-256 of the rendered text after wrapping, the token count and the label,
  without text.
- `typed_decide_jdev` and `typed_decide_j` run `coreai-verify typed-decide` with the calibrated bundle of `calibrate`.
- `prompt_adoption` (`japanese.py adopt`) scores Jdev and writes `prompt-adoption.json` with `frozen: true`. Its check is
  `frozen`. `typed_input_j` refuses an adoption that is not frozen for the same `wrappers.json`, so J runs only after
  the adoption is fixed.
- `reference_j` (`reference_j.py`) builds the same wrapped rows of J (without the permuted rows) and runs them through
  stage 1' with the `merged_stage` of `reference_sets.py`; it passes with the same checks as `reference_merged_q`.
- `score_j` (`japanese.py score`) writes `japanese.json` (`typed-decision-japanese/1`) and `report.md`. Its check is
  `recorded`: missing rows, rows that differ in ID, order, token IDs, hashes or labels, non-finite values and stage 3
  probabilities that differ from the bundle temperatures fail the step.

Both variants put an instruction of the same content around the unchanged JGLUE text and differ only in its language;
the bare form of the A1 rows (the question without an instruction) is not evaluated. The structure tags stay `bare-v1`.
The prefix, suffix and JNLI options of each variant are in
[`wrappers.json`](../Tools/Conversion/recipes/jev27b/wrappers.json):

| Task | Kind | State | Question | Options |
|---|---|---|---|---|
| JNLI | `choice` | `sentence1` | prefix + `sentence2` + suffix | entailment, contradiction, neutral in the language of the variant |
| JNLI noul (derived from the 200 JNLI rows, ID `<id>:noul`) | `noul` | `sentence1` | prefix + `sentence2` | `false`, `true`; true only for entailment |
| JCommonsenseQA | `choice` | empty | prefix + question + suffix | the five choices, not translated |
| JSTS | `score` | `sentence1` | prefix + `sentence2` + suffix | `0` to `5` |

A row key is the JGLUE ID (with `:noul` for the derived rows), the variant and the option order.

Adoption, fixed before any run: on Jdev at temperature 1, each task adopts the variant with more correct JNLI choice
rows, more correct JCommonsenseQA rows, or the smaller MAE of the expected JSTS score; a tie adopts `ja`. JNLI noul
follows the JNLI choice adoption as a common policy, which is not a claim that it is best for noul. The adoption chooses
the reported variant and the variant of the order test; J itself runs with both variants.

Definitions (fixed before any run):

- Probabilities are recomputed from the 24 logits of each stage at temperature 1 and at the adopted temperatures of the
  calibrated bundle. KL and ECE are not computed.
- JNLI choice and JCommonsenseQA accuracy: the first largest probability is the label. JNLI noul accuracy: P(true) > 0.5
  is true and P(true) = 0.5 is false. AUROC ranks P(true) with entailment as the positive class, ties count one half,
  and it is undefined with one class. The hard Brier score is the mean of (P(true) - y)^2 with y = 1 for entailment.
- JSTS: E[p] = sum k p(k) against the continuous label, as MAE, Pearson and Spearman (average ranks for ties); either is
  undefined when one side is constant.
- Option order (stage 3, adopted JCommonsenseQA variant): the options in four fixed orders (as given, reversed, rotated
  left by one and by two); the label moves with its option, and each answer is mapped back to the original option. The
  accuracy of each order and the number of rows whose four answers agree are recorded.
- Stage 3 minus stage 1' on the same rows, wrapper and temperature is the difference of quantization and execution path
  together.

Results (run `20261103T184500-52091525`: `prompt_adoption` for Jdev, `score_j` for J). The adoption, frozen before J
ran:

| Task (Jdev, temperature 1) | Metric | `ja` | `en` | Adopted |
|---|---|---:|---:|---|
| JNLI choice | accuracy | 0.8300 (83/100) | 0.8900 (89/100) | `en` |
| JCommonsenseQA | accuracy | 0.9600 (96/100) | 0.9500 (95/100) | `ja` |
| JSTS | MAE of E[p] | 0.5806 | 0.5577 | `en` |
| JNLI noul | follows JNLI choice | - | - | `en` |

J at temperature 1 (200 rows per task; the last column is the adopted variant):

| Task | Metric | Stage 1' `ja` | Stage 1' `en` | Stage 3 `ja` | Stage 3 `en` | 3 - 1' `ja` | 3 - 1' `en` | Adopted |
|---|---|---:|---:|---:|---:|---:|---:|---|
| JNLI choice | accuracy | 0.8200 | 0.8700 | 0.8050 | 0.8550 | -0.0150 | -0.0150 | `en` |
| JNLI noul | accuracy | 0.9350 | 0.9200 | 0.9250 | 0.9200 | -0.0100 | +0.0000 | `en` |
| JNLI noul | AUROC | 0.9856 | 0.9801 | 0.9842 | 0.9777 | -0.0014 | -0.0024 | `en` |
| JNLI noul | hard Brier | 0.0499 | 0.0531 | 0.0541 | 0.0588 | +0.0043 | +0.0057 | `en` |
| JCommonsenseQA | accuracy | 0.9600 | 0.9700 | 0.9600 | 0.9600 | +0.0000 | -0.0100 | `ja` |
| JSTS | MAE of E[p] | 0.5819 | 0.5412 | 0.5778 | 0.5287 | -0.0041 | -0.0124 | `en` |
| JSTS | Pearson | 0.8501 | 0.8555 | 0.8474 | 0.8591 | -0.0027 | +0.0035 | `en` |
| JSTS | Spearman | 0.7818 | 0.7858 | 0.7796 | 0.7924 | -0.0022 | +0.0066 | `en` |

- At the adopted temperatures the accuracies and AUROC values are the same, and the other values move by at most
  0.0007 (`japanese.json` and `report.md` hold both tables). 35 of the 200 JNLI rows are entailment.
- JCommonsenseQA option order (stage 3, `ja`): accuracy 0.9600 as given, 0.9700 reversed, 0.9600 rotated by one and
  by two; the four answers agree on 191 of 200 rows.
- Stage 3 against stage 1' row by row at temperature 1: the top option is the same on 179 to 200 of the 200 rows of
  each task and variant (JSTS is lowest), the largest probability difference is 0.17 and the mean 0.011 to 0.033.
- At stage 3 the `elapsed_seconds` of the rows (from the call of `decide` to its return) add up to 278.2 s for Jdev
  (600 rows) and 953.0 s for J (2,200 rows), with medians of 0.464 and 0.436 s per row; the forward runs of stage 1'
  took 975.8 s for the 1,600 rows of J on MPS.

## Terms

Qwen3.8-27B (Qwen) and JEV-27B (AutoTrust AI) are Apache-2.0. JEV-27B was trained on
`SargeDev/jev-distill-corpus-v3` (Apache-2.0), most of whose labels are output distributions of TypeSafe Jev 1.13. The
fixed sets are built from that corpus and from JGLUE (CC BY-SA 4.0) and are not redistributed. The Swift test
fixtures contain the text of eight JGLUE rows under CC BY-SA 4.0; their notice is in `render.json`.
