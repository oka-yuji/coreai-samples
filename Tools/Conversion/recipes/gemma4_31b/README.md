# Gemma 4 31B BF16 reference

This recipe pins [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475` and records the outputs of the original Transformers
implementation with the distributed BF16 weights, BF16 compute and MPS for twelve self-written diagnostic
inputs. The [FP32 anchor and FP16 diagnostics](#fp32-anchor-and-fp16-diagnostics) of this recipe and later work (the
Core AI comparison and quality checks) read these outputs. It does not convert the model, and the diagnostic inputs
are not a quality evaluation set. It also defines the converter environment and a model class for a macOS export
with dynamic shapes, with a tool that compares the first layers of that class with the original implementation in
FP32 ([Converter environment and model class](#converter-environment-and-model-class)). A second recipe,
`recipe-fp16.json`, exports the first layers to a dynamic-shape FP16 Core AI bundle with the outputs that the drafters
of speculative decoding read, and compares the logits of the own engine with an FP32 reference of the same layers
([FP16 export of the first layers](#fp16-export-of-the-first-layers)). A model class of the MTP drafter is compared with
the original drafter in FP32 ([MTP drafter](#mtp-drafter)).

- `model.json`: the contract. It pins the size and SHA-256 of all twelve files of the revision, the main
  text configuration values (60 layers as five sliding-window layers followed by one full-attention layer,
  hidden size 5376, MLP width 21504, vocabulary 262144, window 1024, 16 and 4 key/value heads, head
  dimensions 256 and 512, shared keys and values in the full-attention layers, final logit soft cap 30,
  tied embeddings, BOS 2 and stop tokens 1, 106 and 50), the distributed tokenizer files, the reference
  precision and the upstream converter revision `c8e00c01800ba6ad324707007f7af358efc980a4` of the converter
  environment. A snapshot that differs fails; the contract is not adjusted to the files.
- `assistant.json`: the revision of the MTP drafter and the size and SHA-256 of the two files of it that are read.
- `fixtures.json`: the twelve diagnostic cases (Japanese and English conversations, code, JSON extraction,
  a long Japanese input built from numbered paragraphs, a one-word turn and a raw continuation). All text
  is self-written.
- `pyproject.toml` and `uv.lock`: the reference environment. It is the JEV-27B reference lock without PEFT:
  Python 3.11, torch 2.15.0, transformers 5.16.1, accelerate 1.15.0, numpy 2.4.6, safetensors 0.8.0 and
  huggingface-hub 1.33.0. The first candidate also dropped accelerate, which only PEFT required there;
  transformers 5.16.1 then refused `device_map="mps"` with `Using a device_map ... requires accelerate`,
  so accelerate is pinned at the version of the JEV-27B lock. No other versions were tried.

## Local inputs and setup

Setup never downloads models. Place all files of the revision under
`.local/conversion/models/google--gemma-4-31B-it` with a sibling `google--gemma-4-31B-it.integrity.json`
in the form used for JEV-27B. Its row for every file needs the expected and actual size and the SHA-256 of
the whole file (`actual_sha256`), which must equal the contract, and `ok` set to true; the git blob SHA-1
that the small files are checked against upstream is an additional field. Files of the download state under
`.cache/` may be listed in `local_files_not_in_scope`; any other extra file fails.

~~~sh
python3 -m Tools.Conversion.local setup --recipe gemma4_31b
python3 -m Tools.Conversion.local plan --recipe gemma4_31b
python3 -m Tools.Conversion.local run --recipe gemma4_31b
python3 -m Tools.Conversion.local run --recipe gemma4_31b --refresh
~~~

Setup checks the integrity record against the contract, then under the shared machine lock creates
`.local/conversion/gemma4_31b/.venv` from this folder's lock, the converter environment (below),
`.local/conversion/gemma4_31b/setup.json` and `.local/conversion/gemma4_31b.paths.json`. `--snapshot` selects
another folder. The run uses `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and a Hugging Face home under
`.local/conversion/gemma4_31b/hf-cache`.

## Steps

- `prepare` writes `ready.json`. It requires the integrity record to match the contract (repository,
  revision, every size and SHA-256, no failed entry), the folder to hold exactly the pinned files, and
  then hashes every file again, including both weight shards. It compares `config.json` and
  `generation_config.json` with the contract, checks that the BOS, EOS and end-of-turn tokens of
  `tokenizer_config.json` map to the contract IDs and that no token ID reaches the vocabulary size, and
  requires the tokenizer that `AutoTokenizer` builds to equal the distributed `tokenizer.json`. It records
  the versions of the environment and the Apple toolchain, requires MPS, and stops when `pyproject.toml` or
  `uv.lock` differ from those recorded by setup.
- `reference` is heavy and runs under the shared machine lock. It waits for the first monitor sample that
  reads the resident memory, swap and the free percentage (at most 60 seconds), then hashes the files it loads
  (both weight shards, `config.json`, `generation_config.json`, `model.safetensors.index.json` and the three
  tokenizer files) and stops before loading unless every size and SHA-256 equals the contract and the folder still holds
  exactly the pinned files under the rule of `prepare`, which ignores only the download state under `.cache/` and
  Finder's `.DS_Store` files (the loader would prefer a single `model.safetensors` to the pinned shards).
  `reference.json` records these hashes, not the copies in `ready.json`. It then loads `Gemma4ForConditionalGeneration`
  with `dtype=torch.bfloat16`, `device_map="mps"` and `output_loading_info=True` and requires empty loading
  lists, BF16 parameters on MPS and evaluation mode. Inputs are text only; the vision tower is loaded with
  the distributed weights but never called. Conversations are rendered by `apply_chat_template` with
  `add_generation_prompt=True` and `enable_thinking=False` (the template then appends an empty thought
  channel) and must tokenize to the same IDs with `add_special_tokens=False`; the template writes the
  single BOS. The raw case gets an explicit BOS before text tokenized with `add_special_tokens=False`.
  Greedy decoding uses `generate` with `do_sample=False`, one beam, no penalties and the contract stop
  tokens, under `torch.no_grad()`.

For every case `cases/<id>.tokens.json` holds the rendered input IDs, the greedy output IDs, the stop
reason and the top two IDs and logits of every step. `cases/<id>.logits.npy` holds float32
`[input length + output length, 262144]`: the input followed by the greedy output goes through one forward
pass without a cache, and the logits the implementation returns (after the soft cap) are kept for every
position, so row `j` is the distribution of token `j + 1`. The scoring rows run from the last input
position (`first_scoring_row`, the input length minus one) to the position before the last output token,
so scoring row `i` predicts output token `i`.

`reference.json` records the revision, the weight and contract SHA-256, the versions, dtype, device,
class and loading arguments, the template and BOS handling, and per case the saved and scoring positions,
the non-finite values of every saved row and of the per-step logits, and, on the scoring rows, the
positions where the forced argmax equals the greedy token (mismatches keep the position and the
probability difference of the top two candidates) and the rows whose top two logits are exactly equal.
These two are recorded, not criteria; where the top two are equal, `argmax` picks the smaller ID. It also
records a second greedy run in the same process, the times and the greedy tokens per second. Its `passed`
requires clean loading, no non-finite value (in the saved rows and in the step logits of both greedy runs),
each greedy token at the maximum of its step logits, identical tokens in the second run, a clean monitor
and recorded resources.

A thread samples the resident memory, swap, the memory free percentage and the free disk every 5 seconds
(`monitor.tsv`, from `adapters/jev27b/monitor.py`). The stage stops as failed with `failure.json` and exit code 1 when:

- swap grows by more than 2,000,000,000 B from its first sample;
- the free percentage stays below 8 for three samples in a row;
- swap is above 16,000,000,000 B and rises on three samples in a row (the condition of the shared monitor);
- a sample cannot read the resident memory (the shared monitor writes 0 when `ps` prints nothing), swap or the
  free percentage, or the monitor thread ends with an error.

A process still running 120 seconds after a stop is killed, also when the monitor thread itself ended with an
error (a separate timer takes over the kill). A stop at any point from the start of the monitor to the writing of
`reference.json`, including one while the monitor starts or after the model work, ends with `failure.json`. A run
that finishes after a stop, with an unreadable sample or a monitor error is not `passed`, and neither is a run
without its peak resident memory, kernel lifetime footprint peak, MPS allocation or swap increase. These are
recorded after loading, after every case and at the end. The stage needs 20 GiB of free disk; its saved logits of
the twelve cases take about 3.4 GB. Six diagnostic steps follow `reference`
([FP32 anchor and FP16 diagnostics](#fp32-anchor-and-fp16-diagnostics)).

## Second run and shared numbers

`--refresh` runs every step of the recipe again in new processes: `prepare`, `reference` and the six diagnostic
steps described below. Compare the two reference artifacts (the `artifacts` folders that the two reports list for
`reference`) with the reference environment:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.compare \
  --first <first artifacts> --second <second artifacts> --output .local/conversion/work/gemma4_31b/rerun.json
python3 -m Tools.Conversion.adapters.gemma4_31b.summary --report <first report.json> --reference <first reference.json> \
  --report <second report.json> --reference <second reference.json> --comparison .local/conversion/work/gemma4_31b/rerun.json
~~~

`compare` records per case whether the input and output tokens match, the first position where the forced sequences
differ, and the largest absolute logit difference over the saved rows whose context is still identical (every row
when the sequences match). It passes only when both `reference.json` files passed and differ (an artifact compared
with itself or with a copy fails), the model, environment, fixtures and cases are the same, every token file and
saved array has the size and SHA-256 that its own `reference.json` records, the input and output token counts, the
saved positions and the first scoring row equal that record, every saved array is float32 with the shape
`[saved positions, vocabulary]`, no compared logit is NaN or infinite, and the input and output tokens of every
case match; any other finding is listed in `problems`. Non-finite logits are counted per run and leave the largest
difference of the case and of the comparison unset. `summary` prints only states, numbers, hashes, versions and
case IDs. It stops unless every `--report` is the run that executed the `reference` step whose `reference.json` is
given with it (the report's artifacts folder of that step holds the file), no report or `reference.json` is given
twice, and the comparison lists the SHA-256 of the two `reference.json` files, in the same order (the printed
record keeps them). It lives beside the steps because every `Tools/Conversion/*.py` file is part of each step's
cache key. The published record is [docs/validation/gemma4-31b](../../../../docs/validation/gemma4-31b/README.md).

## Quality evaluation baseline

Two modules write the outputs of the `reference` role of the quality contract `G31-QC-v1` (#669): `inputs` renders the
items of a set into input records without loading the model, and `baseline` generates from those records with the BF16
weights on MPS. They are not steps of `recipe.json`, because the primary set takes hours and runs in sessions that
continue one run folder, releasing the shared machine lock between them. Run them from the repository root with the
reference environment; `baseline` loads the model, so take the heavy-job lock of `AGENTS.md` first.

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.inputs \
  --model Tools/Conversion/recipes/gemma4_31b/model.json --snapshot .local/conversion/models/google--gemma-4-31B-it \
  --contract Tools/LLMEvaluation/contract.json --set Tools/LLMEvaluation/sets/primary.json \
  --items <build>/primary.jsonl --output <new folder>
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.baseline \
  --model Tools/Conversion/recipes/gemma4_31b/model.json --snapshot .local/conversion/models/google--gemma-4-31B-it \
  --prepare <folder of ready.json> --contract Tools/LLMEvaluation/contract.json --inputs <inputs folder>/inputs.jsonl \
  --commit <commit> --output <run folder> --log <new folder> --max-minutes 60
~~~

- `inputs`: `--items` is the `primary.jsonl` of `python3 -m Tools.LLMEvaluation build`, `--set` the frozen set file and
  `--output` a new folder. It writes `inputs.jsonl`, one record per item in item order with `id`, `input_ids`,
  `input_sha256` and `max_new_tokens`, and `inputs.json`, with the version and SHA-256 of `contract.json` and the count
  and SHA-256 of `inputs.jsonl`. It writes nothing when an item does not match. `RecordTests` in
  `Tools/Conversion/tests/test_gemma4_31b_inputs.py` fixes its checks.
- `baseline`: `--prepare` is the output folder of the `prepare` step, `--commit` the commit to record (the command
  does not run git), `--output` the run folder and `--log` a new folder outside it for the session's `monitor.tsv` and
  `session.json`, or `failure.json` when the monitor stops the session. The run folder holds `outputs.jsonl`, the
  output records of contract §5.1; `progress.json`, the progress of those lines; and `run.json`, the run record, which
  has `finished` after a session that ended with exit code 0 (the next session removes it when it starts).
  `InputRecordTests` in `Tools/Conversion/tests/test_gemma4_31b_baseline.py` fixes the checks of the input records,
  `GenerationTests` and `SmallModelTests` the generation, `RunFolderTests` the run folder and its resume, and
  `SessionTests` the end of a session.

With `--max-minutes`, a session ends after the item in progress once that many minutes have passed since the command
started. Its last line of standard output (`record` `session`) has `complete`; run the same command with a new `--log`
folder until `complete` is true. A session that stops for another reason exits with a nonzero code, and the same
command with a new `--log` folder continues the run once the cause is removed. When an item ends with an error, its
record is written and the command stops; a baseline record with an error makes the comparison `incomplete`, so report
it instead of continuing. When the command refuses a run folder, it changes nothing in it; do not edit the files of a
run folder, and start a new run folder instead.

The published record of the primary-set run is
[docs/validation/gemma4-31b/baseline-20261122.md](../../../../docs/validation/gemma4-31b/baseline-20261122.md).

## Layer 1 comparison

`layer1_compare` judges layer 1 of the quality contract (§4, #697). It compares the logits that two implementations give
for the forced input of `python3 -m Tools.LLMEvaluation forced`
([Layer 1 forced input](../../../LLMEvaluation/README.md#layer-1-forced-input)) under the limits of `layer1` in
`Tools/LLMEvaluation/contract.json`: no non-finite value, a probability error of at most 0.03 at every position, a mean
of the per-position maximum of at most 0.01 in each input-length band and no top-1 mismatch outside a near tie. It loads
no model; the runs that write the logits of the two sides are separate. Run it from the repository root with the
reference environment:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layer1_compare \
  --forced <forced folder> --inputs <inputs folder> --run <reference run folder> \
  --baseline <logits folder> --candidate <logits folder> --output <new folder>
~~~

`--inputs` and `--run` are the input records and the finished reference run that the forced input was made from, as
given to `forced`. The command makes the forced input again from them with `make` of `Tools/LLMEvaluation/layer1.py`,
the function of `forced` with all of its checks, and requires `forced.jsonl` and `forced.json` of `--forced` to be equal
to it byte for byte; the items are compared on the records made again.

Each side is one folder with `logits.json` and one array file per item:

| `logits.json` field | Value |
|---|---|
| `schema` | `gemma4-31b-layer1-logits/1` |
| `role` | A role of contract §2, for example `anchor` for the baseline and `fp16` for the candidate |
| `weights` | A description of the weights |
| `forced_sha256` | The SHA-256 of the `forced.jsonl` that was read |
| `items` | One entry per item with `id`, `file` (a file name in the folder), `dtype` (`float16` or `float32`), `shape` (`[rows, vocabulary]`), `sha256` (of the file) and `tokens_sha256` (of the tokens that were read) |

An array file holds the logits of the scored rows of one item, little-endian and row-major, without a header: row i is
the output at position `start` + i of the forced record, over the vocabulary of `model.json`. The `--output` file that
`coreai-verify logits` writes, when its `--tokens` file holds the `tokens` of a forced record and `--start` is its
`start`, is such a file with `dtype` `float16`, and the `tokens_sha256` and `shape` that it prints are those of the
entry. Other fields of `logits.json` are not read.

The output folder gets `layer1.json`:

| Field | Value |
|---|---|
| `passed` | `true` only with no problem and every check passed |
| `problems` | The findings about the records, followed by those of each item prefixed by its id |
| `contract`, `limits` | The version and SHA-256 of `contract.json`, and the limits read from its `layer1` |
| `forced` | The SHA-256 of `forced.jsonl` and of `forced.json` of `--forced`, `null` for a file that was not read |
| `baseline`, `candidate` | The `role`, `weights` and SHA-256 of `logits.json` of each side |
| `vocabulary` | The vocabulary of `model.json` |
| `checks` | Per limit (`non_finite_values`, `max_probability_error`, `band_mean_max_probability_error`, `significant_top1_mismatches`) its value and `passed` |
| `bands` | Per band of `layer1.bands`, by the input token count of the items: the items, the compared items, the positions, `mean_max_probability_error`, `max_probability_error`, the report-only values and `state` |
| `overall` | The same values over all compared positions |
| `items` | Per item its band, `input_tokens`, `rows`, `problems`, the non-finite values of both sides, `max_probability_error`, `top1_matches`, `near_tie_mismatches` and `significant_mismatches` |

The probability error and the near ties are those of `compare` in `Tools/Conversion/adapters/verify_llm.py`. A band
`state` is `not_evaluated` without items, which does not count for `passed`; `passed` when every item of the band was
compared and its mean is within the limit; and `failed` otherwise. The report-only values, which decide nothing, are
`kl_mean`, `kl_p99` and `kl_max` (the KL divergence from the baseline to the candidate at each position),
`max_abs_logit_difference`, `top1_agreement` and `near_tie_count`. A forced input that cannot be made again, or that
differs from the one made again, fails the comparison. A finding about the logits of one item fails that item and the
comparison, and the other items are still compared. Every file that the command reads from the given folders goes
through `file_bytes` of `Tools/LLMEvaluation/layer1.py`: it is opened without waiting and read only when it is a regular
file, and an array file or a file of `--forced` only when it has the expected size. No array file of an item is read
once a finding about the item is known. The
command refuses an existing output, also a link, before it reads the records; once it has made the folder, it writes
`layer1.json` on every path, and a minimal failed record when the full record cannot be written. The exit code is 0
only when the comparison passed. `VerdictTests`, `CorrespondenceTests` and `BoundaryTests` in
`Tools/Conversion/tests/test_gemma4_31b_layer1_compare.py` fix these rules.
`python3 -m Tools.LLMEvaluation compare --layer1 <layer1.json>` names the record in a paired comparison.

## FP32 anchor and FP16 diagnostics

Six more heavy steps (#662) read the `reference` artifacts and run under the same shared lock, one process each. Each
step reads `reference.json` and the token files of the `reference` step once, requires a `reference` record with
`passed` true, each token file to have the size and SHA-256 that the record lists and every token ID to be an integer in the
vocabulary, and takes the forced sequence of every case from them (the input followed by the greedy output). It then
hashes the files it loads like `reference` (`verify_inputs`) and stops before loading when any of them differs. The
monitor, its stop conditions and `failure.json` are those of `reference`; the record is written only after the monitor
has ended without a stop. Each step also needs 20 GiB of free disk when it starts. `anchor`, `anchor_w16`,
`anchor_scale_fp32`, `anchor_scale_fp16` and `half` each save another 3.4 GB of logits, so a run of the recipe keeps
about 20.4 GB of logits with those of `reference`.

- `anchor` loads the BF16 weights on MPS and computes the forced sequence in FP32: right before each unit (the embedding,
  every decoder layer, the final norm and the output head) runs, every floating-point parameter and buffer of that unit
  is replaced by an FP32 value, and the BF16 value is put back right after it, so the FP32 copy of the whole model is
  never resident. The values come from a function from the tensor name to its value. With `--weights bfloat16` it is the
  loaded BF16 tensor in FP32; with `--weights float16` it is the loaded tensor cast to float16 and back to FP32.
  `--embed-scale` sets the value of the embedding scale (`model.language_model.embed_tokens.embed_scale`) to
  √5376 rounded to bfloat16 (73.5), float32 (73.32121) or float16 (73.3125); the original implementation multiplies
  by the scale cast to the weight dtype, so the BF16 reference uses 73.5. The
  layers, norms, attention, RoPE, the soft cap and the masks are the code of the original implementation. Attention runs
  through the original `sdpa` function in blocks of 512 query rows, each with every key and value and the matching rows
  of the causal and sliding-window masks, which the Transformers mask functions build for the whole sequence without
  the causal shortcut. The saved logits have the layout of `reference` (`cases/<id>.logits.npy`, float32, every
  position; `cases/<id>.tokens.json` holds the forced tokens). The step records, for the residual stream before and
  after every layer, the attention output, the input of `down_proj` (gate × up), the input and output of the final norm
  and the output head before the soft cap, the largest absolute value, the case where it occurred and the number of
  elements above 65504 and above 32752. With `--weights float16` it also records, for every tensor that the model
  loaded (a tied tensor once), the elements that the cast changes, those that it turns from nonzero to zero, the
  elements that are not finite after the cast, those above 65504 in magnitude and the largest absolute error among the
  finite results. Its `passed` requires clean loading, no NaN or infinite value in any saved row, a clean monitor and
  recorded resources. The recipe runs it four times: `anchor` (BF16 weights, scale 73.5), `anchor_w16` (float16
  weights, scale 73.5), `anchor_scale_fp32` and `anchor_scale_fp16` (BF16 weights, the other two scales).
- `half` loads the BF16 weights on MPS like `reference` and replaces every BF16 parameter and buffer by its float16
  cast one tensor at a time. The embedding scale therefore keeps the BF16 value 73.5 (exact in float16), the constant of
  `anchor_w16`, while every other tensor has the dtype and value that `from_pretrained(dtype=torch.float16)` gives (a
  unit test compares every tensor on a small model; `from_pretrained` itself would compute the scale as 73.3125). It then
  runs the same one-pass forward as `reference` (`use_cache=False`) in float16. It saves the logits in the same layout and counts the values that are not finite per case. Its `passed`
  requires clean loading, a clean monitor and recorded resources, not finite logits.
- `layers` (role `verify`) loads layers 0 to 5 (five sliding-window layers and one full-attention layer) with the
  embedding, the final norm and the head in BF16 on MPS and requires the loading lists to be empty except the unexpected
  keys, which must be exactly the keys of layers 6 to 59 in `model.safetensors.index.json`. It computes every forced
  sequence (including `ja_long`, 2,066 positions) as `anchor` does with BF16 weights, then converts the same model to
  FP32 with `model.float()` (the embedding scale stays 73.5) and runs the plain forward of `reference`. Its `passed`
  requires clean loading, finite logits on both paths, a largest absolute logit difference of at most 1e-4 for every
  sequence, a clean monitor and recorded resources.

Compare the `reference.json` records of two of the steps `reference`, `anchor`, `anchor_w16`, `anchor_scale_fp32`,
`anchor_scale_fp16` and `half` with the reference environment. The `layers` step writes `layers.json`, which this
comparison does not read:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.anchor_compare \
  --baseline <artifacts of the baseline step> --candidate <artifacts of the candidate step> > <comparison.json>
~~~

The result is written to standard output as JSON and nothing else is written there; the exit code is 0 only when
the comparison passed, and `problems` lists the findings otherwise. A `reference.json` that is missing or is not JSON
ends the command with an error and no result. `CompareTests` in `Tools/Conversion/tests/test_gemma4_31b_anchor.py`
fixes the checks on the records and the case files.

For every row, with p and q the softmax (float64) of the baseline and the candidate, it computes the largest |p − q|
over the vocabulary, KL(p ‖ q), the largest absolute logit difference and whether the two argmax IDs (the smaller ID
on a tie) agree; a mismatch is a near tie when both sides have the same top-two IDs and the baseline probability of
the baseline winner exceeds that of the candidate winner by at most 0.01. Per case and per group it reports the
maximum and the mean of the largest |p − q|, the mean, 99th percentile and maximum of KL, the largest logit
difference, the top-1 matches, the near-tie mismatches and the other mismatches, over every row and over the scoring
rows, and it lists every mismatch. The groups are the case kind (`chat` or `raw`) by input-length band (1–512,
513–2,048, 2,049–4,096, 4,097–16,896 tokens, `other` for any other length; only the bands with cases), and the rows
992 to 1,056 of every case with more than 1,056 positions (the sliding window of 1,024 drops the BOS from position
1,024). A group without rows is not reported.

`anchor_summary` prints the public summary (only states, numbers, hashes and fixed words) to standard output:

~~~sh
python3 -m Tools.Conversion.adapters.gemma4_31b.anchor_summary --report <report.json> \
  --comparison <comparison.json> [--comparison <comparison.json> ...]
~~~

It stops and prints no summary when a check on the report, on the records of its steps or on the given comparisons
fails; `SummaryTests` in the same test file fixes these checks.

## Converter environment and model class

### Converter environment

`local setup --recipe gemma4_31b` also prepares `.local/conversion/upstream-gemma4-31b/coreai-models`: a checkout of
Apple's public converter at `upstream_revision` of `model.json`, with the frozen lock of that checkout installed in
its `.venv`. An existing checkout is used only when it is a clean checkout of that revision; otherwise setup stops
without changing it. This converter environment is separate from the reference environment of the steps above.

### Model class

`adapters/gemma4_31b/model.py` defines `Gemma4ForCausalLM`, a subclass of the converter's `BaseForCausalLM`. It
imports the converter, so it runs only in the converter environment.

- `main` takes `input_ids` [1, n] and `position_ids` [1, P + n] (the positions of the P cached tokens followed by
  those of the n new ones) and returns, in this order, `logits` [1, n, vocabulary], `hidden_states` [1, n, 5376], the
  output of the final norm that the head reads, and `tap_hidden_states` [1, n, 5376 × the number of tap layers];
  `prefill` takes the same inputs and returns `tap_hidden_states` only.
- `tap_hidden_states` joins, in ascending order, the output after the first i layers for each tap layer i: the output
  of layer i − 1 counted from 0, after its `layer_scalar` and before the final norm, which is entry i of
  `output_hidden_states` of the original implementation for every layer but the last (the original implementation
  puts the output of the final norm there). The attribute `tap_layers` of the class holds the tap
  layers. The tools take them as `--taps`, ascending distinct layer counts from 1 to the number of layers separated by
  commas (`1,17,29,47,59` for 60 layers, `1,2,3,4,5` for the first six), without a default, and refuse other values
  before they create a folder or read a weight (`check_taps` in `layers.py`).
- The states are `keyCache` and `valueCache` [full-attention layers, 1, 4, ?, 512] and `slidingKeyCache` and
  `slidingValueCache` [sliding-window layers, 1, 16, ?, 256] (the `forward` arguments `k_cache`, `v_cache`,
  `sliding_k_cache` and `sliding_v_cache`); the sequence is their only dynamic dimension.
- Sliding-window layers cut the cached keys and values to the last positions that the window can reach and mask
  them with the values of `position_ids`. No position or mask comes from an `arange` of a dynamic length, and the
  composite SDPA's `window_size` is not used.
- `from_hf_memory_efficient(..., num_layers=N, hf_config_attr="text_config",
  hf_state_dict_prefix="model.language_model.")` reads the first N layers; their states follow the pattern of five
  sliding-window layers and one full-attention layer.
- A configuration that the class does not cover stops with `ValueError` before any weight is read (`text_config` in
  `model.py`).

### FP32 outputs of the first layers

`adapters/gemma4_31b/layers.py` writes the FP32 `logits`, `hidden_states` and `tap_hidden_states` of the first N layers
of either implementation for two fixed inputs, `short` (26 tokens) and `long` (1,400 tokens, longer than the window of
1,024), and compares two such runs:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers run \
  --implementation original --layers 6 --taps 1,2,3,4,5 --weights float16 \
  --model Tools/Conversion/recipes/gemma4_31b/model.json --snapshot .local/conversion/models/google--gemma-4-31B-it \
  --output <new folder>/original
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers \
  run --implementation class --layers 6 --taps 1,2,3,4,5 --weights float16 \
  --model Tools/Conversion/recipes/gemma4_31b/model.json --snapshot .local/conversion/models/google--gemma-4-31B-it \
  --output <new folder>/class
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers compare \
  --original <new folder>/original --model-class <new folder>/class > <new folder>/comparison.json
~~~

`run` loads the weights: start it under the shared machine lock, which it does not take itself. It accepts only a
layer count whose last layer is a full-attention layer (6, 12, …, 60 here) and tap layers of that count, and refuses
any other before it creates the folder or reads a weight. `--weights float16` rounds every checkpoint tensor to
float16. `--weights 8bit --export <export folder>`, for the original implementation only, puts in the weights restored
from the quantized tensors of an 8-bit export, as `context` does
([FP32 reference restored from the quantized weights](#fp32-reference-restored-from-the-quantized-weights)), refuses
an existing output before it reads the export, and records `weights` and `quantized` as `context` does. It writes `<input>.<output>.npy` for both inputs and the
three outputs and, last, `record.json` into a new folder. The original implementation reads each input in one call;
its tap layer i is entry i of `output_hidden_states`, except for the last layer, whose entry the original
implementation replaces with the output of the final norm: that tap is the output of the last layer, taken with a
forward hook. The model class reads the window and then calls of 1, 2, 4 and 8 tokens with its caches.
`compare` passes only for a run of the original implementation and a run of the model class on the same inputs and
tap layers whose differences stay below the criteria: for `short` a largest absolute logit difference below 1e-3, and
for `long` a largest absolute logit difference below 5e-3, a largest probability difference below 1e-6 and no
position with another `argmax`; for both inputs, at every position of `hidden_states` and of each tap layer of
`tap_hidden_states`, a largest absolute difference below 1e-3 (`ratio_limit`) times the largest absolute value of the
original implementation at that position. `largest_ratio` is the largest quotient of the two over the positions.
`CompareTests` in `Tools/Conversion/tests/test_gemma4_31b_layers.py` fixes the criteria, and `ModelClassTests` in
`Tools/Conversion/tests/export_gemma4_31b_model.py` the outputs of the class against the original implementation on a
synthetic model.

### Tests

The tests of the model class are in `Tools/Conversion/tests/export_gemma4_31b_model.py`, those of the drafter class in
`Tools/Conversion/tests/export_gemma4_31b_drafter.py`. The default pattern of
`unittest discover` (`test*.py`) does not collect them; run them in the converter environment, where
`test_gemma4_31b_layers.py` also runs, and run every `test_gemma4_31b_*.py` in the reference environment:

~~~sh
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m unittest discover -s Tools/Conversion/tests \
  -p 'export_*.py' -v
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m unittest discover -s Tools/Conversion/tests \
  -p 'test_gemma4_31b_layers.py' -v
.local/conversion/gemma4_31b/.venv/bin/python -m unittest discover -s Tools/Conversion/tests \
  -p 'test_gemma4_31b_*.py' -v
~~~

## FP16 export of the first layers

`recipe-fp16.json` exports the first layers of the pinned snapshot to a Core AI bundle for macOS with dynamic shapes
and FP16 weights, without compression, and inspects the asset. It runs the `prepare` step of `recipe.json` and a heavy
`export` step in the converter environment. It needs `upstream`, the converter checkout, which `local setup --recipe
gemma4_31b` writes into the bindings, and three values without defaults: `layers`, a count whose last layer is a
full-attention layer (6, 12, …, 60), `functions`, `main` or `main,prefill`, and `taps`, the tap layers of
`tap_hidden_states` ([Model class](#model-class)). A run without them stops before any
step. `context`, the context limit of the asset, defaults to 16,896 and has to be larger than 2,048, the cache length
that the exporter traces with; otherwise the exporter pins the cache dimensions and the inspection fails. The `python`
of the recipe is the converter environment, so the runner does not reuse an export after that environment changes.
`local plan|run` does not select this recipe:

~~~sh
python3 -m Tools.Conversion run Tools/Conversion/recipes/gemma4_31b/recipe-fp16.json \
  --bindings .local/conversion/gemma4_31b.paths.json --set layers=6 --set functions=main,prefill --set taps=1,2,3,4,5 \
  --work-dir .local/conversion/work/gemma4_31b-fp16 --lock-dir ~/Library/Caches/coreai-samples/heavy-job
~~~

The step writes `export/` into its artifacts: `export.json`, the record, and the bundle with `metadata.json`,
`gemma4_31b_l<layers>_fp16_main.aimodel` (`gemma4_31b_l<layers>_fp16_main_prefill.aimodel` with `main,prefill`) and
`tokenizer/`, the three distributed tokenizer files. The bundle is saved as `bundle.incomplete/` and gets the name
`bundle/` only after a passed run has written `export.json`; the exit code is 0 only for a passed run. A run passes
when the export had no compression setting and no quantized weight, every parameter and buffer of the PyTorch model is
float16, the only storage types of the asset are Float16, Float32, Int32, UInt32 and UInt64 and Float32 holds exactly
`global_head_dim // 2 + head_dim // 2 + 2` elements (two RoPE frequency tables and two constants; 386 for this model),
the asset has exactly the functions asked for, with the inputs, outputs and states that the configuration of the
contract and the tap layers give and the sequence as their only dynamic dimension, `main` returns `logits`,
`hidden_states` and `tap_hidden_states` in this order (`output_order` in `export.json`), the tokenizer files match the
contract and `metadata.json` the contract and the arguments, including `tap_layers`, the tap layers, and
`tap_convention`, `layers-passed:before-final-norm:ascending`, the bundle holds exactly the files of a complete bundle
(`bundle_layout`: `main.mlirb`, `main.hash` and `metadata.json` of the asset, `metadata.json` and the three tokenizer
files) under the name that the layers, the functions and the precision give, and the volume kept 60 GiB free. The step
stages `main`
and `prefill` with the steps that `export_to_coreai` of the converter takes for an eager model, through its
non-public `_build_reference_inputs`, because that function passes no output name for `prefill`, whose output would
then take the name of a graph node. The storage counts are totals over the
asset, in which the export folds identical and constant-valued tensors, so they cannot show which tensor is stored in
which type; the type of each tensor is checked on the PyTorch model, and `export.json` records Float16 minus the
elements of the weights without judging it. The step starts only with twice the FP16 weights of the chosen layers plus
60 GiB free; the monitor, its stops and `failure.json` are those of `reference`. The loading work area `export/layers/`
is removed when the step ends, raises an error or is stopped by its monitor, and stays when a signal ends the process,
as when the runner stops the step at its timeout or on an interruption (SIGTERM, then SIGKILL). The inspection reads
the asset with
`coreai.runtime._AIModelAsset(...).summary(include_statistics=True)`, a non-public interface of the converter
package. `AssetTests`, `StartTests`, `RecordTests` and `RecipeTests` in
`Tools/Conversion/tests/test_gemma4_31b_export.py` fix the inspection, the checks before the start, the record and the
recipe; `ExportTests` in `Tools/Conversion/tests/export_gemma4_31b_export.py` runs the step on a synthetic model in the
converter environment (the `export_*.py` command of [Tests](#tests)).

### FP32 reference of the comparison

`layers.py context --weights float16` writes the FP32 logits of the first layers of the original implementation, with
the weights of `run` (every checkpoint tensor rounded to float16, computed in float32 on the CPU), for `short` and for
`context`, a
sequence of 16,896 tokens made like the other inputs (seed 2). It reads each input with the cache of the original
implementation in calls of 1,024 tokens. Into a new folder it writes `short.npy` and `context.npy` (float32, only the
26 and 1,428 rows that the conditions below compare) and, last, `record.json`. Start it under the shared machine lock,
which it does not take itself, like `run`:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers context --layers 6 \
  --model Tools/Conversion/recipes/gemma4_31b/model.json --snapshot .local/conversion/models/google--gemma-4-31B-it \
  --weights float16 --output <new folder>
~~~

`ConditionTests` and `ContextReferenceTests` in `Tools/Conversion/tests/test_gemma4_31b_layers.py` fix the rows, the
refusals and the reading in calls, which gives the logits of one call within 1e-3 on a synthetic model.

### Comparison with the own engine

Build `coreai-verify` with `swift build -c release` in `Tools/Verify`, then run the comparison under the shared machine
lock, which it does not take itself:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.native \
  --export <artifacts of the export step>/export --reference <folder of layers.py context> \
  --verifier Tools/Verify/.build/release/coreai-verify --output <new folder>
~~~

It refuses an existing output before it reads anything. Into the new folder it writes, for every condition,
`<condition>.tokens.json`, the logits of `coreai-verify logits` (`<condition>.f16`) and its standard error
(`<condition>.log`), and last `comparison.json` with `passed`, the findings in `problems` and the numbers of every
condition, also when a record, the verifier or a condition cannot be read, or a failed record without conditions when
the full record cannot be written. It prints a short summary of
`comparison.json`, and the exit code is 0 only when the comparison passed. Each `coreai-verify` call is stopped after
1,800 s.

| Condition | Tokens | `--start` | `--chunk` | Compared rows |
|---|---|---:|---|---|
| `short-c1`, `short-c26` | `short` (26) | 0 | 1, 26 | 0 to 25 |
| `window-c1`, `window-c2`, `window-c4`, `window-c8` | first 1,300 of `context` | 0 | 1, 2, 4, 8 | 0 to 1,299 |
| `prefix-c1` | first 1,300 of `context` | 1,000 | 1 | 1,000 to 1,299 |
| `capacity-2048`, `capacity-4096`, `capacity-8192`, `capacity-16384` | first 2,056, 4,104, 8,200, 16,392 of `context` | 2,040, 4,088, 8,184, 16,376 | 4, 8, 2, 4 | the last 16 |
| `tail` | `context` (16,896) | 16,832 | 1 | 16,832 to 16,895 |

A condition passes with no non-finite value in either logits, no constant row (a row whose largest and smallest values
are equal), a largest absolute logit difference from the reference below 2.0 and no top-1 change outside a near tie,
where the top two reference logits differ by 0.01 or less. The rows of a call from `--start` of 8 or more tokens after
which the positions total 97 to 112 (the call shape that `Engine.callShapeRules` avoids for this model; rows 96 to 111
of `window-c8`) are left out of the difference and the top-1 change and recorded apart; non-finite values and constant
rows count on every row. The record that `coreai-verify` prints has to match the condition, including the function,
the number of calls and the chunk of the reads before `--start`, which follow the rules of the engine.
`MeasureTests`, `KnownCallTests` and `CompareTests` in `Tools/Conversion/tests/test_gemma4_31b_native.py` fix these
criteria and the checks on the records, the bundle, the record of `coreai-verify` and its files, `measure_condition`,
which measures a condition from its logits file and the reference for the comparison and for the summary below, and
`measurement_problems` and `condition_problems`, which compare the numbers of a record with it and read the fields and
the relations that a condition can hold on each path of the comparison. The specialization cache that `coreai-verify`
makes stays in place; remove it yourself.

`fp16_summary` prints the public summary (only states, numbers, hashes and fixed words) of the reference record, of
the comparison of the model class with the original implementation, which it makes itself with `layers.py compare`
from the folders of the two runs, and of each export with its report and its comparison to standard output:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.fp16_summary \
  --reference <record.json> \
  --original <folder of layers.py run --implementation original> --model-class <folder of the model-class run> \
  --report <report.json> --export <export.json> --comparison <comparison.json> [...]
~~~

It stops and prints no summary when a check on the records fails, including that each `export.json` is the output that
the manifest of its stage recorded and lists the bundle files (and for 8 bits `quantized.safetensors`) of that manifest,
that the two runs, the reference and every export are of the contract `model.json` of the repository, that the two runs
are of the layers and tap layers of every export, that the reference is of the layers of every export and uses its kind
of weights (restored from its quantized tensors for 8 bits, with the SHA-256 of its record, the size and SHA-256 of
`quantized.safetensors` and the number of tensors, through `correspondence` of `native.py`), and that each comparison
records the layers, the functions and the asset of its export and the layers of the reference. It reads the report, the
export record and the comparison of each export, the record of the reference, the manifest of each stage and the arrays of
the references only as regular files (not a link, a pipe or a folder), and an array only at its recorded size. The
arguments of the runs (layers, tap layers, functions, precision and context limit) are checked
for their form and for their agreement with the contract, the other records and the names and the metadata that they
give. Every other value that it prints is of one of six kinds, and each kind has one rule:

- A fixed value must equal the constant of the tools or the formula that derives it, and a record that differs is
  refused: the twelve conditions in their order with their inputs, token counts, starts, chunks and rows, the criteria,
  the call shape that the engine avoids, the stages of the report, the chunks, calls, embedding scale, logits type and
  shape and inputs of the reference, and the sizes and counts that they give (the size of each logits file of the
  comparison, the record that the verifier gives for the reads of each condition, the counts of the export that follow
  from its layers, its tap layers and its checkpoint).
- A verdict (`passed` of the comparison, `passed` of each condition and of the export, the checks of the export, and the
  comparison of the model class with the original implementation, which it makes itself) is computed again from the
  numbers and the fixed values with the function that wrote it (`native.py`, `load_export`, `layers.py compare`), and a
  record whose verdict differs is refused.
- A number that the arrays give is computed again from the arrays and must equal the record, as normalized JSON. For
  each condition that reached its numbers, `measure_condition` of `native.py`, the function that the comparison wrote the
  numbers with (it maps the rows of the condition to the rows of the reference, marks the known rows and calls `measure`),
  measures the logits file of the condition, whose size and SHA-256 must be those of the record, against the logits of
  the reference that `load_reference` read, and the rows, the non-finite values of both, the constant rows, the largest
  difference, the top-1 changes, the known rows and the checks of the condition must equal those of the record. The
  count of the non-finite logits that the reference records for each input is counted again from its logits. For 8 bits
  the effect of the quantization is computed by `effect.py` from the restored reference (`--reference`) and the folder of
  the rounded reference (`--rounded`) in place, and it stops when the effect has a problem or was not computed from the
  record of the restored reference. The summary evaluates no relation among these numbers, because the equality gives
  them all.
- A measurement of a run that no array gives (a time, a reading of memory, disk or swap, a reading of the monitor, the
  counts and the cache hit that the verifier records) is copied after its type, its range and the relations of the table
  below, which its writer gives it. The readers of the records return the problems of a field of another type alone, and
  evaluate no relation on it.
- A hash is checked when it leads to a file that the summary reads (the files of the bundle, `quantized.safetensors`,
  the records and the logits files of the reference, of the rounded reference and of the comparison, the contract) and
  copied when it does not (the SHA-256 of the verifier).
- A version of a package is copied, for exactly the packages that its writer records.

| Relation of measurements of a run | Why it holds |
|---|---|
| The time of a condition is at least the sum of the steps that its verifier recorded. `specialize_s` is 0 when the asset was in the cache, above 0 when it was not, and within `load_s` | `loadS`, `prefixS` and `scoringS` are consecutive intervals of the process of `coreai-verify` (`PositionLogits.swift`), and `specialize_s` is measured inside the load (`Engine.resolveModel`) |
| The verifier counted other non-finite values than the file exactly when the condition records a problem | The comparison records a problem when the count of the verifier differs from that of its file |
| The time of an export is at most the `execute_s` of the stage that ran it. Times that make up another time (the steps of a stage, the stages of a run, the steps of an export or of a reference) do not exceed it | The runner measures around the process of the step, and the export reads its time inside that process |
| The resident memory that `ps` gave the monitor is at most the peak that `getrusage` gave it. The context limit of an export is at least the 16,896 tokens of the longest condition | `ru_maxrss` is the peak of the resident size that `ps` reads at each sample, and the comparison reads up to 16,896 tokens |

It does not check the measurements of a run beyond these types, ranges and relations, the number of tensors of the
PyTorch model (`weights.tensors`, which would need a copy of the structure of the model class), the SHA-256 of the
verifier (the file is not read) or the values of the versions. It also leaves unchecked what no writer relates to
another number: the peak footprint of the process (a different measure from the resident size), the counts of the
constants that the asset stores (`Int32`, `UInt32`, `UInt64`) and an upper bound for the time of a condition (the
verifier is stopped after 1,800 s, and the checks after it add to the time).

Where it prints a mapping of a record key by key, a constant or the reader of the record (`load_export`,
`load_reference`, `reference_record_problems`, `condition_problems`) fixes the set of keys. It needs NumPy and PyTorch,
reads every logits file of the comparison and holds one in memory while it measures it (681,574,400 B for the largest).
A comparison that fails a criterion is printed with `passed` false, as a failed condition of the own engine is;
`SummaryTests` and `QuantizedSummaryTests` in `Tools/Conversion/tests/test_gemma4_31b_fp16_summary.py` fix these
checks. The published record is
[docs/validation/gemma4-31b](../../../../docs/validation/gemma4-31b/README.md#outputs-for-speculation-in-the-first-six-layers-695).

## 8-bit export of the first layers

`recipe-8bit.json` is `recipe-fp16.json` with `--precision 8bit` and `--compression compression-8bit.yaml`, and its
`functions` can be `main,prefill` only: the export refuses `main` alone with `--precision 8bit`. Before it stages the
functions, the `export` step quantizes the weights of the `torch.nn.Linear` layers of the model class with
the converter's `quantize_for_export` and the config of `compression-8bit.yaml`: uint8, asymmetric, blocks of 32 along
the input axis, float16 minimums, eager execution, no activation or KV cache quantization. The embedding, the shared
head (the embedding table), the norms and `layer_scalar` stay float16. The step reads the file with
`_load_compression_config_object` of `coreai_models.llm.export`, a non-public function of the converter package. It
takes the same `layers`, `functions` and `taps`, and the asset has the outputs of the FP16 asset:

~~~sh
python3 -m Tools.Conversion run Tools/Conversion/recipes/gemma4_31b/recipe-8bit.json \
  --bindings .local/conversion/gemma4_31b.paths.json --set layers=6 --set functions=main,prefill --set taps=1,2,3,4,5 \
  --work-dir .local/conversion/work/gemma4_31b-8bit --lock-dir ~/Library/Caches/coreai-samples/heavy-job
~~~

Next to `export.json` and the bundle (`gemma4_31b_l<layers>_8bit_<functions>.aimodel`, with `compression` `8bit` in
`metadata.json` next to `tap_layers` and `tap_convention`), the step writes `quantized.safetensors`: the uint8 data,
the float16 scales and the float16 minimums of every quantized weight as the step passed them to the converter, with
the SHA-256 of each tensor in `export.json`.
The asset summary does not give the tensors one by one, so these hashes are of the tensors that the step passed to the
converter; the asset itself is checked through its operations and through the logits of the own engine below. The
converter's work area `export/quantization/` is removed with `export/layers/`. An error or an interruption (Ctrl-C,
`KeyboardInterrupt`) after the step made `export/`, with either precision, leaves `export/failure.json` with the type of
the error; when the monitor stopped the step, its own `failure.json` with the reason stays and is not overwritten. A
process ended by SIGTERM or SIGKILL, as the runner ends a step at its timeout, leaves no record. The step starts only
with seven bytes per weight element of the chosen layers plus 60 GiB free. `QuantizedAssetTests` in
`Tools/Conversion/tests/test_gemma4_31b_export.py` fixes the checks of a passing run. Their values: the fixed config,
the Linear weights of the layers as the quantized tensors (41 for six layers, 410 for 60; the full-attention layers have
no value projection), no `Skipping quantization.` from the converter, and as many `blockwise_shift_scale` operations
(the dequantization of a weight) in the asset as the functions read quantized weights: every one in `main`, and in
`prefill` all but the query, output and MLP projections of the last layer, unless the last layer is a tap layer (77
for six layers with the tap layers 1 to 5 and 82 with 6, 815 for 60 with 1, 17, 29, 47 and 59). `export.json` records
the UInt8 and Float16 counts of the asset minus the uint8 and float16 elements of the PyTorch model without judging
them.
`QuantizedExportTests` in `Tools/Conversion/tests/export_gemma4_31b_export.py` runs the step on synthetic models in the
converter environment.

### FP32 reference restored from the quantized weights

`layers.py context --weights 8bit --export <the export folder of the step>` writes the reference of the comparison with
the weights that the export quantized. It reads the export with `load_export` of `native.py`, the one definition of an
export that may be used, which the comparison below also applies: a bundle that holds exactly the files of a complete
bundle (`bundle_layout` of `export.py`) under the name that its layers, functions and precision give, with the sizes
and SHA-256 of its record, the tokenizer files of the contract, `metadata.json` built from the contract and the record,
`main.hash` the SHA-256 of `main.mlirb` and the size of the bundle the sum of its files; the bundle, its asset and its
tokenizer folder real folders, and `export.json`, every file of the bundle and `quantized.safetensors` regular files, not
symbolic links or pipes, each opened without waiting and read only when its size is that of its record; no
`failure.json` and no `bundle.incomplete` next to it; `passed` the JSON true, no problem and both checks passed with
60 GiB free; the schema, the contract, the converter checkout, the layers, the function layout, the precision, the
context limit and the tap layers checked by type (a boolean is not taken for a number); the inspection of the export
step run again on the record (`check_asset`), with each number that the export derives from others consistent with them
(the element counts of the weights add up and follow from the size of the checkpoint and, for 8 bits, from the shapes of
the quantized weights; the recorded differences of the storage counts follow from them; the disk record follows the rule
of the free disk; the swap record, the lowest free disk and the peak resident memory follow from the monitor, which ran
under its fixed limits and did not stop the export, whose record has the keys that the monitor writes, at least one
sample and no more samples than its interval allows in the time of the export, with one sample its peak swap and its
lowest free disk those of that sample, a positive resident memory that ps gave in KiB and a free memory percentage of at
most 100, and the swap increase stays within its limit; the times are those of the steps of the export and their total
is not shorter than them); and for 8 bits exactly the Linear weights of the layers as uint8 with float16 scales and
minimums in blocks of [1, 32] and no zero point, their tensors, and the size and SHA-256 of `quantized.safetensors`.
Before it creates its folder, it also checks each tensor against its SHA-256 in `export.json`, its weight against the
shape in the checkpoint, whose index and files must be regular files, and its scales and minimums against exactly [rows,
columns / 32]. It
then restores each weight as uint8 × scale + minimum with the blocks widened to the weight and each operation rounded to
float16 (the dequantization of the converter), and puts it into the original implementation in
float32. The other tensors are rounded to float16 as with `--weights float16`. `record.json` records the kind of the
weights, the SHA-256 of `export.json`, the size and SHA-256 of `quantized.safetensors` and the number of tensors that
it checked. An error or an interruption after the folder exists, with either kind of weights, leaves `failure.json`
with the type of the error:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers context --layers 6 \
  --model Tools/Conversion/recipes/gemma4_31b/model.json --snapshot .local/conversion/models/google--gemma-4-31B-it \
  --weights 8bit --export <artifacts of the export step>/export --output <new folder>
~~~

`RestoredReferenceTests` in `Tools/Conversion/tests/test_gemma4_31b_layers.py` and
`test_restoring_gives_the_weights_that_the_converter_dequantizes` in the converter tests fix the formula and the
refusals.

### Comparison and the effect of the quantization

`native` compares an 8-bit export only with the reference restored from its own `quantized.safetensors`, and an FP16
export only with a reference of float16-rounded weights, with the conditions and criteria of
[Comparison with the own engine](#comparison-with-the-own-engine). `effect` prints to standard output, without a
judgment, how far the restored reference is from a reference of float16-rounded weights on the rows of each condition:
the largest absolute logit difference and the top-1 changes inside and outside a near tie of the rounded reference.

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.effect \
  --restored <restored reference folder> --rounded <float16 reference folder>
~~~

It refuses a restored reference that did not record the SHA-256 of the export record and the check of the data, scale
and minimum of every quantized weight, and two references that hold the same logits of an input (a hard link or a copy
of the same file). `EffectTests` in `Tools/Conversion/tests/test_gemma4_31b_effect.py` fixes it.
`fp16_summary` summarizes 8-bit records
with the restored reference as `--reference` and the folder of the rounded reference as `--rounded`, and without
`--original` and `--model-class`, which only FP16 records take. It computes the effect itself with the function of
`effect.py` from the two folders and does not read the output of `effect`; it stops with its fixed error when an input
of the other precision is given or one of its own is missing, and when `effect` finds a problem (a rounded reference of
other weights, of another contract, of other layers than the restored reference, not complete with its logits files, or
holding the same logits as the restored one) or did not start from the record of the restored reference.
`QuantizedSummaryTests` fixes these checks. The published record is
[docs/validation/gemma4-31b](../../../../docs/validation/gemma4-31b/README.md#8-bit-export-of-the-first-six-layers).

## Outputs for speculation read by the own engine

`speculation` compares the outputs for speculation that `coreai-verify logits` reads from an export with
`--main-output` and `--prefill-output` (`hidden_states` and `tap_hidden_states` of `main`, `tap_hidden_states` of
`prefill`) with a run of `layers.py` of the original implementation that holds the output of every layer, with the
weights of the export: `--weights float16` for an FP16 export, and `--weights 8bit --export <the same export folder>`
for an 8-bit export. Start both under the shared machine lock, which they do not take themselves:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers run \
  --implementation original --layers 6 --taps 1,2,3,4,5,6 --weights float16 \
  --model Tools/Conversion/recipes/gemma4_31b/model.json --snapshot .local/conversion/models/google--gemma-4-31B-it \
  --output <new folder>
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.speculation \
  --export <artifacts of the export step>/export --reference <folder of the run> \
  --verifier Tools/Verify/.build/release/coreai-verify --output <new folder>
~~~

It refuses an existing output before it reads anything, and stops without writing anything when its inputs cannot be
compared. It runs `coreai-verify` once per condition, in a new process, and stops each call after 1,800 s:

| Condition | Tokens | `--start` | `--chunk` | Compared rows |
|---|---|---:|---:|---|
| `short-c1`, `short-c8` | `short` (26) | 0 | 1, 8 | `main` 0 to 25 |
| `long-main` | `long` (1,400) | 0 | 1,400 | `main` 0 to 1,399 |
| `long-prefill` | `long` (1,400) | 1,100 | 1 | `prefill` 0 to 1,099, `main` 1,100 to 1,399 |

`long-main` reads the input in one `main` call, the call that `Engine.callLengths(from:count:)` gives for 1,400
tokens from position 0. `long-prefill` reads the first 1,100 positions, past the window of 1,024, with the `prefill`
function and the rest with `main` one token at a time.

For each output, and for `tap_hidden_states` for each of its tap layers, a condition passes with:

| Check | Value |
|---|---|
| Mixup | the sum over the rows of the squared differences from the reference of the same layer is at most 1/2 of the sum from every other layer of the reference (1 to the layer count); for `hidden_states`, from the final norm at most 1/2 of the sum from the last layer before the norm |
| Cosine | the cosine of every row with the reference of the same layer (for `hidden_states`, the final norm) at least 0.99, and their median at least 0.999 |
| Finite | no non-finite value |

`SectionTests` and `CompareTests` in `Tools/Conversion/tests/test_gemma4_31b_speculation.py` fix these values and the
checks on the inputs, the record of `coreai-verify` and its files. Into the new folder it writes, for every condition,
`<condition>.tokens.json`, the logits (`<condition>.f16`), the outputs (`<condition>.hidden_states.f16`,
`<condition>.tap_hidden_states.f16` and, for `long-prefill`, `<condition>.prefill.tap_hidden_states.f16`) and the
standard error of `coreai-verify` (`<condition>.log`), and last `comparison.json`: for every output its rows, width,
SHA-256 and non-finite values with their rows, and for an output without them the sums of squares by layer of the
reference, the nearest other layer and the ratio to it, the smallest cosine with its position, the median and the rows
below 0.99, and the checks. A
folder is complete only with `comparison.json`: an error or an interruption after the folder exists removes it and
leaves `failure.json` with the type of the error. It reads the reference, the verifier and the files that
`coreai-verify` writes only as regular files, of the recorded size where a record gives one, and the arrays of the
reference only in the `.npy` format. It prints a short
summary, and the exit code is 0 only when the comparison passed. The specialization cache that `coreai-verify` makes
stays in place; remove it yourself.

## MTP drafter

`adapters/gemma4_31b/drafter.py` defines `Gemma4AssistantDrafter`, a model class of the MTP drafter
[google/gemma-4-31B-it-assistant](https://huggingface.co/google/gemma-4-31B-it-assistant) for a macOS export with
dynamic shapes. It imports the converter, so it runs only in the converter environment. `adapters/gemma4_31b/mtp.py`
compares it in float32 on the CPU (with the weights rounded to float16) with the drafter of the reference environment
(Transformers 5.16.1) on the features that #691 recorded, follows the accepted drafts on the teacher path, and exports a
synthetic drafter. `assistant.json` pins the revision `627c5ec1458b9086b841a91e0512fd31fd2fbbf1` and the size and
SHA-256 of the two files that are read, `config.json` and `model.safetensors`. Before `run` creates its folder and
before `run` or `acceptance` reads a tensor, they check the arguments (`--steps` a whole number from 1; for `run` also
that `--implementation` is `original` or `class` and that its output folder does not exist) and then, in this order: the
two contracts (including the kinds of the values of `model.json` that the tools read: whole numbers from 1 for the
widths, vocabulary, heads, window and layers, the two layer kinds, and stop tokens below the vocabulary); the size and
SHA-256 of the index and the shard of `model.json` that hold the embedding table of the target, and of the two files of
the assistant; the records of #691, with the names, BF16 type and shapes of the tensors of their features that are read;
the configuration with `assistant_config`; the names, BF16 type and shapes of every tensor of the assistant checkpoint;
and the name, BF16 type and shape of the embedding table in the shard. The size and SHA-256 of the features are taken
last of these; `acceptance` then also checks `trajectories.json` and the drafter records of #691 (see Accepted drafts)
before it reads a weight.

Values from these files are compared with what they must equal only after their kind is checked, or as normalized JSON,
so a `20.0` or `true` never stands for `20` or `1`; the one exception is the comparison of the two reads of `model.json`
below. Every file that comes from outside must be a regular file (and, where the contract or the record gives it, of
that size) before its content is read, except that the drafters read `config.json` of the assistant a last time and
check only that it is a regular file; where the tool reads a file itself, it checks a descriptor opened without waiting.
`load_contract` of `prepare.py` reads `model.json` once more by its path, and the parsed values of the two reads must be
equal: that is a comparison of values, so `5376.0` equals `5376` there, the kinds are checked on the second read, whose
values are used, and the SHA-256 that is recorded is that of the first read. The index and `config.json` of the
assistant are also parsed from a second read after they are hashed, and `safetensors` opens a checkpoint, the shard or a
features file again by its path after the check. The files that the contracts pin and the tools read (the index and the
shard of `model.json`, `config.json` and `model.safetensors` of the assistant), the features and the arrays of a run
must not be links. A hash that a tool records is taken from the bytes it read for the hash.

### Drafter class

- `Gemma4AssistantDrafter(assistant_config(<config.json>, <target>), steps)`: the target is exactly the values of
  `model.json` that the drafter reads (`TARGET`, each a whole number from 1; other keys or another kind are refused),
  and `steps`, the number of drafts, has no default and must be a whole number from 1 (`ValueError` otherwise).
  `assistant_config` checks every value it reads against a table of kinds (whole numbers from 1 for widths and counts, a
  whole number from 0 for the width of the per-layer inputs, booleans for the flags, text for the activation, a positive
  finite float for `rms_norm_eps`, a list for the layer types and objects for the RoPE and the text configuration), so a
  width written as `32.0` is refused, and stops with `ValueError` for a configuration that the class does not cover:
  query heads that are not a multiple of either kind of key and value heads (the grouped attention of `SDPA`), odd head
  widths (both RoPE implementations rotate two halves of the head), layer types that are not one for each layer, with
  sliding-window and full-attention layers only and at least one of each, layers that compute their own keys and values,
  per-layer inputs, and values unlike those of the target or the fixed RoPE, activation and flags. A configuration that
  it accepts builds the class and the original drafter and computes a step (`BuildTests`). It and `checkpoint_shapes`,
  the tensors that the checkpoint must hold for the configuration that `assistant_config` returns, are in
  `adapters/gemma4_31b/assistant.py`, which imports neither PyTorch nor the converter, so the reference environment
  checks the same configuration.
- Inputs, in this order: `input_ids` int32 [1, 1], the last confirmed token at position a; `hidden_states` [1, 1, 5376],
  the output of the final norm of the target at position a − 1; `position_ids` int32 [1, 1], the value a; `k_sliding`
  and `v_sliding` [1, 1, 16, w, 256], the keys and values of the last sliding-window layer of the target (layer 58, slot
  49 of the sliding-window states) at the w = min(a, 1,025) positions a − w to a − 1; `k_full` and `v_full`
  [1, 1, 4, n, 512], those of the last full-attention layer (layer 59, slot 9) at all n = a positions 0 to a − 1. The
  keys are the stored ones, after `k_norm` and RoPE. The class has no mask: the caller cuts the ranges. `outputs` and
  `forward` check their inputs first and stop with `ValueError` (`The drafter inputs differ from the contract`) for a
  tensor of another shape, dtype or number of dimensions: `hidden_states`, the keys and the values must be in the dtype
  of the weights of the class (float16 in the export), the keys and the values of one shape, with w from 1 to
  `sliding_window + 1` and n from 1 without an upper limit.
- A token enters as its row of the target table times 73.5 (the BF16 value of √5376). The query uses position a at every
  step of a round. The logits are the product with the assistant's own embedding over the whole vocabulary, without a
  soft cap.
- `outputs(...)` returns the logits [1, 1, 262144] and the output of `post_projection` [1, 1, 5376] of one step.
  `forward(...)` returns `draft_tokens` int32 [1, steps]: each step reads the argmax token and the `post_projection`
  output of the step before, with the same keys, values and position. Steps after a stop token are computed; `cut` in
  `mtp.py` keeps the drafts up to and including the first stop token, where the original generator stops. The output of
  `post_projection` is no output of the function: the hidden state of the next round comes from the target.
- `export(drafters, context)` stages one function `draft_<steps>` per drafter with the steps of `staged` in `export.py`,
  so the drafters need different `steps`: the converter refuses two functions of one name. w (declared from 1 to
  `sliding_window + 1`, that is 1,025 for the assistant of the 31B model) and n (declared from 1 to `context`) are the
  dynamic dimensions of its inputs, which take the dtype of the weights of the class (float16 in `mtp.py export`). It
  traces each function with example inputs of 5 sliding-window positions and 9 full-attention positions, so
  `sliding_window` must be at least 4 and `context` at least 9. With less, `export` stops with an error from
  `torch.export` or the converter and returns nothing. Seen with torch 2.9.0 and `coreai-torch` 0.4.3: windows 1 to 3
  and a `context` of 2 to 8 stop with `RuntimeError` (for example `Your model failed to export: 5 not in range [2, 4]`
  for a window of 3 and `9 not in range [2, 8]` for a `context` of 8), and a `context` of 1, 0 or −1 stops earlier with
  `AssertionError` (for 1: `Cannot create Dim with inconsistent min=1, max=1`), which `torch.export.Dim` raises before
  the converter is called. The assistant of the 31B model (window 1,024) and the synthetic model of `mtp.py export`
  (window 1,024, `--context` above 1,025) meet both bounds.

### Comparison with the original drafter

Run both implementations under the shared machine lock, which the tools do not take themselves, then compare them:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.mtp run --implementation original \
  --model Tools/Conversion/recipes/gemma4_31b/model.json --assistant Tools/Conversion/recipes/gemma4_31b/assistant.json \
  --snapshot .local/conversion/models/google--gemma-4-31B-it \
  --assistant-snapshot .local/conversion/models/google--gemma-4-31B-it-assistant --records <records of #691> \
  --inputs <inputs.jsonl> --outputs <outputs.jsonl> --steps 7 --output <new folder>/original
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.mtp run \
  --implementation class <the same arguments> --output <new folder>/class
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.mtp compare \
  --original <new folder>/original --model-class <new folder>/class \
  --model Tools/Conversion/recipes/gemma4_31b/model.json --assistant Tools/Conversion/recipes/gemma4_31b/assistant.json \
  --snapshot .local/conversion/models/google--gemma-4-31B-it --records <records of #691> --inputs <inputs.jsonl> \
  --outputs <outputs.jsonl> --steps 7 > <new folder>/comparison.json
~~~

- `--records` is the record folder of #691: `subset.json` (48 items of the primary set of the quality contract with the
  SHA-256 of their input and output records; `run`, `compare` and `acceptance` refuse any other counts per category than
  the fixed ones of #691 in `SUBSET`, 13 categories, any other ids than `<category>-0001` onwards in that order, items
  without an input or an output token, and input and output records other than the input records and the reference
  outputs of the primary set of the quality contract, whose SHA-256 `RECORDS` pins) and `tf/features/<id>.safetensors`
  (the BF16 features of the original implementation under teacher forcing; the files of #691 hold other tensors too, and
  the tools read only `postnorm`, the output of the final norm, and the keys and values `k58`, `v58`, `k59` and `v59`).
  `--inputs` and `--outputs` are the input records and the outputs of the reference run of the set.
- Rows: the first item of each category in the numeric order of the ids, and in each item the anchors at the first
  output token, at a third and two thirds of the output and at the last output token, without repeats.
- Every checkpoint tensor is rounded to float16 and computed in float32 on the CPU. The original implementation gets the
  sliding-window keys and values of every recorded position up to a − 1 and keeps the last 1,025 with its own mask; its
  drafts come from its MTP generator `SinglePositionMultiTokenCandidateGenerator` with the stop tokens of `model.json`.
- `run` writes into a new folder `logits.npy`, `projected.npy`, `drafts.npy` (the generator drafts, padded with −1, or
  the unrolled drafts of the class), `repeated.npy` (the class only: `outputs` repeated `steps` times) and last
  `record.json`, or `failure.json` with the name of the error when it stops after making the folder.
- `compare` checks each of the two records against values it builds from the fixed inputs in the same process, not
  against the other record: the two contracts (content and SHA-256), the pins of the index and of the shard that the
  index names, `subset.json` (SHA-256) with its items checked against the pinned input and output records, the size and
  SHA-256 of the features of the chosen items (hashed again), the rows of the fixed selection, the stop tokens, window,
  layers and embedding scale of `model.json`, the weight rounding and `--steps`. Eleven keys of a record, `schema` (its
  fixed name), `steps`, `stops`, `window`, `layers`, `model`, `assistant`, `records`, `weights`, `embed_scale` and
  `rows`, hold these values and are compared as normalized JSON, so a missing or `null` entry fails even when both runs
  lack it. A record must hold exactly the keys of `RUN`, these eleven and `implementation`, `arrays`, `versions`,
  `timing` and `memory`: a missing key or an extra one fails. `implementation` must be `original` in the `--original`
  folder and `class` in the `--model-class` folder. `versions`, `timing` and `memory` (the package versions, the times
  and the memory of the run) are measured, not built from the fixed inputs: they only have to be present, their values
  are not compared and `null` passes. Each array file must hold one array with the size and SHA-256 in its record, the
  type, shape and count of non-finite values that the record gives, and the type and shape of those rows; a float array
  must be finite, and a token array must hold ids below the vocabulary (the original drafts at least one per row,
  followed only by −1 padding, and the arrays of the class no padding).
- `compare` passes when every row has a largest absolute logit difference of at most 1e-3, a largest absolute
  difference of the `post_projection` output of at most 1e-3 times its largest absolute value in the original
  implementation, and unrolled drafts equal to the repeated ones. A row whose original `post_projection` output is all
  zero has the ratio 0 when the output of the class is all zero too, and otherwise no bounded ratio (`null`) and fails;
  the largest ratio of the summary is then `null` too. It also records per row whether the argmax is the same and
  whether the cut unrolled drafts equal the generator drafts. After the fixed inputs are checked, a content of the two
  runs that breaks a check on their records or arrays (another container, an empty or zero-dimensional array,
  non-finite values, other types) gives a JSON result with `passed` false, the reasons in `problems` and no rows and no
  summary, and so does any error in the comparison, with the problem `the comparison stopped with` and the name of its
  type. Arrays that are well formed but outside a criterion, with extreme values for instance, give the full result
  with `passed` false, an empty `problems` and the failing checks of each row in `rows[].checks`. It prints its JSON
  result, and the exit code is 0 only when it passed.

`PureTests`, `RecordTests`, `CompareTests`, `ReadTests` and `RefusalTests` in
`Tools/Conversion/tests/test_gemma4_31b_mtp.py` fix the helper functions, the records, the reads and the criteria. In
`Tools/Conversion/tests/export_gemma4_31b_drafter.py`, `OneStepTests` compare the logits and the `post_projection`
output of one step with those of the original drafter of the converter environment on a synthetic assistant (at most
1e-4), at positions 1, 1,024, 1,025, 1,026, 4,000 and 16,895 and with large values at both sides of the window edge;
`UnrolledTests` compare the unrolled drafts of 1, 2 and 7 steps with one step repeated and with the drafts of the
original generator; `ContractTests` fix the inputs and configurations that the class and `assistant_config` refuse;
`BuildTests` require `assistant_config` to refuse 11 of 46 configurations at boundary values and build both drafters
from the other 35; `RunTests` run both implementations and the acceptance on recorded features of a synthetic target;
`ExportTests` convert the synthetic drafter.

### Accepted drafts on the teacher path

~~~sh
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.mtp \
  acceptance <the arguments of run without --implementation and --output> > <new file>.json
~~~

It drafts with the unrolled class at every anchor of the 48 items, cuts after the first stop token and counts the
leading drafts that match the teacher, the reference output. Along the teacher path, a round accepts at most `steps`
drafts and adds one token of the target. Per category, and for all items together as `pooled`, it prints the number of
items and of anchors, the rounds, the tokens and the tokens per round, next to those of #691 (`trajectories.json` of
`--records`), the number of anchors whose accepted drafts differ from those of the records of #691
(`drafters/<id>.json`, with their accepted counts capped at `steps`) and, of the top-two margin that #691 recorded at
the first draft that differs, how many are known, the largest and how many are at most 0.25. The accepted count of #691
at every anchor is computed again from its recorded drafts and the teacher, as for the class; a record whose `accepted`
differs at any anchor, with a draft token outside the vocabulary, or with a row that the drafting of #691 cannot give,
is refused before a weight is read, and so is a `trajectories.json` whose tokens per round are not its tokens over its
rounds rounded to four places (`null` without rounds), as #691 wrote them, and so is a `--steps` for which
`trajectories.json` records no fixed position (it records 1 to 7: `trajectories.json holds no per-round values of the
fixed position for 15 steps`). A row is one that the drafting of #691 can give when `available` is the remaining teacher
tokens up to 15, its margins are rounded to four places, and, with a minimum of either 0 or 6 drafts (the two settings
of #691), the drafting stops at its last draft and at no draft before: it stops after a stop token, after the fifteenth
draft, or, from the minimum on, after a draft beyond the remaining teacher tokens or unlike the teacher. `problems`
lists categories where the recomputed counts do not give the rounds and tokens of `trajectories.json`, an acceptance in
which no anchor was drafted, and an error after the inputs are checked (by the name of its type), with the JSON printed
in each case. The values are records, not criteria.

### Export of a synthetic drafter

~~~sh
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.mtp export \
  --steps 7 --context 16896 --seed 11 --output <new folder>
~~~

It converts the class with random weights of `SYNTHETIC` in FP16 as one step (`draft_1`), unrolled (`draft_<steps>`) and
both functions in one asset, inspects each asset and writes `export.json` with the functions, their inputs and outputs,
the storage types, the operations and the sizes. The exit code is 0 only when every asset has exactly the expected
functions, each with the expected input and output names, types, shapes and dynamic positions, no states and the single
output `draft_tokens`, and every parameter and buffer of the PyTorch models is float16 without quantized layouts. It
needs `--steps` of at least 2, a `--context` above 1,025 (the window of the synthetic model plus one) and an output
folder that does not exist, and refuses a call that does not meet them before it converts anything, with `ValueError`
or `FileExistsError` and without making a folder. `ExportTests` in
`Tools/Conversion/tests/export_gemma4_31b_drafter.py` runs it.
The published record is [docs/validation/gemma4-31b](../../../../docs/validation/gemma4-31b/README.md#mtp-drafter-class-704).

## Terms

Gemma 4 31B is distributed under Apache-2.0. The weights stay outside Git.
