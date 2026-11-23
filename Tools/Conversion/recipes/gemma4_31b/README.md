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
([FP16 export of the first layers](#fp16-export-of-the-first-layers)).

- `model.json`: the contract. It pins the size and SHA-256 of all twelve files of the revision, the main
  text configuration values (60 layers as five sliding-window layers followed by one full-attention layer,
  hidden size 5376, MLP width 21504, vocabulary 262144, window 1024, 16 and 4 key/value heads, head
  dimensions 256 and 512, shared keys and values in the full-attention layers, final logit soft cap 30,
  tied embeddings, BOS 2 and stop tokens 1, 106 and 50), the distributed tokenizer files, the reference
  precision and the upstream converter revision `c8e00c01800ba6ad324707007f7af358efc980a4` of the converter
  environment. A snapshot that differs fails; the contract is not adjusted to the files.
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
  --implementation original --layers 6 --taps 1,2,3,4,5 --model Tools/Conversion/recipes/gemma4_31b/model.json \
  --snapshot .local/conversion/models/google--gemma-4-31B-it --output <new folder>/original
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers \
  run --implementation class --layers 6 --taps 1,2,3,4,5 --model Tools/Conversion/recipes/gemma4_31b/model.json \
  --snapshot .local/conversion/models/google--gemma-4-31B-it --output <new folder>/class
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers compare \
  --original <new folder>/original --model-class <new folder>/class > <new folder>/comparison.json
~~~

`run` loads the weights: start it under the shared machine lock, which it does not take itself. It accepts only a
layer count whose last layer is a full-attention layer (6, 12, …, 60 here) and tap layers of that count, and refuses
any other before it creates the folder or reads a weight. It writes `<input>.<output>.npy` for both inputs and the
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

The tests of the model class are in `Tools/Conversion/tests/export_gemma4_31b_model.py`. The default pattern of
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
`tap_convention`, `layers-passed:before-final-norm:ascending`, and the volume kept 60 GiB free. The step stages `main`
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
criteria and the checks on the records, the bundle, the record of `coreai-verify` and its files. The specialization
cache that `coreai-verify` makes stays in place; remove it yourself.

`fp16_summary` prints the public summary (only states, numbers, hashes and fixed words) of the reference record, of
the comparison of the model class with the original implementation, which it makes itself with `layers.py compare`
from the folders of the two runs, and of each export with its report and its comparison to standard output:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.fp16_summary \
  --reference <record.json> \
  --original <folder of layers.py run --implementation original> --model-class <folder of the model-class run> \
  --report <report.json> --export <export.json> --comparison <comparison.json> [...]
~~~

It stops and prints no summary when a check on the records fails, including that each `export.json` is the output
that the manifest of its stage recorded, that the two runs, the reference and every export are of the contract
`model.json`, and that the two runs are of the layers and tap layers of every export. A comparison that fails a
criterion is printed with `passed` false, as a failed condition of the own engine is; `SummaryTests` in
`Tools/Conversion/tests/test_gemma4_31b_fp16_summary.py` fixes these checks. The published record is
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
export that may be used, which the comparison below also applies: the bundle under its final name with exactly the
files of the record and their sizes and SHA-256, no `failure.json` and no `bundle.incomplete` next to it, `passed` the
JSON true and no problem, the schema, the contract, the layers, the function layout, the precision and the tap layers
checked by type (a boolean is not taken for a number), and for 8 bits exactly the Linear weights of the layers as uint8
with float16 scales and minimums in blocks of [1, 32] and no zero point, their tensors, and the size and SHA-256 of
`quantized.safetensors`. Before it creates its folder, it also checks each tensor against its SHA-256 in `export.json`,
its weight against the shape in the checkpoint and its scales and minimums against exactly [rows, columns / 32]. It
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
and minimum of every quantized weight. `EffectTests` in `Tools/Conversion/tests/test_gemma4_31b_effect.py` fixes it.
`fp16_summary` summarizes 8-bit records
with the restored reference as `--reference` and the output of `effect` as `--effect`, and without `--original` and
`--model-class`, which only FP16 records take; it stops with its fixed error when an input of the other precision is
given or one of its own is missing. `QuantizedSummaryTests` fixes these checks. The published record is
[docs/validation/gemma4-31b](../../../../docs/validation/gemma4-31b/README.md#8-bit-export-of-the-first-six-layers).

## Terms

Gemma 4 31B is distributed under Apache-2.0. The weights stay outside Git.
