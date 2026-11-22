# Gemma 4 31B BF16 reference

This recipe pins [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475` and records the outputs of the original Transformers
implementation with the distributed BF16 weights, BF16 compute and MPS for twelve self-written diagnostic
inputs. The [FP32 anchor and FP16 diagnostics](#fp32-anchor-and-fp16-diagnostics) of this recipe and later work (the
Core AI comparison and quality checks) read these outputs. It does not convert the model, and the diagnostic inputs
are not a quality evaluation set. It also defines the converter environment and a model class for a macOS export
with dynamic shapes, with a tool that compares the first layers of that class with the original implementation in
FP32 ([Converter environment and model class](#converter-environment-and-model-class)). Nothing is exported to Core
AI yet.

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

Compare two of these records (or one of them and the `reference` record) with the reference environment:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.anchor_compare \
  --baseline <artifacts of the baseline step> --candidate <artifacts of the candidate step> > <comparison.json>
~~~

It reads each `reference.json`, token file and logits file once and uses the same bytes for the SHA-256 and the
values. A record must first have the shape of a BF16, `anchor`, `half` or `layers` record (the type of every entry that
the comparison reads, the cases and their files included); a record of another shape gets that one problem and no
further check. Each record is checked on its own against values computed from `model.json` and `fixtures.json`, not only
against the other record: the model entries (ID, revision, the SHA-256 of `model.json`, the shard SHA-256 and the files
verified before loading) must equal the contract, the cases must be the twelve cases of `fixtures.json` in order with
their IDs and kinds (the BF16 reference also with their `max_new_tokens`, and with the SHA-256 of `fixtures.json` in
`fixtures_sha256`), the device must be the contract device, and the record must be one of the known definitions: the BF16 reference (with its `dtype`
entries) or an `anchor` or `half` record whose `stage` is the stage of one of its definitions and whose `compute` holds
exactly the weights, compute, embedding scale and attention rows of that definition and nothing else, with the three
scales computed from `hidden_size`. The comparison passes only when both records pass
these checks and passed, differ in SHA-256 and are not the same file (a hard or symbolic link included), every token
and logits file has the size and SHA-256 that its own record lists and is not the same file as its counterpart, the
token counts, saved positions, first scoring row and scoring positions equal that record, every array is float32 with
the shape `[saved positions, 262144]`, the forced tokens of both sides are the same, an `anchor` or `half` record
names in `source` exactly the SHA-256 of the BF16 reference when that is the other side (two such records name the
same SHA-256) and no logit is NaN or infinite.
Otherwise `problems` lists the findings: a case with a finding has no metrics, a finding about the records stops before
the cases, and no group is reported. The output also names the definition of each side (`identities`). The result is
written to standard output as JSON and nothing else is written there; the exit code is 0 only when it passed. For every
row, with p and q the softmax (float64) of the baseline and the candidate,
it computes the largest |p − q| over the vocabulary, KL(p ‖ q), the largest absolute logit difference and whether the
two argmax IDs (the smaller ID on a tie) agree; a mismatch is a near tie when both sides have the same top-two IDs and
the baseline probability of the baseline winner exceeds that of the candidate winner by at most 0.01. Per case and per
group it reports the maximum and the mean of the largest |p − q|, the mean, 99th percentile and maximum of KL, the
largest logit difference, the top-1 matches, the near-tie mismatches and the other mismatches, over every row and over
the scoring rows, and it lists every mismatch. The groups are the case kind (`chat` or `raw`) by input-length band
(1–512, 513–2,048, 2,049–4,096, 4,097–16,896 tokens; only the bands with cases), and the rows 992 to 1,056 of every
case with more than 1,056 positions (the sliding window of 1,024 drops the BOS from position 1,024).

`anchor_summary` prints only states, numbers, hashes and fixed words. It stops unless the six steps were executed by
the given run (`status` executed and `origin` the run ID); every record file of the report (the BF16 reference
included) is the output that the runner recorded for its step, that is, the `manifest.json` beside the step's
`artifacts` folder has version 1, status `completed`, the step's `origin` as its run ID, the step ID, and the size
and SHA-256 of that file (read once for this check and for the values) in its output inventory; every record passes
the checks of the comparison and matches the definition of its step in `recipe.json` (for `anchor` the `--weights`
and `--embed-scale` of its command); the `layers` record has the contract model and the fixture case IDs in order;
every record of the six steps names, as `source`, exactly the SHA-256 of the report's `reference.json`; no two steps
hold the same record, no two comparisons compare the same pair of records (in either order) and every comparison
lists the SHA-256 of two records of that report. It also stops when the report, a manifest, a record or a comparison does not have the shape of the entries
that it reads:

~~~sh
python3 -m Tools.Conversion.adapters.gemma4_31b.anchor_summary --report <report.json> \
  --comparison <comparison.json> [--comparison <comparison.json> ...]
~~~

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
  those of the n new ones) and returns `logits` [1, n, vocabulary]; `prefill` takes the same inputs and returns
  nothing.
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

### FP32 logits of the first layers

`adapters/gemma4_31b/layers.py` writes the FP32 logits of the first N layers of either implementation for two fixed
inputs, `short` and `long`, and compares two such runs:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers run \
  --implementation original --layers 6 --model Tools/Conversion/recipes/gemma4_31b/model.json \
  --snapshot .local/conversion/models/google--gemma-4-31B-it --output <new folder>/original
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers \
  run --implementation class --layers 6 --model Tools/Conversion/recipes/gemma4_31b/model.json \
  --snapshot .local/conversion/models/google--gemma-4-31B-it --output <new folder>/class
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers compare \
  --original <new folder>/original --model-class <new folder>/class > <new folder>/comparison.json
~~~

`run` loads the weights: start it under the shared machine lock, which it does not take itself. It accepts only a
layer count whose last layer is a full-attention layer (6, 12, …, 60 here) and refuses any other before it creates
the folder or reads a weight. It writes `short.npy`, `long.npy` and, last, `record.json` into a new folder.
`compare` passes only for a run of the original implementation and a run of the model class on the same inputs
whose differences stay below the criteria: for `short` a largest absolute logit difference below 1e-3, and for
`long` a largest absolute logit difference below 5e-3, a largest probability difference below 1e-6 and no position
with another `argmax`.

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

## Terms

Gemma 4 31B is distributed under Apache-2.0. The weights stay outside Git.
