# Gemma 4 31B BF16 reference

This recipe pins [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475` and records the outputs of the original Transformers
implementation with the distributed BF16 weights, BF16 compute and MPS for twelve self-written diagnostic
inputs. Later steps (an FP32 reference, the Core AI comparison and quality checks) read these outputs. It
does not convert the model, and the diagnostic inputs are not a quality evaluation set. It also defines the
converter environment and a model class for a macOS export with dynamic shapes, with a tool that compares the
first layers of that class with the original implementation in FP32 ([Converter environment and model
class](#converter-environment-and-model-class)). A second recipe, `recipe-fp16.json`, exports the first layers to a
dynamic-shape FP16 Core AI bundle and compares the logits of the own engine with an FP32 reference of the same layers
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
recorded after loading, after every case and at the end. The stage needs 20 GiB of free disk; the saved logits of
the twelve cases take about 3.4 GB per run.

## Second run and shared numbers

`--refresh` runs both steps again in new processes. Compare the two reference artifacts (the `artifacts`
folders that the two reports list for `reference`) with the reference environment:

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

## Converter environment and model class

### Converter environment

`local setup --recipe gemma4_31b` also prepares `.local/conversion/upstream-gemma4-31b/coreai-models`: a checkout of
Apple's public converter at `upstream_revision` of `model.json`, with the frozen lock of that checkout installed in
its `.venv`, and writes the checkout as `upstream` in the bindings. An existing checkout is used only when it is a
clean checkout of that revision; otherwise setup stops without changing it. This converter environment is separate
from the reference environment of the steps above.

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

The tests of the model class are in `Tools/Conversion/tests/export_gemma4_31b_model.py`, and
`export_gemma4_31b_export.py` runs the `export` step of `recipe-fp16.json` on a synthetic model. The default pattern of
`unittest discover` (`test*.py`) does not collect them; run them in the converter environment, where
`test_gemma4_31b_layers.py` and `test_gemma4_31b_native.py` also run, and run every `test_gemma4_31b_*.py` in the
reference environment:

~~~sh
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m unittest discover -s Tools/Conversion/tests \
  -p 'export_*.py' -v
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m unittest discover -s Tools/Conversion/tests \
  -p 'test_gemma4_31b_layers.py' -v
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m unittest discover -s Tools/Conversion/tests \
  -p 'test_gemma4_31b_native.py' -v
.local/conversion/gemma4_31b/.venv/bin/python -m unittest discover -s Tools/Conversion/tests \
  -p 'test_gemma4_31b_*.py' -v
~~~

## FP16 export of the first layers

`recipe-fp16.json` exports the first layers of the pinned snapshot to a Core AI bundle for macOS with dynamic shapes
and FP16 weights, without compression, and inspects the asset. It runs the `prepare` step of `recipe.json` and a heavy
`export` step in the converter environment. The step needs `upstream`, the converter checkout, in the bindings (setup
writes it) and two values without defaults: `layers`, a count whose last layer is a full-attention layer (6, 12, …,
60), and `functions`, `main` or `main,prefill`. A run without them stops before any step. `context`, the context limit
of the asset, defaults to 16,896. `local plan|run` does not select this recipe:

~~~sh
python3 -m Tools.Conversion run Tools/Conversion/recipes/gemma4_31b/recipe-fp16.json \
  --bindings .local/conversion/gemma4_31b.paths.json --set layers=6 --set functions=main,prefill \
  --work-dir .local/conversion/work/gemma4_31b-fp16 --lock-dir ~/Library/Caches/coreai-samples/heavy-job
~~~

The `export` step (`adapters/gemma4_31b/export.py`) writes into a new folder `export/` of its stage:

- It refuses a layer count that does not end with a full-attention layer before it makes the folder, and an existing
  folder, file or link before it reads the snapshot.
- Before it starts, the volume of the folder must have twice the FP16 weights of the chosen layers (the asset and the
  loading work area, counted from the model class without reading weights) plus 60 GiB free.
- Under the monitor of the `reference` step (a swap increase above 2,000,000,000 B stops it with `failure.json`), it
  hashes the files to load against the contract, checks the converter checkout, loads the model class with
  `from_hf_memory_efficient` in FP16, exports with `export_macos_model` without a compression setting, and saves
  `bundle.incomplete/` with `metadata.json`, `<name>.aimodel` and `tokenizer/`, the three distributed tokenizer files.
- It inspects the asset and writes `export.json`. Only a run that passes renames the bundle to `export/bundle`.

The inspection reads the asset with `coreai.runtime._AIModelAsset(...).summary(include_statistics=True)`, a
non-public interface of the converter package. The step fails when:

- the export was given a compression setting, or the PyTorch model has a quantized weight;
- the asset stores a type other than Float16 and the constant types Float32, Int32, UInt32 and UInt64, or Float16
  holds fewer elements than the parameters and buffers of the model. Every weight has to be Float16; what the other
  four types hold are constants of the graph (in #665, 386, 148, 29 and 1 elements for six layers);
- the asset has other functions than those asked for, or an input, output or state other than: `input_ids` and
  `position_ids` int32 [1, ?]; `logits` float16 [1, ?, 262144] from `main` only; `keyCache` and `valueCache` float16
  [full-attention layers, 1, 4, ?, 512] and `slidingKeyCache` and `slidingValueCache` float16 [sliding-window layers,
  1, 16, ?, 256], each with the sequence as its only dynamic dimension;
- a tokenizer file differs from the SHA-256 of the contract, or `metadata.json` differs from the one for the contract
  and the arguments (vocabulary 262,144, the context limit, and `source.hf_model_id` and `source.revision` of the
  contract);
- the free space of the volume fell below 60 GiB at any monitor sample.

`export.json` records the layer count and layer types, the functions, the times, the peak resident memory, the kernel
lifetime footprint peak, the swap increase, the free space before, required and lowest, the storage types, the
functions of the asset, `main.hash` and the size and SHA-256 of every bundle file. `metadata.json` does not mark the
layer count. The context limit has to be larger than the 2,048 positions of the cache that the exporter traces with;
otherwise the states come out static and the inspection fails.

### FP32 reference of the comparison

`layers.py context` computes, with the original implementation and the weights of `run` (rounded to FP16, FP32 on the
CPU), the logits of `short` and of `context`: 16,896 tokens made like the other inputs (seed 2), read with the cache
of the original implementation in calls of 1,024 tokens. It keeps only the rows that the conditions below compare
(26 of `short`, 1,428 of `context`) and refuses a layer count and an existing output like `run`. The tests check on a
synthetic model that reading in calls with the cache gives the logits of one call within 1e-3. Like `run`, it loads
the weights without taking the shared lock:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers context --layers 6 \
  --model Tools/Conversion/recipes/gemma4_31b/model.json --snapshot .local/conversion/models/google--gemma-4-31B-it \
  --output <new folder>
~~~

### Comparison with the own engine

Build `coreai-verify` with `swift build -c release` in `Tools/Verify`. `native.py` runs it on the bundle, so take the
shared lock first:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.native \
  --export <artifacts of the export step>/export --reference <context folder> \
  --verifier Tools/Verify/.build/release/coreai-verify --output <new folder>
~~~

Before it runs `coreai-verify`, it requires a passed `export.json` whose bundle holds exactly its recorded files with
their sizes and SHA-256, a reference of `context` with the rows of the conditions and logits of their recorded size
and SHA-256, layer counts and layer types of both that match the contract, the model and contract of `model.json`, and
the same layer count in both. For each condition it then writes the token file, runs `coreai-verify logits` (stopped
after 1,800 s) and requires exit code 0, one JSON record that matches the condition (token count and SHA-256, start,
chunk, shape, the number of `main` calls and the `main_hash` of the export) and a file of rows × 262,144 float16
values:

| Condition | Tokens | `--start` | `--chunk` | Compared rows |
|---|---|---:|---|---|
| `short-c1`, `short-c26` | `short` (26) | 0 | 1, 26 | 0 to 25 |
| `window-c1`, `window-c2`, `window-c4`, `window-c8` | first 1,300 of `context` | 0 | 1, 2, 4, 8 | 0 to 1,299 |
| `prefix-c1` | first 1,300 of `context` | 1,000 | 1 | 1,000 to 1,299 |
| `capacity-2048`, `capacity-4096`, `capacity-8192`, `capacity-16384` | first 2,056, 4,104, 8,200, 16,392 of `context` | 2,040, 4,088, 8,184, 16,376 | 4, 8, 2, 4 | the last 16 |
| `tail` | `context` (16,896) | 16,832 | 1 | 16,832 to 16,895 |

A condition passes with no non-finite value, no constant row (a row whose largest and smallest values are equal), a
largest absolute logit difference from the reference below 2.0 and no top-1 change outside a near tie, where the top
two reference logits differ by 0.01 or less. The rows of a call from `--start` of 8 or more tokens after which the
positions total 97 to 112 (the call shape that `Engine.callShapeRules` avoids for this model; rows 96 to 111 of
`window-c8`) are left out of the difference and the top-1 judgment and recorded apart; non-finite values and constant
rows count on every row. The first six layers saturate the logits, so the probability thresholds of the quality
contract are not used. `comparison.json` holds the results of every condition. The specialization cache that
`coreai-verify` makes stays in place; remove it yourself. `fp16_summary.py` prints only the shareable states, numbers,
hashes and fixed names of the reference, the exports with their pipeline reports and the comparisons; the published
record is
[docs/validation/gemma4-31b](../../../../docs/validation/gemma4-31b/README.md#fp16-export-of-the-first-six-layers).

## Terms

Gemma 4 31B is distributed under Apache-2.0. The weights stay outside Git.
