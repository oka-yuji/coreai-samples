# Gemma 4 31B BF16 reference

This recipe pins [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475` and records the outputs of the original Transformers
implementation with the distributed BF16 weights, BF16 compute and MPS for twelve self-written diagnostic
inputs. Later steps (an FP32 reference, the Core AI comparison and quality checks) read these outputs. It
does not convert the model, and the diagnostic inputs are not a quality evaluation set. It also defines the
converter environment and a model class for a macOS export with dynamic shapes, with a tool that compares the
first layers of that class with the original implementation in FP32 ([Converter environment and model
class](#converter-environment-and-model-class)). Nothing is exported to Core AI yet.

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

`local setup --recipe gemma4_31b` also prepares `.local/conversion/upstream-gemma4-31b/coreai-models`. When the
folder is missing, it clones Apple's public converter and checks out `upstream_revision` of `model.json`. An
existing checkout is used only when its origin is the public repository, its HEAD is that revision and no tracked
file is edited; otherwise setup stops without changing it. Setup then installs the frozen lock of the checkout into
`<checkout>/.venv` (`uv sync --frozen --no-dev --package coreai-models --python 3.11`) and checks the checkout
again. It builds no Swift runner. The lock of this revision installs coreai-core 1.0.0b3, coreai-torch 0.4.3,
coreai-opt 0.4.0, torch 2.9.0 and transformers 5.12.1. The reference environment above stays separate and keeps
transformers 5.16.1.

### Model class

`adapters/gemma4_31b/model.py` defines `Gemma4ForCausalLM`, a subclass of the converter's `BaseForCausalLM` for
the text model. It imports the converter, so it runs only in the converter environment.

- `main` takes `input_ids` [1, n] and `position_ids` [1, P + n] (the positions of the P cached tokens followed by
  those of the n new ones) and returns `logits` [1, n, vocabulary] after the soft cap `tanh(x / 30) · 30`.
  `prefill` (prefill mode) takes the same inputs and returns nothing.
- The states are four growing caches whose only dynamic dimension is the sequence (index 3), written by the
  converter's `KVCache.update_and_fetch`: `keyCache` and `valueCache` [full-attention layers, 1, 4, ?, 512] and
  `slidingKeyCache` and `slidingValueCache` [sliding-window layers, 1, 16, ?, 256] (the `forward` arguments
  `k_cache`, `v_cache`, `sliding_k_cache` and `sliding_v_cache`). The first N layers follow the pattern: 1 and 5
  layers for 6, 5 and 25 for 30, 10 and 50 for 60.
- Sliding-window layers cut the prefix that `update_and_fetch` returns to its last `L = sym_min(seq, window − 1 + n)`
  positions and mask it with the boolean `0 ≤ q − k < window`, where `q` and `k` are values of `position_ids` (the
  last n and the last L). They call the composite SDPA with `is_causal=False` and scale 1.0. Full-attention layers
  call it with `is_causal=True` over the whole prefix. The class builds no position or mask from an `arange` of a
  dynamic length (both come from `position_ids`), and it does not use the composite `window_size`.
- The rest of the graph: the embedding times the BF16 rounding of √hidden (73.5 for 5376), RMSNorm with the weight
  `w`, `q_norm` and `k_norm` before RoPE, `v_norm` without a weight (in full-attention layers on the K projection
  before `k_norm`), the proportional RoPE through `DecomposedRoPE` in full-attention layers and the composite RoPE
  in sliding-window layers, `layer_scalar` on the whole stream after the second residual, the final norm and the
  head shared with the embedding.
- `num_layers` keeps the first N layers: `from_hf_memory_efficient(snapshot, num_layers=N,
  hf_config_attr="text_config", hf_state_dict_prefix="model.language_model.")` reads them layer by layer without
  changes to the converter.
- The configuration is read from the flat fields of transformers 5.12.1 (`head_dim`, `global_head_dim`,
  `num_key_value_heads` and `num_global_key_value_heads`). A missing field stops; there is no fallback
  (transformers 5.16.1 keeps the head dimensions per layer).
- These configurations stop with `ValueError` in `_get_reauthored_config`, which the converter calls before it
  reads any weight: per-layer inputs, shared key and value layers, a mixture of experts, `attention_k_eq_v` false,
  attention bias, `use_bidirectional_attention` `"all"`, untied embeddings, another activation than
  `gelu_pytorch_tanh`, another RoPE type in either kind of layer, a scaled full-attention RoPE and another layer
  type.

### FP32 logits of the first layers

`adapters/gemma4_31b/layers.py` writes the logits of the first N layers for two fixed inputs with either
implementation, and compares two such runs.

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers run \
  --implementation original --layers 6 --model Tools/Conversion/recipes/gemma4_31b/model.json \
  --snapshot .local/conversion/models/google--gemma-4-31B-it --output <new folder>/original
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers \
  run --implementation class --layers 6 --model Tools/Conversion/recipes/gemma4_31b/model.json \
  --snapshot .local/conversion/models/google--gemma-4-31B-it --output <new folder>/class
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers compare \
  --original <new folder>/original --model-class <new folder>/class --output <new folder>/comparison.json
~~~

- `run` loads model weights: start it under the shared machine lock, which it does not take itself (six layers
  hold 4,338,096,902 values, 17.4 GB in FP32). It refuses an existing output folder. Right before loading it checks
  the snapshot with `verify_inputs` of the reference step (the size and SHA-256 of every file to load, and exactly
  the pinned files). Both implementations round every checkpoint tensor to FP16 and compute in FP32 on the CPU,
  with the BF16 rounding of √hidden as the embedding multiplier.
  - `original` (reference environment) is the Transformers `Gemma4ForCausalLM` with eager attention, built for the
    first N layers of `config.json`. It loads exactly those layers, the embedding and the final norm under
    `model.language_model.` and stops when a key is missing or left over. Its multiplier buffer is set to the BF16
    rounding that the distributed BF16 model computes with (73.5, where FP32 would use √5376 = 73.32).
  - `class` (converter environment) is the model class read with `from_hf_memory_efficient(..., target_dtype=float16,
    num_layers=N)` and converted to FP32. It first requires that `coreai_models` is imported from a checkout that
    passes the setup check for `upstream_revision`.
- Inputs: `short` has 26 tokens and `long` 1,400 (more than the window of 1,024): BOS followed by IDs from BOS + 1
  to the vocabulary size, drawn by Python's `random.Random` with the seeds 0 and 1. `original` reads each input in
  one call without a cache. `class` reads the first `sliding_window` tokens in one call and the rest in chunks of
  1, 2, 4 and 8 tokens in turn, with states sized to the input, so the calls after position 1,024 cut the window.
- Output: `short.npy` and `long.npy` (float32 [positions, vocabulary], the logits after the soft cap; row j is the
  output at position j) and, written last, `record.json`: the implementation, the layer count and types, the model
  ID, revision and contract SHA-256 with the hashes verified before loading, the weight handling, the multiplier,
  the package versions (and the converter revision for `class`), per input the positions, the SHA-256 of the token
  list (canonical JSON), the calls and the size, SHA-256, dtype, shape and non-finite count of the logits file, the
  times, the peak resident memory and the kernel lifetime footprint peak. A run that stops writes no `record.json`.
- `compare` refuses an existing output file. It passes only when the first run is `original` and the second
  `class` (a run compared with itself fails), both used this contract with the same layers, each read the fixed
  input (positions and token SHA-256), every logits file has the size and SHA-256 of its record and is float32
  [positions, vocabulary], no compared value is NaN or infinite, and the criteria hold. The criteria were fixed from
  the measurement of #665 before this check ran: for `short` the largest absolute logit difference is below 1e-3;
  for `long` it is below 5e-3, the largest absolute difference of the softmax probabilities is below 1e-6 and no
  position has another `argmax`. The differences are computed in float64 over blocks of 256 rows.

### Tests

The tests of the model class import the converter and live in `Tools/Conversion/tests/export_gemma4_31b_model.py`.
The default pattern of `unittest discover` (`test*.py`) does not collect them, so other environments neither run
nor report them. Run them in the converter environment, and the remaining tests of this recipe in both
environments:

~~~sh
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m unittest discover -s Tools/Conversion/tests \
  -p 'export_*.py' -v
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m unittest discover -s Tools/Conversion/tests \
  -p 'test_gemma4_31b_layers.py' -v
.local/conversion/gemma4_31b/.venv/bin/python -m unittest discover -s Tools/Conversion/tests \
  -p 'test_gemma4_31b_*.py' -v
~~~

They build a synthetic Gemma 4 text model (five sliding-window layers and one full-attention layer, twice; window
8; hidden size 48; `attention_k_eq_v`; soft cap 30; random weights) and compare the model class with the original
implementation of the converter environment (Transformers 5.12.1, eager, FP32, CPU) on 61 tokens read in one call,
one token at a time, in chunks of 2, 4 and 8, and as a prefill of 20 tokens followed by one token at a time; every
reading stays below a largest logit difference of 1e-3. The same comparison detects a window one position wider
or narrower, unmasked later positions in the window or in the full-attention layers, a window slice one position
shorter and the positions of the new tokens shifted by one. They also check the state shapes for 1 to 12 layers,
that each state keeps one dynamic dimension in the graph of `torch.export` (no Core AI conversion), the unsupported
configurations, and `run` of both implementations on a synthetic snapshot followed by `compare`, which reports no
problem and logit differences below 1e-3 (the test does not apply the probability criterion of `long`, which is
meant for the vocabulary of 262,144 tokens). `test_gemma4_31b_layers.py` covers the original implementation and
`compare` without the converter.

## Terms

Gemma 4 31B is distributed under Apache-2.0. The weights stay outside Git.
