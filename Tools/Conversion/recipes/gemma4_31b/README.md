# Gemma 4 31B BF16 reference

This recipe pins [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475` and records the outputs of the original Transformers
implementation with the distributed BF16 weights, BF16 compute and MPS for twelve self-written diagnostic
inputs. Later steps (an FP32 reference, the Core AI comparison and quality checks) read these outputs. It
does not convert the model, and the diagnostic inputs are not a quality evaluation set.

- `model.json`: the contract. It pins the size and SHA-256 of all twelve files of the revision, the main
  text configuration values (60 layers as five sliding-window layers followed by one full-attention layer,
  hidden size 5376, MLP width 21504, vocabulary 262144, window 1024, 16 and 4 key/value heads, head
  dimensions 256 and 512, shared keys and values in the full-attention layers, final logit soft cap 30,
  tied embeddings, BOS 2 and stop tokens 1, 106 and 50), the distributed tokenizer files, the reference
  precision and the upstream converter revision `c8e00c01800ba6ad324707007f7af358efc980a4` (recorded only;
  no converter checkout or environment is created). A snapshot that differs fails; the contract is not
  adjusted to the files.
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
`.local/conversion/gemma4_31b/.venv` from this folder's lock, `.local/conversion/gemma4_31b/setup.json` and
`.local/conversion/gemma4_31b.paths.json`. `--snapshot` selects another folder. The run uses
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and a Hugging Face home under `.local/conversion/gemma4_31b/hf-cache`.

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

## Quality evaluation baseline

Two modules write the outputs of the `reference` role of the quality contract `G31-QC-v1` (#669). They are not steps
of `recipe.json`: a step runs to its end in one execution and each execution writes a new object, while the primary
set takes hours and runs in slices that resume one run folder, releasing the shared machine lock between slices. Run
them from the repository root with the reference environment.

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

`inputs` reads no weights. `--items` is the `primary.jsonl` of `python3 -m Tools.LLMEvaluation build`. It stops before
it writes unless `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` have the size and SHA-256 of
`model.json`, the tokenizer that `AutoTokenizer` builds equals the distributed `tokenizer.json`, the items are those of
`--set` in its order, each with its `item_sha256`, and every item, rendered by `apply_chat_template` with
`add_generation_prompt=True` and `enable_thinking=False`, equals the frame of `contract.json` around its content. Its
IDs must tokenize from that text with `add_special_tokens=False`, hold one BOS, start with the first IDs of the frame
and hold exactly the special tokens of the frame (the added tokens of `tokenizer.json` marked special), and an item
with a `token_count` in the set must have that many IDs. The new output folder gets `inputs.jsonl`, one record per item
in item order with `id`, `input_ids`, `input_sha256` (the SHA-256 of the IDs written in decimal and joined by commas,
as UTF-8) and `max_new_tokens`, as JSON with sorted keys and no spaces, so the same items give the same bytes; and
`inputs.json` with the `version` and SHA-256 of `contract.json` and the `count` and `sha256` of `inputs.jsonl`.

`baseline` loads the model, so take the heavy-job lock of `AGENTS.md` first. Before it loads the model it requires
`ready.json` of `prepare` for this `model.json`, the package versions of `pyproject.toml` and MPS (the check of
`prepare`), and input records each with a new string `id`, `input_ids` of integers from 0 to below the vocabulary size,
an `input_sha256` equal to that of the IDs and a positive integer `max_new_tokens`. It parses and hashes the same bytes
of `contract.json` and `inputs.jsonl`. It then opens the run folder as described below and, within the supervised
section of `reference` (the monitor, its first sample and its stops, with `monitor.tsv` and `failure.json` in the
`--log` folder), hashes the files it loads with the check of `reference`, loads the model as `reference` does, stops
unless the loading is clean, and generates.

For each item, an input of more than 1,024 tokens is read into a `DynamicCache` in chunks of 1,024, the last chunk
holding the remaining 1 to 1,024 tokens; a shorter input is read in one call. Each call keeps the logits of the last
position only. After each chunk and after the item, MPS is synchronized, its driver allocation recorded and its cache
emptied. Each step moves the logits (after the soft cap, over the whole vocabulary) to float32 on the CPU. A step with
a non-finite logit ends the item with `error`, the count in `non_finite` and `stop` set to `limit`. Otherwise `top2`
gets the two IDs with the largest logits, a tie going to the lower ID, with their probabilities from the temperature-1
softmax over the whole vocabulary in double precision; the first is the generated token. A stop token of
`contract.json` ends the item with `stop_token`; any other token is appended to `output_ids`, the item ends with
`limit` at `max_new_tokens`, and otherwise the token is read for the next step. An error raised while generating ends
the item with `error`. After an item with `error` is written, the command stops with an error, because a baseline
record with an error makes the comparison `incomplete`. Items run in the order of the input records, except that those
of `long-ja-4k`, `long-ja-16k`, `synth-long-4k` and `synth-long-16k` run last.

The run folder holds three files:

| File | Contents |
|---|---|
| `outputs.jsonl` | The output records of contract §5.1, one per item, as JSON with sorted keys, appended, flushed and synced as each item ends |
| `progress.json` | `lines`, the number of confirmed lines; `sha256`, the SHA-256 of their bytes with their newlines; `next_sha256`, the SHA-256 after the line about to be written, only while it is written; `run`, the SHA-256 of the run's `id` and conditions as JSON with sorted keys. Before each line it is replaced through a temporary file in the folder to announce the line; the next announcement or a session end that writes `finished` confirms it |
| `run.json` | `id` (a UUID made at the first start), `contract` (`version`, `sha256`), `set` (`primary`), `role` (`reference`), `asset` (`model`, `revision` and the `weights` SHA-256 of `model.json`, which the check before loading requires), `inputs` (`sha256`, `count`), `implementation` (`tool`, `commit`, and `environment`: Python, the package versions and the OS build), `call_shape` (`prefill_chunk` 1024), `order` (`last`, the categories that run last), `items`, `started` and `finished` |

`run.json` is written when a session starts, with `items` set to the finished items and without `finished`, and when
it ends normally, with `items` and `finished`. With `--max-minutes`, once that many minutes have passed since the
command started, the session ends normally after the item in progress, so at least one item runs when items remain.
The last line of standard output (`record` `session`) gives `items`, `count` and `complete`; the line of each item
(`record` `item`) has its token counts, stop, times and memory. The `--log` folder, which must be new, then gets
`session.json`: the loading, the times and memory of each item, and the resources and `passed` of `reference`'s record
(a clean monitor and recorded resources). Without `passed`, the session exits with 1 and does not write `finished`.

A folder without `run.json` starts a new run only when it is empty, or holds nothing but an empty `outputs.jsonl` and
possibly a `progress.json` of no lines; it then writes `outputs.jsonl`, `progress.json` and `run.json` in that order.
With a `run.json`, the command stops before it loads the model and removes nothing unless `run.json` has a run `id`,
every field except `id`, `items`, `started` and `finished` equals this run's, `progress.json` exists and its `run`
equals the digest of `run.json`, `items` is an integer from 0 to the number of input records, `lines` is an integer from
0 to the number of complete lines of `outputs.jsonl`, there are at least `items` complete lines, the complete lines are
either `lines` lines with the SHA-256 `sha256` or one more with `next_sha256`, and each complete line has an `id` of the
input records, once. Only then are the bytes after the last newline, a line cut while written, removed; a complete line
is never removed, and the finished items do not run again. The meaning of the lines is not checked again. These checks
detect a cut last line; `outputs.jsonl` and `progress.json` that disagree, when only one of them is older, newer or
changed; files of a run with other conditions; files of another run with the same conditions; and fewer complete lines
than `items`. They do not detect `outputs.jsonl` and `progress.json` put back together to an earlier point of the same
session, which cannot be told from an interruption at that point, nor deliberate changes to all three files.

The published record of the primary-set run is
[docs/validation/gemma4-31b/baseline-20261122.md](../../../../docs/validation/gemma4-31b/baseline-20261122.md).

## Terms

Gemma 4 31B is distributed under Apache-2.0. The weights stay outside Git.
