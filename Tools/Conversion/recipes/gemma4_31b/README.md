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
  reads both swap and the free percentage (at most 60 seconds), then hashes the files it loads (both weight
  shards, `config.json`, `generation_config.json`, `model.safetensors.index.json` and the three tokenizer
  files) and stops before loading unless every size and SHA-256 equals the contract; `reference.json` records
  these hashes, not the copies in `ready.json`. It then loads `Gemma4ForConditionalGeneration`
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

A thread samples swap, the memory free percentage and the free disk every 5 seconds (`monitor.tsv`,
from `adapters/jev27b/monitor.py`). The stage stops as failed with `failure.json` and exit code 1 when:

- swap grows by more than 2,000,000,000 B from its first sample;
- the free percentage stays below 8 for three samples in a row;
- swap is above 16,000,000,000 B and rises on three samples in a row (the condition of the shared monitor);
- a sample cannot read swap or the free percentage, or the monitor thread ends with an error.

A process still running 120 seconds after a stop is killed, also when the monitor thread itself ended
with an error (a separate timer takes over the kill). A run that finishes after a stop, with an
unreadable sample or a monitor error is not `passed`, and neither is a run without its peak resident
memory, kernel lifetime footprint peak, MPS allocation or swap increase. These are recorded after loading,
after every case and at the end. The stage needs 20 GiB of free disk; the saved logits of the twelve
cases take about 3.4 GB per run.

## Second run and shared numbers

`--refresh` runs both steps again in new processes. Compare the two reference artifacts (the `artifacts`
folders that the two reports list for `reference`) with the reference environment:

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.compare \
  --first <first artifacts> --second <second artifacts> --output .local/conversion/work/gemma4_31b/rerun.json
python3 -m Tools.Conversion.adapters.gemma4_31b.summary --report <first report.json> --reference <first reference.json> \
  --report <second report.json> --reference <second reference.json> --comparison .local/conversion/work/gemma4_31b/rerun.json
~~~

`compare` records per case whether the input and output tokens match, the first position where the forced
sequences differ, and the largest absolute logit difference over the saved rows whose context is still
identical (every row when the sequences match). It passes only when both `reference.json` files passed, the
model, environment, fixtures and cases are the same, every token file and saved array has the size and SHA-256
that its own `reference.json` records, the input and output token counts, the saved positions and the first
scoring row equal that record, every saved array is float32 with the shape `[saved positions, vocabulary]`,
no compared logit is NaN or infinite, and the input and output tokens of every case match; any other finding
is listed in `problems`. Non-finite logits are counted per run and leave the largest difference of the case
and of the comparison unset. `summary` prints only states, numbers, hashes, versions
and case IDs; it lives beside the steps because every `Tools/Conversion/*.py` file is part of each step's
cache key. The published record is [docs/validation/gemma4-31b](../../../../docs/validation/gemma4-31b/README.md).

## Terms

Gemma 4 31B is distributed under Apache-2.0. The weights stay outside Git.
