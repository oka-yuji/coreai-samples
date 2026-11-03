# JEV-27B typed decisions

This recipe pins the inputs, environments and evaluation sets for converting the System 1
path of [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) (revision
`14b425269ed24452e0a5ad5a8f3e51d139647a88`): the Qwen3.8-27B text backbone with its
unmerged LoRA and the 24-slot fp32 decision head. This stage does not merge, convert or
evaluate the model. It records what later stages may rely on:

- `model.json`: the contract checked against the downloaded adapter, head, judge and
  calibration files. A mismatch fails the check; the contract is not adjusted to the files.
- `ready.json` from the `prepare` step: integrity, tokenizer comparison, adapter and head
  layout, and the versions of both environments and the Apple toolchain.
- `fixtures/`: identifiers and hashes of the fixed sets F, Q, Cal, H and J
  ([fixtures/README.md](fixtures/README.md)). No corpus or JGLUE text is stored.
- `Tools/Conversion/adapters/jev27b/typed_head.py`: rendering of the `bare-v1` template,
  slot ranges, temperature and softmax, token bands, tokenization and the final hidden state.

## Local inputs

Setup never downloads models or datasets. Place the pinned inputs under the workspace
(default `.local/conversion`), each with a sibling `<folder>.integrity.json` that records the
size and the LFS SHA-256 or git blob SHA-1 of every file:

| Folder | Source |
|---|---|
| `models/autotrust--JEV-27B` | all 65 files at the revision above |
| `models/Qwen--Qwen3.8-27B-meta` | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`: config, tokenizer files, README, LICENSE |
| `datasets/SargeDev--jev-distill-corpus-v3` | `README.md`, `test.jsonl`, `test_set_30k.jsonl` and `calibration.jsonl` at `75111a13a8455032f0435050bfc7e0c9026ecfe8` |
| `datasets/yahoojapan--JGLUE` | the JNLI, JCommonsenseQA and JSTS validation files of JGLUE v1.2.0 (`c96c6af54200a404f31a3a4cd0b289786aa78809`) |

Use `hf download <id> --revision <revision> --local-dir <folder>` with `HF_HOME` and
`HF_HUB_CACHE` pointing at `.local/conversion/jev27b/hf-cache`, so that no shared Hugging Face
cache is written. The JGLUE files are fetched from
`https://raw.githubusercontent.com/yahoojapan/JGLUE/<revision>/<path>` with the paths in
`fixtures/jglue.json`. These are the files that the `shunk031/JGLUE` loading script
(`41cc99f1c01d41b1ab13435ae97b7076c2199f4c`) downloads; the script itself is not executed.

## Setup, plan and run

~~~sh
python3 -m Tools.Conversion.local setup --recipe jev27b
python3 -m Tools.Conversion.local plan --recipe jev27b
python3 -m Tools.Conversion.local run --recipe jev27b
~~~

Setup checks the integrity records and then, under the shared machine lock, creates:

- the reference environment `.local/conversion/jev27b/.venv` from this folder's
  `pyproject.toml` and `uv.lock` (Python 3.11, torch 2.15.0, transformers 5.16.1, peft 0.21.2);
- the converter environment `.local/conversion/upstream-jev27b/coreai-models`: a clean checkout
  of `apple/coreai-models` at `d3382a59ecce6ba13536130aa87367611b6c4676`, its frozen
  `coreai-models` package environment and the release `llm-runner`;
- `.local/conversion/jev27b.paths.json` and `.local/conversion/jev27b/setup.json`.

`--snapshot` selects another JEV-27B folder. The run has two steps:

- `prepare` writes `ready.json` without re-hashing the weight shards. It requires each folder to
  hold exactly the files of its integrity record (apart from `.cache`), compares sizes and
  modification times with the record, and re-hashes files up to 32 MiB. It stops when
  `pyproject.toml`, `uv.lock`, the converter revision or `llm-runner` differ from those recorded
  by setup; run setup again after changing them.
- `fixture` regenerates every set, requires byte-identical files and restores each row from the
  corpus and JGLUE files through `load_rows` with matching hashes.

## Tokenizer

The distributed `tokenizer.json` is used as is and is never saved again. It is the resaved
form: the pre-tokenizer pattern has no `\p{M}`, the ByteLevel pre-tokenizer trims offsets and
the ByteLevel decoder sets all three options. Its vocabulary, merges, added tokens and
normalizer equal those of `Qwen/Qwen3.8-27B`, and `chat_template.jinja` is identical. The
`pretokenize_regex` in its `tokenizer_config.json` still contains `\p{M}` and therefore differs
from its own `tokenizer.json`. `AutoTokenizer` of transformers 5.16.1 loads the file
unchanged; `prepare`, the fixtures and the load check stop if the tokenizer it builds differs
from the file. Token identifiers in Swift are compared against the same file.

## Load check

Loading the model is a heavy job. Hold the shared machine lock and run with the reference
environment and the offline Hugging Face settings of `recipe.json`:

~~~sh
python -m Tools.Conversion.adapters.jev27b.load_check --model Tools/Conversion/recipes/jev27b/model.json \
  --snapshot <JEV-27B folder> --corpus <corpus folder> --jglue <JGLUE folder> \
  --fixtures Tools/Conversion/recipes/jev27b/fixtures --output <new folder>
~~~

It loads the backbone in bfloat16 on MPS, applies the adapter with PEFT in eval mode, compares
every adapter tensor in the model with the file, and runs the two card examples and one row of
each kind through `last_hidden` and the fp32 head. `load-check.json` records the consumed LoRA
pairs, the probabilities with and without the calibrated temperature, the difference from the
card's example values, timings and the peak resident memory.

## Terms

JEV-27B and `Qwen/Qwen3.8-27B` are Apache-2.0. The JEV corpus is Apache-2.0 and asks for
credit in derived work. JGLUE is CC BY-SA 4.0. The fixtures keep identifiers and hashes only.
