# JEV-27B typed decisions

This recipe pins the inputs, environments and evaluation sets for converting the System 1
path of [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) (revision
`14b425269ed24452e0a5ad5a8f3e51d139647a88`): the Qwen3.8-27B text backbone with its
unmerged LoRA and the 24-slot fp32 decision head. The recipe merges the LoRA and records
PyTorch references; it does not convert or evaluate the model. These files record what the
steps rely on:

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

`--snapshot` selects another JEV-27B folder. The first two steps check the inputs and sets:

- `prepare` writes `ready.json` without re-hashing the weight shards. It requires each folder to
  hold exactly the files of its integrity record (apart from `.cache`), compares sizes and
  modification times with the record, and re-hashes files up to 32 MiB. It stops when
  `pyproject.toml`, `uv.lock`, the converter revision or `llm-runner` differ from those recorded
  by setup; run setup again after changing them.
- `fixture` regenerates every set, requires byte-identical files and restores each row from the
  corpus and JGLUE files through `load_rows` with matching hashes.

The steps that read the weights are heavy, so `run` holds the shared machine lock for the whole
pipeline. Each executed `merge` adds a 54 GB checkpoint under the work directory's `objects/`;
a reused step does not, and `--refresh` always adds another.

## Merge and references

- `merge` adds each LoRA pair to its projection as `bf16(fp32(W) + 2.0 * (fp32(B) @ fp32(A)))`,
  one tensor at a time on the CPU, and copies every other tensor bit for bit. It stops unless all
  400 pairs are used exactly once and the shards add up to the source `total_size`.
  `checkpoint/` holds the 13 shards with the source file names and membership, the keys renamed
  from `model.` to `model.language_model.` (`lm_head.weight` unchanged), a recomputed index, the
  `Qwen/Qwen3.8-27B` `config.json` copied byte for byte (`qwen3_5` with `text_config`; its
  `vision_config` is kept, but no vision or MTP tensors exist) and the three distributed tokenizer
  files copied byte for byte. `manifest.json` lists the path, size and SHA-256 of every checkpoint
  file. `merged_revision` is the first 40 hexadecimal digits of the SHA-256 of that list serialized
  with sorted keys and no spaces, so it depends only on the files. `provenance.json` records the
  input hashes, the formula, the scale, the dtypes, the pairs per target, the mapping of layers 0
  and 3, the elapsed time, the peak resident memory and the library versions.
- `reference` first checks that PEFT holds every adapter pair of the file with the scale 2.0 in eval
  mode. It renders and tokenizes every row of F, rejects rows longer than 8,192 tokens without a
  forward pass, and computes the head logits and the probabilities at temperature 1 and at the
  calibrated temperature with the adapter (stage 1) and inside `disable_adapter()` (stage 0), where
  every LoRA layer must report itself disabled.
- `reference_merged` loads the checkpoint as `Qwen3_5ForCausalLM` with
  `config=Qwen3_5Config.from_pretrained(checkpoint).text_config`. Transformers 5.16.1 removes the
  `language_model` prefix for `qwen3_5_text`, so no `key_mapping` is given; the applied conversion
  rules are recorded, and missing, unexpected and mismatched keys must be empty before any row runs.
  It runs the same rows (stage 1') and also writes `hidden.safetensors`: `hidden` [302, 5120]
  float32 in the order of F, zero for rejected rows, and `mask` [302] uint8 that is 1 for rows with
  a forward pass. Its metadata holds `merged_revision` and the row IDs (`ids`).
- `verify_merge` recomputes every merged projection from the source tensors and the adapter, and
  compares all 851 tensors bit for bit, their dtypes, the index, the shard headers, `config.json`,
  the tokenizer files and the manifest. It requires both references to have passed and their rows
  to come in the same order with the same ID, kind, options, band, hashes, token count, tokens and
  rejection, and passes only when the largest probability difference at temperature 1 is at most
  0.01 on every compared row. `comparison.json` (`typed-decision-merge/1`)
  holds the checks and the statistics by kind and band; `reference.json` joins stages 0, 1 and 1';
  `report.md` compares the card examples with the card values, which is not a pass criterion.
  `python3 -m Tools.Conversion.summary <report.json> --comparison <comparison.json>` prints the
  shareable numbers.

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
