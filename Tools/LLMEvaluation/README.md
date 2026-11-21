# Gemma 4 31B quality evaluation

The rules are in the [quality contract](../../docs/specs/gemma4-31b-quality-contract.md) `G31-QC-v1` (issue #666, parent #658). This folder fixes them in machine-readable form before any candidate is measured, and scores output records with every rule except code (issue #668). Running models, code scoring, paired comparison and reports belong to later child issues.

| File | Contents |
|---|---|
| `contract.json` | Roles, pairs, layer 1 limits, generation conditions and the template frame, verdicts, critical failures, scorer kinds, pinned sources and categories |
| `sets/primary.json`, `sets/reserve.json` | Item ids, source keys or generator arguments, `item_sha256` and token counts only |
| `prompts.json` | All prompt wording, the synthetic templates and the `free-ja` instructions |
| `synthetic/` | Synthetic task generators, the independent solvers that read only the rendered text, and the self-authored vocabulary |
| `scoring.py` | The scoring rules of §6.7 except code, the record checks and `score` |
| `tests/` | Unit tests, the scoring specification examples (`examples.json`) and the expected scoring results derived by hand from §6.7 (`scoring_cases.json`) |

## Commands

Run from the repository root.

```sh
python3 -m Tools.LLMEvaluation validate
python3 -m unittest discover -s Tools/LLMEvaluation/tests -v
```

`validate` uses only the standard library and needs neither data nor a tokenizer. It checks:

- the contract, including pins, thresholds, limits and categories;
- both sets, including counts, ids, overlap rules and token ranges;
- every synthetic and `free-ja` item hash;
- the SHA-256 table and version in the contract document.

In isolated mode Python does not put the repository on `sys.path`, so `python3 -I -m Tools.LLMEvaluation` cannot import the package. Run the entry point by path instead: `python3 -I Tools/LLMEvaluation/__main__.py validate`. It appends the repository root to the end of `sys.path`, after the standard library.

Rebuild the evaluation items from the pinned public data into a new folder. This uses only the standard library:

```sh
python3 -m Tools.LLMEvaluation build \
  --datasets .local/conversion/datasets \
  --output .local/conversion/gemma4_31b/evaluation/new-build
```

`build` verifies each source file's size and SHA-256, recomputes every item and refuses to write when any `item_sha256` differs. It writes `primary.jsonl`, `reserve.jsonl` and `manifest.json`, and an existing output folder is rejected. The built items contain public data text. Keep them under `.local/`, which Git ignores.

`select` fixes the sets once and can recheck them. It is the only command that needs `tokenizers` and `jinja2`. It renders every item with the distributed `chat_template.jinja` and counts tokens with `tokenizer.json`; both files are pinned by SHA-256 in the contract.

```sh
python3 -m Tools.LLMEvaluation select --check \
  --datasets .local/conversion/datasets \
  --tokenizer .local/conversion/models/google--gemma-4-31B-it/tokenizer.json
```

Without `--check` it rewrites both set files. Any change to them requires a new contract version.

## Scoring

`score` scores the output records of one or more runs with the rules of §6.7, except code. The runs of one call share the items, the tokenizer and the scorer, so the baseline and the candidate of a pair belong in the same call.

```sh
python3 -m Tools.LLMEvaluation score \
  --items .local/conversion/gemma4_31b/evaluation/new-build/primary.jsonl \
  --run path/to/baseline-run path/to/candidate-run \
  --tokenizer .local/conversion/models/google--gemma-4-31B-it/tokenizer.json \
  --output .local/conversion/gemma4_31b/evaluation/new-scores
```

`--items` is a file written by `build`. Each run folder holds:

| File | Contents |
|---|---|
| `outputs.jsonl` | The output records of contract §5.1, one JSON object per line |
| `run.json` | The run record of contract §5.1, one JSON object; `score` copies it unchanged into the summary |

`score` first runs the checks of `validate`. It stops when `tokenizer.json` differs in size or SHA-256 from its pin in `contract.json`, when the output folder exists, or when two run folders have the same name. A malformed record does not stop it; the record is reported and not scored. Like `select`, it needs `tokenizers`; the rules use only the standard library, and the unit tests replace the tokenizer. The scored text is `output_ids` decoded with the pinned tokenizer, skipping special tokens; text in a record is never used. `python_tests` items stay `pending_code` for the code scorer, and no output code is run.

The output folder gets one folder per run, named after the run folder. Its `scores.jsonl` has one line per item of `--items`, in that order, as ASCII JSON:

| Field | Value |
|---|---|
| `id`, `category` | From the item |
| `kind` | The scorer kind |
| `correct` | `true` or `false`; `null` for `unscored`, `pending_code`, `missing` and `malformed` |
| `state` | `scored`, `unscored`, `pending_code`, `missing` (no record), `malformed` (only malformed records carry its id), or the reason nothing could be read, which makes the item incorrect: `no_digit`, `no_marker`, `no_integer`, `no_json` or `duplicate_key` |
| `parsed` | The value read: the digit, the integer, the normalized text, the first JSON object or the stripped non-empty lines; `null` when nothing was read |
| `f1` | `reading_exact` only: the character F1 |
| `tp`, `fp`, `fn` | `entity_sets` only: the (key, string) pairs of the item |

The character F1 is 2 × overlap ÷ (output length + answer length), where the overlap is that of the character multisets after normalization; it is the maximum over the answers, and 0 when either side is empty. The `entity_sets` predictions are the (key, string) pairs of those of the 8 keys whose values are arrays of strings, each pair counted once. An output without a JSON object, or whose first object has a duplicate key, predicts nothing.

Its `summary.json` holds:

- `contract` (the version and the SHA-256 of `contract.json`), `items` (the file name and SHA-256 of `--items`), `tokenizer` (its SHA-256 and `vocabulary`, its size with the added tokens) and `run` (the copy of `run.json`).
- `checks`: facts about the records. The checks give no verdict, and every item with a well-formed record is scored.
  - `malformed`: every record that does not have the form of contract §5.1, in file order: a line that is not a JSON object, or a missing field or a value of another type. `id` and `input_sha256` are strings, `input_tokens` and `non_finite` integers, `output_ids` a list of token ids, `stop` either `stop_token` or `limit`, `stop_token` a token id when `stop` is `stop_token`, `top2` a list of `[id, probability, id, probability]`, and `error`, when present, a string. A token id is an integer from 0 to below `vocabulary`. A record is listed by its id when the id is a string, and otherwise by its line number.
  - The other checks are sorted lists of ids. `missing`: an item that no record names; a malformed record with a string id names its item, and a record without one names none. `item_hash_differs`: an item whose `item_sha256` differs from `sets/*.json`.
  - The remaining checks use only the well-formed records. `not_in_set`: a record whose id is not an item. `duplicate`: an id with more than one record; only its first record is scored. `top2_differs`: a record whose `top2` first ids differ from the generated tokens in a value or in number; the generated tokens are `output_ids` followed by `stop_token` when the run stopped on one.
- `categories`: per category, `kind`, `items`, `correct`, `unscored` (items whose `correct` is `null`) and `states` (the count of each state). `entity_sets` adds the sums `tp`, `fp` and `fn` and `micro_f1` = 2·tp ÷ (2·tp + fp + fn); `reading_exact` adds `mean_f1`, the mean character F1. Either is `null` when its denominator is 0.

`score` prints, for each run, the counts per category and the number of ids in each check.

## Data

Retrieve each source anonymously at the revision pinned in `contract.json` into `.local/conversion/datasets/<folder>/`, with `<folder>.integrity.json` beside it. The JCommonsenseQA file comes from the existing `yahoojapan--JGLUE` folder and is only read. The data table in the contract document lists the distributors, licenses and credits. Only ids, keys, hashes, counts and token counts are committed.
