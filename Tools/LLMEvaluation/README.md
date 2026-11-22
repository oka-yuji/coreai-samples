# Gemma 4 31B quality evaluation

The rules are in the [quality contract](../../docs/specs/gemma4-31b-quality-contract.md) `G31-QC-v1` (issue #666, parent #658). This folder fixes them in machine-readable form before any candidate is measured, scores output records with every rule except code (issue #668), and compares the scored baseline and candidate of a pair to give the verdict of §7 and its report (issue #671). Running models and code scoring belong to other child issues.

| File | Contents |
|---|---|
| `contract.json` | Roles, pairs, layer 1 limits, generation conditions and the template frame, verdicts, critical failures, scorer kinds, pinned sources and categories |
| `sets/primary.json`, `sets/reserve.json` | Item ids, source keys or generator arguments, `item_sha256` and token counts only |
| `prompts.json` | All prompt wording, the synthetic templates and the `free-ja` instructions |
| `synthetic/` | Synthetic task generators, the independent solvers that read only the rendered text, and the self-authored vocabulary |
| `scoring.py` | The scoring rules of §6.7 except code, the record checks and `score` |
| `comparison.py` | The paired comparison and the verdicts of §7, the report and `compare` |
| `tests/` | Unit tests, the scoring specification examples (`examples.json`), the expected scoring results derived by hand from §6.7 (`scoring_cases.json`) and the expected comparison results derived by hand from §7 (`comparison_cases.json`) |

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

`select` fixes the sets once and can recheck them. It needs both `tokenizers` and `jinja2`; `score` needs only `tokenizers`, and `validate` and `build` need neither. It renders every item with the distributed `chat_template.jinja` and counts tokens with `tokenizer.json`; both files are pinned by SHA-256 in the contract.

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
| `run.json` | The run record of contract §5.1, one JSON object; `score` copies it into the summary, where a `NaN`, `Infinity` or `-Infinity` in it becomes a string because the summary is standard JSON |

`score` first runs the checks of `validate`. It stops before it writes anything when:

- `tokenizer.json` differs in size or SHA-256 from its pin in `contract.json`;
- the ids of `--items` are not exactly those of one frozen set, `sets/primary.json` or `sets/reserve.json`; the message gives the numbers of missing and extra ids;
- the output folder exists;
- two run folder names are equal after NFC normalization and case folding, or a run folder has no name, as the root does.

A malformed record does not stop it; the record is reported and not scored. It makes every run folder before it writes any result, and when the output cannot be completed it removes the output folder that it made. It needs `tokenizers` but not `jinja2`; the rules use only the standard library, and the unit tests replace the tokenizer. The scored text is `output_ids` decoded with the pinned tokenizer, skipping special tokens; text in a record is never used. `python_tests` items stay `pending_code` for the code scorer, and no output code is run.

The output folder gets one folder per run, named after the run folder once its path is made absolute, so `--run .` takes the name of the current folder. Its `scores.jsonl` has one line per item of `--items`, in that order, as standard ASCII JSON:

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

The first JSON object starts at the first `{` from which a standard JSON object decodes. `NaN`, `Infinity` and `-Infinity` are not JSON values, so an object containing one is skipped. A number too large for a float, such as `1e400`, is valid JSON and decodes as infinite; it is compared as a number, and only when `scores.jsonl` is written is it replaced in `parsed` by the string `"Infinity"` or `"-Infinity"`.

Its `summary.json`, also standard ASCII JSON, holds:

- `contract` (the version and the SHA-256 of `contract.json`), `items` (the file name and SHA-256 of `--items`, and `set`, the name of the frozen set it holds), `tokenizer` (its SHA-256 and `vocabulary`, its size with the added tokens), `run` (the copy of `run.json`) and `outputs` (`sha256`, the SHA-256 of the `outputs.jsonl` bytes that were scored). `score` reads `--items` and each `outputs.jsonl` once and takes both the content and the SHA-256 from the same bytes.
- `checks`: facts about the records. The checks give no verdict, and every item with a well-formed record is scored.
  - `malformed`: every record that does not have the form of contract §5.1, in file order: a line that is not a standard JSON object (a line with `NaN`, `Infinity` or `-Infinity` is not), or a missing field or a value of another type. `id` and `input_sha256` are strings, `input_tokens` an integer of at least 1, `non_finite` an integer of at least 0, `output_ids` a list of token ids, `stop` either `stop_token` or `limit`, `stop_token` a token id when `stop` is `stop_token`, `top2` a list of `[token id, probability, token id, probability]` with two different token ids and probabilities that are numbers from 0 to 1, the first at least the second (`true` and `false` are not numbers, and a number such as `1e400`, which reads as infinite, is out of range), and `error`, when present, a string. A token id is an integer from 0 to below `vocabulary`. A record is listed by its id when the id is a string, and otherwise by its line number.
  - The other checks are sorted lists of ids. `missing`, `not_in_set` and `duplicate` count every record with a string id, well-formed or malformed; a record without one counts for none. `missing`: an item that no record names. `not_in_set`: a record whose id is not an item. `duplicate`: an id with more than one record; only the first well-formed record of an id is scored. `item_hash_differs`: an item whose `item_sha256` differs from `sets/*.json`.
  - `top2_differs` uses only the well-formed records: a record whose `top2` first ids differ from the generated tokens in a value or in number; the generated tokens are `output_ids` followed by `stop_token` when the run stopped on one.
- `categories`: per category, `kind`, `items`, `correct`, `unscored` (items whose `correct` is `null`) and `states` (the count of each state). `entity_sets` adds the sums `tp`, `fp` and `fn` and `micro_f1` = 2·tp ÷ (2·tp + fp + fn); `reading_exact` adds `mean_f1`, the mean character F1. Either is `null` when its denominator is 0.

`score` prints, for each run, the counts per category and the number of ids in each check.

## Comparison

`compare` compares the scored baseline and candidate of one pair and writes the verdict of contract §7 with the required report items of §7.4. It uses only the standard library, needs no tokenizer, and does not score again: it reads what `score` wrote, the run records and the output records.

```sh
python3 -m Tools.LLMEvaluation compare \
  --baseline path/to/baseline-run --candidate path/to/candidate-run \
  --scores .local/conversion/gemma4_31b/evaluation/new-scores \
  --items .local/conversion/gemma4_31b/evaluation/new-build/primary.jsonl \
  --output .local/conversion/gemma4_31b/evaluation/new-verdict
```

| Option | Value |
|---|---|
| `--baseline`, `--candidate` | The run folders, with `run.json` and `outputs.jsonl` |
| `--scores` | The output folder of the one `score` call that scored both runs; `compare` reads its folders named after the two run folders |
| `--items` | The `build` file that was scored; its SHA-256 must be `items.sha256` of both summaries |
| `--output` | A new folder |
| `--rerun-scores` | Optional: the folder that `score` wrote for a rerun of the candidate's assets with another call shape, holding its `summary.json` and `scores.jsonl` |
| `--layer1` | Optional: the layer 1 result file; the record keeps its file name and SHA-256 |
| `--substitute-reason` | The reason why `fp16` could not be run, required to use `reference` as the baseline of `w8` or `w4` |

`compare` first runs the checks of `validate`. It stops before it writes anything when:

- the output folder exists;
- the two run folders have the same name after NFC normalization and case folding, which one `score` call cannot hold;
- the ids of `--items` are not exactly those of one frozen set, an id repeats, or an item's category or `token_count` differs from the set file;
- the `scores.jsonl` of a run does not have exactly one row per item of `--items` with its category and scorer kind;
- the files are not those that were scored, and the pair has to be scored again: `--items` has another SHA-256 than `items.sha256` of either summary, the `outputs.jsonl` of a run folder has another SHA-256 than `outputs.sha256` of its summary, its `run.json` differs from the `run` copied into the summary, or the summary has no `outputs` because an earlier `score` wrote it;
- the rerun scoring is not that of a rerun of the candidate's assets with only another call shape (contract §7.4): it does not have exactly one row per item of `--items`, it differs from the candidate scoring in `contract`, `items`, `tokenizer` or `scorer`, the run record copied into either scoring lacks `asset`, `inputs`, `implementation`, `call_shape`, `set` or `role` (a `null` counts as lacking), the two run records differ in `asset`, `inputs`, `implementation`, `set` or `role`, or they have the same `call_shape`. The message names each of these that holds;
- the roles in the two `run.json` files are not a pair of `contract.json` (`reference` → `fp16`, `fp16` → `w8`, `fp16` → `w4`). `reference` → `w8` and `reference` → `w4` are accepted only with `--substitute-reason`, and `--substitute-reason` is refused for the other pairs.

`compare` reads `--items`, each `outputs.jsonl` and `run.json` and the `--layer1` file once, so every hash it checks or records and the content it judges come from the same bytes. Each run uses its first well-formed record of each id, read with the record checks of `score`; malformed records reach the verdict only through the `malformed` check of the scoring. When the output cannot be completed, `compare` removes the output folder that it made. It prints the verdict, the number of ids of each reason, the numbers of critical failures, regressions and improvements, and the categories that could not be evaluated.

The output folder gets `verdict.json`, standard ASCII JSON, and `summary.md`, an English summary with the verdict, the reasons, a table per category, the regression and improvement ids, the changed items, the critical failures, the categories that could not be evaluated, the upper bounds, the baseline quality, the code scoring counts, the rerun counts, the layer 1 reference and the claim. Neither file holds output text or public data text. `summary.md` is UTF-8, and a character that UTF-8 cannot encode, such as a lone surrogate that a JSON escape in an id can carry, is written as a backslash escape.

The verdict is the first label of the precedence in `contract.json` whose condition holds: `incomplete` when any reason below holds, `critical_failure` when any critical failure was found, `regression_found` when any item regressed, and `no_regression_found` otherwise. Improvements never offset a regression, and a regression at a near tie counts as any other.

The reasons of `incomplete`, in this order:

| Reason | Ids |
|---|---|
| `hash_mismatch` | A run record or a scoring summary has another contract version or `contract.json` SHA-256 than this contract, a run record names another set than `--items`, the two summaries differ in `items`, `tokenizer` or `scorer`, or a summary reports `item_hash_differs`. The ids are those of `item_hash_differs`, possibly none |
| `run_unfinished` | The run record of either run has an integer `items` and an `inputs.count`, and they differ or `finished` is absent or `null`, as in the record of a run that has not finished: the generating command writes the number of finished items to `items`, the number of input records to `inputs.count` and, last, the finish time to `finished`, so a run stopped after its last output and before `finished` has equal counts. A run record without `items` or `inputs.count` is not judged, whatever its `finished`. No ids |
| `missing_output`, `not_in_set`, `duplicate_id`, `top2_differs` | The ids of the scoring checks `missing`, `not_in_set`, `duplicate` and `top2_differs` of either run |
| `malformed_record` | The scoring check `malformed` of the baseline, then the entries of the candidate that the list does not hold yet, in order and without repeats; an entry is an id or a line number |
| `input_differs` | Items whose baseline and candidate records differ in `input_sha256` |
| `token_count_differs` | Items with a `token_count` whose `input_tokens` differs in a record of either run |
| `baseline_error` | Items whose baseline record has an `error` |
| `unscored_item` | Items of a scored category whose `correct` is `null` in either run with a state other than `missing` and `malformed`, for example `pending_code` |
| `score_inconsistent` | Items whose correctness differs although the baseline and candidate `output_ids` are equal; the scored text is decoded from `output_ids` alone, so a different stop does not explain it |

Except for `malformed_record`, the ids are sorted; `run_unfinished` has none. The generated tokens of a record are `output_ids`, followed by `stop_token` when the run stopped on one.

A critical failure is judged for each item with a candidate record: `non_finite_logits` when `non_finite` is above 0, and otherwise `runtime_error` when the record has an `error`. When the baseline also has a record, `empty_output`, `control_token` and `repetition` count only when the candidate output has the property and the baseline output does not. A repetition is a run of `min_consecutive` (4) copies, in a row, of a sequence of `min_length` (8) or more tokens of `output_ids`. The control tokens are `control_token_ids` of `contract.json`.

`verdict.json` holds:

| Field | Value |
|---|---|
| `verdict` | The label |
| `incomplete` | The reasons that hold, in the order above, each as `reason` and `ids` |
| `critical_failures` | `id` and `kind` for each failure, items in set order, and for one item `runtime_error`, `non_finite_logits`, `empty_output`, `control_token`, `repetition` in that order |
| `regressions`, `improvements` | Sorted ids of the items of a scored category where the baseline is correct and the candidate incorrect, or the reverse; an item with `null` on either side is not counted |
| `changed` | For each regression and improvement, sorted by id: `id`, `position`, the first output position where the generated tokens differ (the length of the shorter list when one starts the other, `null` when they are equal), and `baseline_gap`, the first minus the second probability of the baseline `top2` there (`null` without an entry) |
| `categories` | Per category of the set: `kind`, `items`, `baseline_correct`, `candidate_correct`, `regressions` and `improvements` |
| `not_evaluable` | Sorted scored categories whose baseline has fewer than `min_baseline_correct` (10) correct items |
| `upper_bounds` | For each `upper_bound` category of `contract.json` in the set with no regression, 1 − 0.05^(1/N) with N its item count |
| `baseline_quality` | `empty_output`, `control_token` and `repetition`: the ids of the baseline outputs that are empty, hold a control token or hold a repetition, in set order |
| `code` | `baseline` and `candidate`, the `code` of each summary, and `scorer`, that of the baseline summary; each is `null` when the summary has none |
| `rerun` | `correct_to_incorrect` and `incorrect_to_correct` from the candidate scores to the rerun scores, or `null`; never used for the verdict |
| `layer1` | `file` and `sha256` of `--layer1`, or `null` |
| `claim` | For `no_regression_found` only, `layer2.claim` of `contract.json`, followed by `; not evaluated: ` and the `not_evaluable` categories joined by `, ` when there are any; otherwise `null` |
| `contract` | The contract version and the SHA-256 of `contract.json` |
| `runs` | `baseline` and `candidate`: the two `run.json` records; with `--rerun-scores`, also `rerun`: the run record copied into the rerun scoring |
| `substitute_reason` | The text of `--substitute-reason`, or `null`; a blank text counts as none |

## Data

Retrieve each source anonymously at the revision pinned in `contract.json` into `.local/conversion/datasets/<folder>/`, with `<folder>.integrity.json` beside it. The JCommonsenseQA file comes from the existing `yahoojapan--JGLUE` folder and is only read. The data table in the contract document lists the distributors, licenses and credits. Only ids, keys, hashes, counts and token counts are committed.
