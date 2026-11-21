# Gemma 4 31B quality evaluation

The rules are in the [quality contract](../../docs/specs/gemma4-31b-quality-contract.md) `G31-QC-v1` (issue #666, parent #658). This folder fixes them in machine-readable form before any candidate is measured. Running models, scoring outputs, paired comparison and reports belong to later child issues.

| File | Contents |
|---|---|
| `contract.json` | Roles, pairs, layer 1 limits, generation conditions and the template frame, verdicts, critical failures, scorer kinds, pinned sources and categories |
| `sets/primary.json`, `sets/reserve.json` | Item ids, source keys or generator arguments, `item_sha256` and token counts only |
| `prompts.json` | All prompt wording, the synthetic templates and the `free-ja` instructions |
| `synthetic/` | Synthetic task generators, the independent solvers that read only the rendered text, and the self-authored vocabulary |
| `tests/` | Unit tests and the scoring specification examples (`examples.json`) |

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

## Data

Retrieve each source anonymously at the revision pinned in `contract.json` into `.local/conversion/datasets/<folder>/`, with `<folder>.integrity.json` beside it. The JCommonsenseQA file comes from the existing `yahoojapan--JGLUE` folder and is only read. The data table in the contract document lists the distributors, licenses and credits. Only ids, keys, hashes, counts and token counts are committed.
