# JEV-27B fixed sets

`F.json`, `Q.json`, `Cal.json`, `H.json` and `J.json` fix the rows used to check and evaluate
the JEV-27B conversion. They hold identifiers and hashes only. `jglue.json` pins the JGLUE
files and the JNLI option labels. Restore rows with

~~~python
from Tools.Conversion.adapters.jev27b.fixtures import load_rows
rows = load_rows("Tools/Conversion/recipes/jev27b/fixtures/F.json", corpus_dir, jglue_dir)
~~~

`load_rows` rebuilds `state`, `question`, `options` and the corpus `target` or JGLUE `label`,
and fails unless the hash, kind and option count match. Rows built from several corpus states
have no target. The hash does not cover targets or labels: JGLUE files are compared with their
pins when they are read, and the corpus must be checked with
`Tools.Conversion.adapters.jev27b.prepare.check_integrity` before its targets are used, as the
`fixture` step does.

## Schema

Each file has `name`, `version` (1), `seed`, `sources` (dataset and JGLUE revisions, SHA-256
of the tokenizer), `summary` (row counts by kind, source and band) and `rows`. Every row has:

| Field | Meaning |
|---|---|
| `id` | corpus id, `jglue:<task>:<id>`, `synthetic-long:<band>:<n>`, `boundary:<case>` or `card-example:<name>` |
| `source` | `test_set_30k`, `test`, `calibration`, `jglue:<task>`, `synthetic-long`, `boundary` or `card-example` |
| `kind`, `n_options` | decision kind and option count |
| `sha256` | SHA-256 of the UTF-8 text `state + "\n" + question + "\n" + options`, with `options` as compact JSON (`ensure_ascii=False`) |
| `tokens`, `band` | token count of the rendered `bare-v1` input with the distributed tokenizer and `add_special_tokens=False`, and its band (`<=512`, `513-2048`, `2049-4096`, `4097-8192`, `null` above 8,192) |
| `has_target` | whether the restored row carries a corpus target or a JGLUE label |
| `family` | corpus family of the sampled or first row; `null` for JGLUE and card rows |
| `parts`, `cut` | `test_set_30k` ids whose states are joined with a blank line, and the number of characters kept from the last state; `null` otherwise |

## Procedure

Rows with source `yuri_v1` are excluded everywhere. Each set uses
`random.Random("20261102:<name>")` of Python 3.11 over rows sorted by id.

- Stratified sampling: per kind, families are sorted, each gets `count // families` rows and
  `count % families` families drawn from the generator get one more.
- F (302 rows): 240 stratified `test_set_30k` rows (80 per kind); 8 rows per JGLUE task; 30
  synthetic long rows, 10 per band above 512 tokens with kinds in the order noul, choice,
  score, which take the question and options of their first part and add whole states until a
  drawn target between the band's lower bound and its limit minus 512 is reached; boundary
  rows: two choice rows with 2 options, two with 16, one noul row cut to exactly 8,192 tokens
  and one choice row cut to 8,193 tokens for the rejection test; the two card examples.
- Q (900): stratified `test_set_30k` rows, 300 per kind, excluding every id used by F,
  including the parts of joined rows.
- Cal (3,000): 1,000 `calibration` rows per kind. H (1,500): 500 `test` rows per kind.
- J (600): per JGLUE task, `random.Random("20261102:J:<task>")` draws 208 ids from the sorted
  validation ids; the first 8 belong to F and the other 200 to J.

JGLUE rows use only JGLUE fields and the pinned labels:

| Task | Kind | `state` | `question` | `options` | label |
|---|---|---|---|---|---|
| JNLI | choice, 3 | `sentence1` (premise) | `sentence2` (hypothesis) | `jnli_options`, in the order of `jnli_labels` (entailment, contradiction, neutral) | index |
| JCommonsenseQA | choice, 5 | empty | `question` | `choice0` to `choice4` | index |
| JSTS | score | `sentence1` | `sentence2` | `0` to `5` | similarity 0 to 5 |

The prompt wording for the Japanese evaluation is fixed separately before it is measured. The
`fixture` step of `recipe.json` regenerates every set and requires byte-identical files.
