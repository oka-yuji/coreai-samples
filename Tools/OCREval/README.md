# TeleOCR evaluation

A fixed evaluation set, its scorer and the original PyTorch reference runs for the
TeleOCR conversion. Quantization comparisons and the Core AI OCR pipeline score their
outputs with the same code and compare them with the stored baseline. This is maintainer
tooling, not a stable public API.

## Environment

Run from the repository root on macOS with Apple Silicon. The versions are pinned to the
TeleOCR investigation environment (Python 3.11, torch 2.9.0, torchvision 0.24.0,
transformers 4.57.3, Pillow 13.0.0, NumPy 2.4.6).

~~~sh
export UV_PROJECT_ENVIRONMENT="$PWD/.local/conversion/ocr_eval/.venv"
uv sync --frozen --no-dev --project Tools/OCREval
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
~~~

`run-reference` needs the snapshot `XingChen-AGI/TeleOCR@70a19ec6a3b036311af19ae5e762e650f41a5980`
in the Hugging Face cache. It never downloads the model and refuses to run when a file
that affects the output differs from the pinned SHA-256.

## Commands

~~~sh
$PY -m Tools.OCREval make                   # F1-F5 and S01-S30 -> .local/research/teleocr-eval/set
$PY -m Tools.OCREval make --omnidocbench    # pinned subset -> .local/research/teleocr-eval/omnidocbench
$PY -m Tools.OCREval run-reference --out DIR [--set DIR] [--only ID ...] --job NAME --session NAME
$PY -m Tools.OCREval score --outputs DIR/outputs [--set DIR] [--out FILE]
$PY -m Tools.OCREval score --results DIR [--set DIR] [--out FILE]
$PY -m Tools.OCREval score --selftest
$PY -m unittest discover -s Tools/OCREval/tests -v
~~~

`make` is deterministic: the same environment regenerates byte-identical images and
ground truth, and `manifest.json` lists their SHA-256 values. `run-reference` takes the
shared heavy-job lock (`COREAI_HEAVY_JOB_LOCK` or `~/Library/Caches/coreai-samples/heavy-job`),
waits up to `--wait-minutes` for other model jobs, starts one process per run and renames
the lock to `heavy-job-done-<job>-<time>` when it finishes. Each run uses bf16 on MPS,
greedy decoding, the checkpoint's repetition penalty 1.05, at most 4,096 new tokens and a
600-second generation limit. Outputs are written as `outputs/<id>-<prompt>-bf16.txt`
(and `.raw.txt` with special tokens), one line per run in `runs.jsonl`, and the
environment and lock times in `invocations.jsonl`. A run still alive after 900 seconds is
killed and recorded as a failed sample; the next sample continues. Each invocation record
lists the completed samples and, when it stopped early, why (`aborted`). Missing ground
truth or images stop the command before the lock is taken. Records contain no environment
variables or home-directory paths; console logs go to `--logs`.

## Evaluation set

| Group | Samples | Run |
|---|---|---|
| `f178` | F1-F5 from the earlier TeleOCR investigation: body, table, specification, two columns, camera. The ground truth is copied in `data/`; the images are regenerated identically. Vertical text is out of scope. | F1, F3, F5: full-page text. F2: table crop (and full-page text). F4: layout then block text (and full-page text). |
| `body`, `numbers`, `table`, `columns`, `camera` | S01-S30, six per stratum. Within each stratum every font (Hiragino Mincho W3, Kaku Gothic W3 and W6), body size (28, 36, 44 px) and degradation (none, Gaussian blur, perspective) is used twice, arranged with seed 313. The camera stratum uses the camera chain instead (perspective, 2-degree rotation, blur, brightness gradient, JPEG 80). The text in `data/corpus.json` is fictional. | Full-page text, except a table crop for `table` and layout then block text for `columns`. |
| `omnidocbench_*` | 24 blocks from OmniDocBench (rules below). Aggregated separately. | Annotation-box crop with the text or table prompt. |
| user photos | Not provided yet. Put images and ground truth JSON in `.local/research/teleocr-eval/user/` and score that directory separately. Never commit them. | As declared in each JSON. |

Images are never committed. The synthetic pages are rendered with the macOS Hiragino
fonts and are regenerated from the seed instead. OmniDocBench pages are third-party
content released for research and non-commercial use only.

Ground truth JSON fields: `id`, `kind` (`text`, `mixed`, `table`, `two_column`,
`camera`), `stratum`, `rule` (the judgment rule F1-F5), `pipelines` (`text`, `table`,
`two_stage`; the first is scored), the content (`lines`; `title` with `cells` and
`table_bbox`; or `title` with `columns` and `line_boxes`), optional `crop_px` and
`image`, `conditions` for synthetic pages, and `redistribute: false` for OmniDocBench.

### OmniDocBench subset

`data/omnidocbench.json` pins dataset revision `aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec`,
the SHA-256 of `OmniDocBench.json`, the selection rules, the 24 selected image names and
annotation IDs, and the image SHA-256 values. `make --omnidocbench` downloads only the
annotation file and those images, and stops when the selection or a hash differs.

- Pages whose `language` is `simplified_chinese` or `english`, excluding `note` and
  `historical_document`.
- Text: `text_block`, not ignored, 40-300 characters, no `$`, `\` or `<`.
- Tables: `table`, not ignored, 3-12 rows and at most 8 columns after expanding
  `rowspan`/`colspan`, no `$`, `\` or `<` in cells.
- Order: ascending SHA-256 of `313|<image_path>|<anno_id>`; one block per page for each
  kind and language; 8 text blocks and 4 tables per language.
- Crop: the annotation polygon's bounding box plus 4 px.

Committed scores for these samples omit every field that contains text. Raw outputs stay
under `.local`.

## Scoring

Scoring follows the earlier investigation's fixed definition without changes:

- Normalization: NFKC, dashes to `-`, wave dashes to `~`, LaTeX macros and degree forms,
  tags, decoration characters and whitespace removed. `CER` uses normalized text,
  `cer_raw` only NFC and whitespace removal.
- Numbers: one extractor on both sides and a one-to-one multiset match; recall
  (`numbers_match`) and precision.
- Tables: OTSL rows before padding must match the ground-truth shape (`shape_strict_match`),
  then cells by position (`cell_match`) and as a multiset (`cell_match_unordered`).
- Reading order: each ground-truth line is located in the prediction; unmatched lines
  (`unmatched_rate`) and the edit distance of the found order (`read_order_edit`).
- Layout (two-stage pages): a ground-truth line whose box centre lies in no layout block
  is a layout miss (`layout_missing_rate`); `cer_covered` scores the block text against
  the covered lines only, separating recognition errors from layout misses.

| Rule | Good | Conditional |
|---|---|---|
| F1 | CER <= 0.02 | CER <= 0.10 |
| F3 | CER <= 0.03 and numbers >= 0.90 | CER <= 0.10 and numbers >= 0.70 |
| F2 | strict shape and cells >= 0.90 | unordered cells >= 0.70 |
| F4 | CER <= 0.05 and reading order <= 0.10 | CER <= 0.15 |
| F5 | CER <= 0.10 and numbers >= 0.80 | CER <= 0.25 |

Judgment labels and the `f178` compatibility section keep the original Japanese
wording from `data/labels.json`. Strata are summarized by sample counts, judgment counts
and arithmetic means; no single overall score is produced.

## OCR result JSON

`score --results DIR` reads one `<id>.json` per sample. This minimal draft is the shape
the scorer needs; the OCR pipeline issue fixes the final format.

~~~json
{
  "schema": "teleocr-ocr-result/0",
  "image": {"name": "F4.png", "size_px": [1240, 1754]},
  "layout": {"status": "parsed", "resize": 1036},
  "blocks": [
    {"label": "text", "box_px": [84, 124, 1052, 181], "prompt": "text", "text": "...",
     "otsl": null, "cells": null, "seconds": 2.1}
  ]
}
~~~

Blocks are read in order. Text samples use the block texts joined by newlines; table
samples use the first block with `otsl`; `box_px` feeds the layout metrics.
`"layout": {"status": "unparsed"}` marks a page read without layout boxes.

## Baseline

`docs/results/teleocr-eval/2026-10-25/` stores the original-model outputs, run records and
scores for F1-F5 and S01-S30, and only run records and redacted scores for the
OmniDocBench subset. `set.json` lists the input hashes the baseline used.
