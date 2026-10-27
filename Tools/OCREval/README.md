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

`coreai-verify ocr <bundle> <image> --out DIR` (in `Tools/Verify`) reads a page with the
Core AI bundle and writes `DIR/<image name>.json` and `DIR/<image name>.md`. `score --results DIR`
reads one `<id>.json` per sample in this format (`teleocr-ocr-result/1`):

~~~json
{
  "record": "ocr", "schema": "teleocr-ocr-result/1",
  "image": "F4.png", "image_sha256": "...", "image_px": [1240, 1754],
  "layout": {"status": "parsed", "resize": 1036, "text": "<box:068 071 848 103><label:title><up>\n...",
             "seconds": {"vision": 1.4, "first_token": 1.8, "decode": 1.1, "total": 2.9},
             "tokens": {"prompt": 1396, "image": 1369, "output": 73}, "finished_by": "eos", "skipped_boxes": []},
  "blocks": [
    {"index": 0, "label": "title", "box_norm": [68, 71, 848, 103], "box_px": [84, 124, 1052, 181],
     "prompt": "text", "text": "...",
     "seconds": {"vision": 0.1, "first_token": 0.2, "decode": 0.2, "total": 0.4},
     "tokens": {"prompt": 100, "image": 70, "output": 21}, "finished_by": "eos"}
  ],
  "generate_s": 6.0, "load_s": 1.2, "assets": [], "memory": {}, "footprint_peak_bytes": 0
}
~~~

- The page is resized to 1036 x 1036 (Pillow bicubic) for the layout prompt. `layout.text` is
  its output. Each usable `<box>` becomes a block in output order; `box_px` is the crop in
  source pixels, `[x0, y0, x1, y1)`, with the 0-1000 box scaled and rounded outward.
  A crop is usable when both sides are positive and at most 32768, its area is at most
  100000000 pixels, and its aspect ratio is at most 200. Other parsed boxes are recorded
  in `layout.skipped_boxes` with `label`, `box_norm` and `box_px`, and are not recognized.
  The remaining blocks keep their output order and contiguous indices.
- Each block is cropped from the source page and read with the table prompt when its label
  is `table`, otherwise with the text prompt. `text` is the output decoded without special
  tokens and stripped. Table blocks also carry `otsl` (the same string), `cells` (`row`,
  `column`, `row_span`, `column_span`, `text`, assembled with the model card's rules) and
  `grid` (cells expanded over their spans).
- `layout.status` is `unparsed` when the layout output has no usable box. The whole page is
  then read with the text prompt as one block without `label` and `box_norm`; skipped boxes
  remain in the record. With some usable boxes, the status stays `parsed` even if others
  were skipped. Cancellation and model execution errors still fail the page request.
- `seconds` are measured from the start of each request; `decode` is `total` minus
  `first_token`. `generate_s` is the whole page. The run fields (`load_s`, `assets` with
  `main_hash`, `memory`, `peak_rss_bytes`, `footprint_peak_bytes`, `pageins`,
  `wired_delta_bytes`, `os`, `executable_sha256`) are the same as `coreai-verify ocr-block`.
- Scoring: text samples use the block texts joined by newlines, table samples the first
  block with `otsl`, two-stage samples also the layout metrics from `box_px`, and an
  `unparsed` layout adds `layout_unparsed: true`.
- The pipeline reads every page in two stages. The baseline reads `body`, `numbers` and
  `camera` pages and F1, F3 and F5 as full-page text, and tables from the ground-truth
  `table_bbox` crop, so compare those strata by judgment, not by string.

## Baseline

`docs/results/teleocr-eval/2026-10-25/` stores the original-model outputs, run records and
scores for F1-F5 and S01-S30, and only run records and redacted scores for the
OmniDocBench subset. `set.json` lists the input hashes the baseline used.
