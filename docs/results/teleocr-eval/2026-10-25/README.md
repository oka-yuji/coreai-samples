# TeleOCR evaluation baseline, 2026-10-25

Original model `XingChen-AGI/TeleOCR@70a19ec6a3b036311af19ae5e762e650f41a5980` in PyTorch:
bf16 on MPS, greedy decoding, the checkpoint's repetition penalty 1.05, at most 4,096 new
tokens, one process per run. M4 Max with 128 GB, macOS 27.0 (26A428), Python 3.11.9,
torch 2.9.0, transformers 4.57.3. The set, the pipelines and the scoring are described in
[`Tools/OCREval`](../../../../Tools/OCREval/README.md).

| File | Content |
|---|---|
| `set.json` | Input manifest from `make`: SHA-256 of every image and ground-truth file, font files, Pillow and NumPy versions |
| `outputs/` | Raw outputs for F1-F5 and S01-S30: `.txt` decoded without special tokens, `.raw.txt` with them |
| `runs.jsonl` | One record per run: image and output tokens, load and generation time, peak RSS, physical footprint |
| `invocations.jsonl` | Environment, repository commit, lock wait and hold time for each invocation |
| `scores.json` | `score` output: per-sample metrics, per-stratum summaries and the `f178` section |
| `omnidocbench/` | Run records, manifest and redacted scores for the OmniDocBench subset; raw outputs are not committed |

All 83 runs (59 here, 24 for OmniDocBench) stopped at EOS. Generation took at most 30 s.

| Stratum | Samples | Good / conditional / insufficient | Mean CER | Mean number recall | Mean cell match | Mean reading-order edit | Mean layout miss |
|---|---:|---|---:|---:|---:|---:|---:|
| `f178` | 5 | 5 / 0 / 0 | 0.0086 | 1.000 | 0.950 | 0.000 | 0.000 |
| `body` | 6 | 6 / 0 / 0 | 0.0054 | - | - | - | - |
| `numbers` | 6 | 5 / 1 / 0 | 0.0125 | 0.964 | - | - | - |
| `table` | 6 | 6 / 0 / 0 | - | 1.000 | 0.992 | - | - |
| `columns` | 6 | 5 / 1 / 0 | 0.0194 | - | - | 0.000 | 0.000 |
| `camera` | 6 | 6 / 0 / 0 | 0.0215 | 1.000 | - | - | - |
| `omnidocbench_table_en` | 4 | 4 / 0 / 0 | - | 1.000 | 0.991 | - | - |
| `omnidocbench_table_zh` | 4 | 4 / 0 / 0 | - | 1.000 | 1.000 | - | - |
| `omnidocbench_text_en` | 8 | 8 / 0 / 0 | 0.0015 | - | - | - | - |
| `omnidocbench_text_zh` | 8 | 6 / 0 / 2 | 0.0437 | - | - | - | - |

The `f178` row scores each fixture's primary pipeline, so F4 is the layout-then-block
reading. The `f178` section of `scores.json` keeps the earlier investigation's
structure, where full-page F4 is insufficient and the two-stage result is reported
separately. Rerunning F1-F5 reproduced the earlier investigation (#178): all 20 output
files are byte-identical and every scored field and judgment is equal. The overall
adoption rule differs only because the vertical-text fixture is out of scope.

- S11 (Mincho, 44 px, blur) writes units in LaTeX such as `$1\sim35 \degree\mathrm{C}$`.
  The fixed normalization removes `\sim`, so three ranges lose their wave dash.
- S19 (Gothic W3, 44 px, perspective) splits the title into two layout blocks and reads
  its right half after the left column.
- The OmniDocBench numbers are a Chinese and English reference, not a Japanese result,
  and are not comparable with the model card's end-to-end scores.

Reproduce the scores from the committed outputs:

~~~sh
PY=.local/conversion/ocr_eval/.venv/bin/python
$PY -m Tools.OCREval make
$PY -m Tools.OCREval score --outputs docs/results/teleocr-eval/2026-10-25/outputs --out scores.json
~~~
