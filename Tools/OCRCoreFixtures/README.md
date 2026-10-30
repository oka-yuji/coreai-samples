# OCRCore reference fixtures

`generate.py` writes the JSON resources in `CoreAISamplesKit/Tests/OCRCoreTests/Resources/`.
They pin the TeleOCR host-side contracts that `OCRCore` reproduces in Swift. The script never
reads model weights: it loads only the processor, tokenizer and configuration from the pinned
snapshot, and calls the original functions of `modeling_naviocr.py` through small stand-in
objects.

## Run

```sh
.local/conversion/teleocr/.venv/bin/python -I Tools/OCRCoreFixtures/generate.py \
  --knowledge /path/to/teleocr-20261022 \
  --out CoreAISamplesKit/Tests/OCRCoreTests/Resources
```

- Snapshot: `XingChen-AGI/TeleOCR@70a19ec6a3b036311af19ae5e762e650f41a5980` in the Hugging Face
  cache (`--snapshot` overrides the path). Before anything is loaded, the script copies the 13
  files pinned in `Tools/Conversion/recipes/teleocr/model.json` other than the weights into a new
  temporary folder, checks their size and SHA-256 there, and stops if any differs. The processor,
  configuration and `modeling_naviocr.py` are loaded only from that copy, so other files and cached
  bytecode in the snapshot are never used. Loading the processor then executes the pinned remote
  code, as `trust_remote_code=True` does.
- `--knowledge` is the #178 investigation folder. The script reads `fixtures/F1.png`,
  the F2 and F4 image sizes, `records/outputs/F2-table-bf16.txt`,
  `records/outputs/F2-layout-bf16.txt`, `records/outputs/F4-layout-bf16.txt` and the F2 grid in
  `records/scores.json`.
- Environment used for the committed files: Python 3.11.9, torch 2.9.0, torchvision 0.24.0,
  transformers 4.57.3, Pillow 13.0.0 and NumPy 2.4.6 on an Apple silicon CPU. `provenance.json`
  records the versions and the SHA-256 of every file.
- Regeneration is deterministic. To check it, write to an empty folder and compare:
  `diff -r <folder> CoreAISamplesKit/Tests/OCRCoreTests/Resources`.

## Contents

| File | Reference |
|---|---|
| `images.json` | `smart_resize` of the processor; torchvision bicubic with antialiasing (the fast image processor) and Pillow bicubic (the model card's 1036 px layout resize) on a synthetic pattern |
| `vision.json` | Processor `pixel_values` digests, `get_window_index`, its inverse, and window-order rotary positions with the cosine and sine tables of `rot_pos_emb` |
| `prompts.json` | Tokenizer encodings of the template segments, processor `input_ids`, `get_rope_index` positions and `rope_deltas`, and the mRoPE section merge of `apply_multimodal_rotary_pos_emb` |
| `generation.json` | `RepetitionPenaltyLogitsProcessor(1.05)` on Float32 logits, greedy argmax, and the EOS and maximum-length stopping criteria |
| `layout.json` | The #178 `layout_blocks.py` parser, including outward rounding to source pixels |
| `tables.json` | The model card's OTSL parser (`_otsl_parse_texts` and its grid placement) and the row widths before padding |

Synthetic images use the integer pattern in `pattern()`; the Swift tests rebuild the same
pixels. The card parser raises `IndexError` when a horizontal span follows a skipped row;
the fixture records that case with `cardRaises` and expects the bounded count that
`score.py` in #178 uses instead.
