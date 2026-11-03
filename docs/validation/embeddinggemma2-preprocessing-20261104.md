# EmbeddingGemma 2 preprocessing across the lock update

2026-11-04. Issue #478 / PR #479. The conversion environment's lock moved Pillow, filelock and setuptools to versions at or above the first patched versions named by the open Dependabot advisories. Source model revision remains `914f7f89142e33e77833254d9c9b90c3cef7303b`, text 128 / vision 70.

## What was compared

Two environments were created with `uv sync --frozen` from the previous and the updated `pyproject.toml`/`uv.lock`, and both ran the same repository code. In the lock, only the three packages below changed version, and the project's own requirement now names Pillow 13.0.0. The other 64 of the 68 entries, including torch, torchvision, torchao, transformers, coreai-core/opt/torch, numpy, safetensors and huggingface-hub, are identical.

The eight image fixtures come from the function the reference stage uses: five copied sample PNGs and three deterministic gradients encoded to PNG by each environment's Pillow. For every fixture, the decoded RGB pixels (`Image.open(...).convert("RGB")`) and the processor's `input_ids`, `attention_mask`, `pixel_values` and `image_position_ids` (same arguments as the reference stage) were compared by shape, dtype and every element (`numpy.array_equal`). The processor's image processor is `Gemma4ImageProcessor` on the torchvision backend; it resizes with `torchvision.transforms.v2.functional.resize`, so Pillow decodes and converts the images and encodes the gradients. The twelve text fixtures do not involve image decoding and were not compared.

| Package | Previous lock | Updated lock |
|---|---|---|
| Pillow | 12.2.0 | 13.0.0 |
| filelock | 4.2.0 | 4.2.7 |
| setuptools | 84.0.0 | 84.2.1 |
| torch / transformers / numpy | 2.9.0 / 5.19.0 / 2.3.5 | unchanged |
| Python | 3.11.9 | unchanged |

Apple M5 Max 128 GB, macOS 27.2 (26B123), uv 0.9.2. The model weights were not loaded.

## Result

All eight fixtures are equal. `input_ids` and `attention_mask` are (1, 128) int64, `pixel_values` is (1, 630, 768) float32 and `image_position_ids` is (1, 630, 2) int64; every maximum absolute difference is 0.

| Fixture | Pixels | input_ids | attention_mask | pixel_values | image_position_ids | PNG bytes |
|---|---|---|---|---|---|---|
| 01-oyakodon | equal | equal | equal | equal | equal | equal (copied) |
| 02-ginger-pork | equal | equal | equal | equal | equal | equal (copied) |
| 03-omelette | equal | equal | equal | equal | equal | equal (copied) |
| 04-gratin | equal | equal | equal | equal | equal | equal (copied) |
| 05-caprese | equal | equal | equal | equal | equal | equal (copied) |
| gradient_square 144×144 | equal | equal | equal | equal | equal | equal (encoded) |
| gradient_wide 320×96 | equal | equal | equal | equal | equal | equal (encoded) |
| gradient_tall 96×320 | equal | equal | equal | equal | equal | equal (encoded) |

PNG byte equality is reported but is not part of the verdict, because a different encoder could change the bytes without changing the decoded pixels. Evidence: [comparison.json](embeddinggemma2-preprocessing-20261104/comparison.json). The arrays remain local.

```sh
uv lock --upgrade-package pillow --upgrade-package filelock --upgrade-package setuptools --project Tools/Conversion/recipes/embeddinggemma2
UV_PROJECT_ENVIRONMENT=<environment> uv sync --project <recipe folder> --frozen --no-dev --python 3.11
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false <environment>/bin/python -m Tools.Conversion.adapters.embeddinggemma2.preprocessing dump --contract Tools/Conversion/recipes/embeddinggemma2/model.json --source <snapshot> --images CoreAISamples/Resources/Recipes --output <dump>
python -m Tools.Conversion.adapters.embeddinggemma2.preprocessing compare --left <previous dump> --right <updated dump> --output comparison.json
```

`<recipe folder>` is a folder holding the previous `pyproject.toml` and `uv.lock`, or `Tools/Conversion/recipes/embeddinggemma2` for the update. `dump` checks the fourteen pinned source files of `<snapshot>` by size and SHA256 before reading the processor. `compare` exits with 1 unless every fixture is equal.

## Limits

This checks preprocessing equivalence only. It is not the six-stage fidelity gate, the published bundle was not reconverted, and the next conversion must start from setup with the new lock.
