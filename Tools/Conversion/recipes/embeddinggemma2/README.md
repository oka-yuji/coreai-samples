# EmbeddingGemma 2 text and vision

This maintainer recipe produces the bundle used by the [local image search demo](../../../../docs/embeddinggemma2.md). It runs entirely from CoreAI-Samples. It does not depend on the retired CoreLLM workspace, upload weights, or provide a general conversion service.

## Setup and run

Use Apple Silicon, macOS 27, Xcode 27 with Metal Toolchain, uv, and Python 3.11. From the repository root, download the pinned snapshot once, then prepare its dedicated environment:

```sh
hf download google/embeddinggemma-2 --revision 914f7f89142e33e77833254d9c9b90c3cef7303b
python3 -m Tools.Conversion.local setup --recipe embeddinggemma2
python3 -m Tools.Conversion.local plan --recipe embeddinggemma2
python3 -m Tools.Conversion.local run --recipe embeddinggemma2
```

An existing complete snapshot can be supplied to setup with `--snapshot /path/to/<revision>`. Setup also recognizes `.local/conversion/inputs/embeddinggemma2/<revision>`. Fourteen source files are pinned by size and SHA256. Setup never downloads model weights. `--workspace` selects another ignored workspace; pass the same workspace on subsequent commands.

The environment is `.local/conversion/embeddinggemma2/.venv`, separate from Qwen. Its committed `uv.lock` pins the transitive dependencies. The direct stack includes Torch 2.9.0, Transformers 5.19.0, torchao 0.17.0 and coreai-torch 0.4.3. Setup imports the processor/model classes before writing its stamp. With this Torch version, resolving torchao 0.18.0 caused an import failure for `torch.nn.functional.ScalingType`; the failed attempt was retained without reusable artifacts.

Setup builds `embedding-verify` and stamps its binary, Swift sources, Package.resolved, Xcode/SDK, and environment inputs. A changed verifier, toolchain, pyproject/lock, or direct installed version requires setup again. Changing a source worktree can require setup even when the existing Python environment is reused. Model jobs and setup share `~/Library/Caches/coreai-samples/heavy-job`.

## Outputs and reuse

The run prints `.local/conversion/work/embeddinggemma2/runs/<run>/report.json`. Require both `status: completed` and `verification: passed`. The `bundle` stage's `artifacts` path contains the complete `bundle` directory. Select that directory in the Mac demo, or copy its contents as the iPhone app's `Documents/embeddinggemma2` folder. A separately copied convenience bundle is not automatically refreshed on later runs.

The six stages are prepare → reference → text export → vision export → bundle → native verification. Prepare always checks identities. The other five reuse only successfully completed, unchanged artifacts. Text and vision both depend on the original reference because conversion also compares the wrapper's FP32 results before exporting. Changing the numerical fixtures therefore reruns both exports; UI-only search example changes do not.

Repeat `run` to reuse successful stages, or add `--refresh` for fresh artifacts. The output contains safe FP16 text/vision assets, an exact BF16 vocabulary table, the original tokenizer and card, Apache-2.0 license text, conversion notices and a SHA256 manifest. Packaging requires explicit source/reference/text/vision identities. The old `package_bundle.py ASSETS OUTPUT` experiment interface is retired; the recipe supplies the four named inputs and rejects old contracts without a source revision.

## Verification scope

The conversion preserves the previously measured scaled RMSNorm, materialized KV and scaled final L2 operations. It supports text length 128, vision budget 70 and 768-dimensional embeddings. It does not include audio, video segmentation, long text, OCR or generative responses.

The original FP32 processor/model generates twelve text and eight image fixtures. The text wrapper FP32 output must satisfy cosine ≥ 0.99999 and maximum difference ≤ 0.0001 before export. Vision soft-token tensor metrics are recorded as diagnostics; their export-time values are not a separate pass gate. Both encoders must pass the final native-vector comparison below. Native Swift raw text/image inputs must satisfy cosine ≥ 0.999, maximum difference ≤ 0.01, finite values and norm deviation ≤ 0.001. All twenty ordered, unique IDs and token comparisons are required. Python independently recalculates the returned native vectors; a reported `passed` flag alone is insufficient.

These tests measure conversion fidelity, not broad retrieval quality. The [device/search report](../../../../docs/validation/embeddinggemma2-ready-20261016.md) separately retains the held-out search failure and the user's pending acceptance.

```sh
.local/conversion/embeddinggemma2/.venv/bin/python -m unittest discover -s Tools/Conversion/tests -v
```

The tests include existing Qwen/orchestrator regressions, source identities, environment setup, mixed packaging rejection and native-vector failures. Raw commands, reports, paths, images and weights remain ignored under `.local`; only reviewed, bounded evidence belongs in public docs. The [conversion record](../../../../docs/validation/embeddinggemma2-pipeline-20261016.md) includes fresh/reuse conditions and limits.


## Experimental text INT4 / vision INT8

The optional weight-only PTQ profile includes INT4 text projections and vocabulary rows, INT8 vision projections and the learned position table. Norms, biases, and scalar parameters retain floating precision; the quality report lists retained parameters. Activations use the same safe FP16 export path. This is a PTQ experiment, not Google's QAT checkpoint, and is not the default.

```sh
python3.11 -m Tools.Conversion.local setup --recipe embeddinggemma2 --precision int4_int8
python3.11 -m Tools.Conversion.local plan --recipe embeddinggemma2 --precision int4_int8
python3.11 -m Tools.Conversion.local run --recipe embeddinggemma2 --precision int4_int8
```

This uses `recipe-int4-int8.json` and a separate `work/embeddinggemma2-int4-int8` directory with seven stages. The added `quantized_reference` stage produces a PyTorch reference from the same quantization configuration as the two exports. The final native gate still requires the original cosine/difference/finite/norm tolerances, but against that matched PTQ reference. `quantization-quality.json` reports degradation and fixed-fixture ranking changes from original precision. `comparison.json.original_precision_comparison` is diagnostic and does not override the matched-reference pass gate.

INT4 vocabulary storage is a row-major signed nibble array, low nibble first, values -7 through 7, with one little-endian FP32 scale per vocabulary row. The runtime decodes only selected rows and applies the original sqrt(512) embedding scale. The manifest precision is `int4-int8-ptq-s128-v70`; table data, scales, matched reference vectors and quality metadata are required and hash-checked. The old BF16 bundle remains supported.

Use `Documents/embeddinggemma2-int4-int8` on iPhone, keeping `Documents/embeddinggemma2` for FP16. The image-search model menu and camera Search settings offer both installed folders. Stop camera search before switching. `verify_device.py` selects the destination from the bundle manifest and passes `--embedding-profile int4_int8` for the quantized self-test; it does not replace the FP16 folder. Configure ignored `Config/Local.xcconfig` for a signed iOS build before device installation.

See [the comparison record](../../../../docs/validation/embeddinggemma2-int4-int8-20261018.md) for measured coverage and quality differences. A completed conversion is not a claim that INT4 maintains the original task quality. Repeat `run` without `--refresh` only validates/reuses cache; do not report it as a fresh benchmark.
