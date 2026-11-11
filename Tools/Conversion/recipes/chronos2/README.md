# Chronos-2 FP32 maintainer recipe

This recipe converts the pinned `amazon/chronos-2` revision
`29ec3766d36d6f73f0696f85560a422f50e8498c` into a stateless Core AI asset for a
single time series and one 16-step forecast. It is a maintainer workflow for
macOS 27, Python 3.11 and the frozen environment in this directory.

The fixed inputs are `model.json`, `fixtures.json`, `pyproject.toml`, `uv.lock`
and `LICENSE`. Source identity, FP32 precision, quantile order and tolerances
must stay fixed. Other precision profiles are rejected. There is no model
download, quantization, retraining or general forecasting pipeline here.

## Run

Provide a local snapshot directory whose name is the complete revision above.
Its config, README and safetensors must match the sizes and hashes in
`model.json`. Setup uses `uv sync --frozen` in a dedicated
`<workspace>/chronos2/.venv`, verifies imports and direct package versions, and
records source, lock, Python and Apple toolchain identities. Re-run setup when
those identities change. Use the same machine-wide heavy-job lock as every
other model job, including when overriding it with `--lock-dir`.

```sh
python3 -m Tools.Conversion.local setup --recipe chronos2 \
  --workspace .local/conversion \
  --snapshot /path/to/29ec3766d36d6f73f0696f85560a422f50e8498c
python3 -m Tools.Conversion.local plan --recipe chronos2 --workspace .local/conversion
python3 -m Tools.Conversion.local run --recipe chronos2 --precision fp32 --workspace .local/conversion
```

Omitting `--precision` also selects FP32. Local bindings are in
`<workspace>/chronos2.paths.json`; absolute paths stay in the ignored workspace.
Setup refuses symlink workspace/profile directories and stamp/binding files.
The interpreter symlink created by a standard uv virtual environment is
supported; its resolved binary is hashed. Setup never changes global settings.

All five stages use the existing orchestrator's shared heavy-job lock, including
source hashing. `prepare` always runs. Cache keys cover the snapshot, recipe
inputs, adapter/framework code, setup stamp, Python environment and dependency
artifacts. `--refresh` creates new stage outputs; assets and bundles are never
overwritten. Invoke the stages through the orchestrator so the lock is held.

| Stage | Result |
| --- | --- |
| `prepare` | Checks source hashes, config/ABI and setup identity; writes `ready.json`. |
| `reference` | Independently loads the unmodified official CPU FP32 model offline in eval/no-grad; writes `reference.json`. |
| `export` | Compares the export wrapper against all references, then exports `Model.aimodel` in RELEASE mode; saves wrapper vectors, comparison and asset inventory. |
| `verify` | Inspects ABI/storage/compute, loads the asset with Python Core AI GPU preference, and independently compares all native outputs; saves actuals and comparison. |
| `bundle` | Recomputes both comparisons, checks asset identity and exact coverage, and publishes a new bundle plus external `inventory.json`. |

## Numerical and ABI contract

Raw contexts have 1 through 8,192 values, at least one finite observation, and
no infinity or Float32 overflow. JSON `null` means missing; it is explicitly
mapped to NaN before `prepare_context` pads on the left to a multiple of 16.
There is no interpolation or scale change at this boundary. The exported
function is `main(context: Float32[1, length]) -> quantiles: Float32[1, 21, 16]`,
with no state. Length is 16 through 8,192 in steps of 16. Consumers must enforce
these limits: the native ABI's dynamic dimension alone does not encode them.

The quantile axis follows the exact 21 levels in `model.json`. Prediction
values are retained in that order without sorting or clipping. All 18 fixed
cases and 6,048 scalar outputs must satisfy
`abs(actual - reference) <= 1e-4 + 1e-4 * abs(reference)`.
Missing, duplicate, extra or reordered case IDs, nonfinite values, malformed
shapes and unexpected dtypes fail verification. A boolean pass report without
the full vectors is insufficient for packaging.

The conversion preserves the official normalization formulas and inverse.
Only the zero-scale epsilon is materialized with `torch.full_like` to preserve
FP32. A private copy of the Core AI decomposition table replaces `isnan` with
self-inequality and the supported nonoverlapping last-axis unfold with reshape.
Other unfold contracts fail. The dynamic export length is
`16 * Dim('patches', min=1, max=512)` with a 1,008-value example; a 16-value
example triggers the tested Torch derived-dimension boundary failure.

`reference.json` is the downstream Swift verification contract: versioned
format, source/model/fixture hashes, ABI, precision, quantile levels, fixed
tolerances and ordered cases containing raw nullable contexts and full nested
quantile vectors. `actuals.json` has the same vector layout. The wrapper has a
separate vector document. Comparison records include per-case metrics and
the observed case/element counts. Finite tolerance failures retain actuals
and a failed comparison; invalid native arrays are rejected before JSON output.

## Bundle and verification boundaries

`bundle/chronos2.json` uses `format = coreai-samples-chronos2/1`. Its strict
relative `files` inventory contains bytes and SHA-256 for every other file:
the complete asset tree, reference contract, `SOURCE_README.md`, `LICENSE`,
`NOTICE` and sanitized `verification.json`. The external `inventory.json`
also includes `chronos2.json` for later Models catalog pinning. Unknown files,
symlinks, changed assets and incomplete comparisons fail packaging. Failed
copy/validation leaves no published bundle.

Raw `inspection-raw.json` and orchestrator logs remain private: native inspection
can contain local paths. Only explicitly selected ABI, precision, attribution,
hash and comparison fields enter the bundle. Model weights and generated
artifacts remain outside Git. Public release still requires the repository's
model release audit.

CPU tests use synthetic tensors, files and fake runtime outputs:

```sh
python -m unittest discover -s Tools/Conversion/tests -p 'test_chronos2_*.py' -v
```

Run them in the pinned environment; NumPy/Torch checks must not be treated as
passed if dependencies are missing. CPU tests do not establish native accuracy.
The synthetic exported-program test runs its 16-value boundary with Torch's
auxiliary guards disabled while retaining the range/dimension hooks; Torch
specializes the derived one-patch dimension in that synthetic graph. Native
16-value acceptance must still pass the real recipe's unchanged fixture.
The real recipe must complete under the shared lock, and its evidence must be
checked independently of the operator's exploratory probe. Swift integration,
iPhone behavior, application UI and production readiness are separate checks.

## Attribution

Chronos-2 and chronos-forecasting are by Amazon.com, Inc. or its affiliates,
licensed under Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE), the
[official model](https://huggingface.co/amazon/chronos-2) and
[source implementation](https://github.com/amazon-science/chronos-forecasting).
The adapted normalization code and export changes are identified in NOTICE.
