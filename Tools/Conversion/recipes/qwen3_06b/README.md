# Pinned Qwen3-0.6B integration recipe

This recipe demonstrates the complete reference → export → Swift-check path on a real
checkpoint. It keeps reference and export independent so a fixture/criterion change does
not repeat an unchanged export. The reference is the original model in FP32 on the CPU.
The export is unquantized FP16 on macOS using Apple's coreai-models 1.0.0 checkout.

A lightweight prepare stage checks the source and exporter revisions before either
model job. It always runs, including on a cache hit, because Git HEAD can change while
the relevant source bytes remain the same. Its stable result allows the expensive
reference and export stages to remain reusable.
It also records the selected Xcode, macOS SDK and Core AI compiler bytes, so switching
the Apple toolchain invalidates dependent stages even when the Python environment is unchanged.

It is a fixed-input integration check, not the ten-item production qualification.
Three cases cover English, Japanese and a short chat turn. A passing comparison does not
establish long-context quality, Swift tokenizer parity, endurance or deployment on iPhone.
Swift receives the reference's raw token IDs and reports Float16 logits on Apple Silicon.

## Local setup

The maintainer can use `python3 -m Tools.Conversion.local setup`, then `plan` and `run`.
The explicit bindings below describe the underlying interface.

Use an already-downloaded immutable HF snapshot. The model and exporter revisions are
recorded in model.json. Do not update the normal HF cache's refs/main to make the recipe run.
The export adapter creates and removes its own temporary cache and uses the original HF
model ID so Apple's registry and metadata still apply. Original tokenizer files replace
the converter's reserialized tokenizer only inside the new stage's bundle.
The adapter runs the official CLI's --dry-run with the same explicit flags before
starting export; its resolved settings remain in the stage log.

Example .local/conversion/local-paths.json (keep machine paths outside the committed recipe):

~~~json
{
  "upstream": "/path/to/coreai-samples/.local/conversion/upstream/coreai-models",
  "snapshot": "/path/to/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
}
~~~

From the sample repository root:

~~~sh
python3 -m Tools.Conversion plan Tools/Conversion/recipes/qwen3_06b/recipe.json \
  --bindings .local/conversion/local-paths.json
python3 -m Tools.Conversion run Tools/Conversion/recipes/qwen3_06b/recipe.json \
  --bindings .local/conversion/local-paths.json --work-dir .local/conversion/work/qwen3_06b \
  --lock-dir "$HOME/Library/Caches/coreai-samples/heavy-job"
~~~

Use the same run command to resume or reuse. Each report points to immutable stage
directories. Run the adapter tests with the recipe's Python environment (NumPy required):

~~~sh
/path/to/coreai-models/.venv/bin/python -m unittest discover -s Tools/Conversion/tests -v
~~~

Use --set fixture=/absolute/path/to/fixtures.json or --set criteria=/absolute/path/to/criteria.json
for an explicitly versioned alternative. Keep the three case IDs when using this recipe's
declared output list; a different case set needs its own recipe declaration.

The shared heavy-job lock must be available and no unmanaged model job should be running.
All model commands are foreground processes. No global environment, existing bundle or
reference output is overwritten.

## Criteria and extension

criteria.json is fixed before the first model run: maximum probability error 0.03,
zero top-1 mismatches outside a reference probability margin of 0.01. A near-tie must
involve the same top-two token candidates on both sides, and the observed winner's
reference probability must be within that margin of the reference winner. Near-ties are
reported separately. Wrong token IDs, misaligned/absent positions, wrong widths and
non-finite logits fail independently. A failed criterion stops the stage and keeps the
diagnostic comparison; it never becomes a completed verification cache entry.

These are this small integration recipe's criteria. They do not replace any existing
model's publication criteria. Quantized models need a reference with the corresponding
quantization and a separate comparison of quantization degradation; the adapter refuses
to reuse this unquantized reference for them.

For an existing model-specific script, first make all input/output locations explicit and
keep its model-specific math and established criteria. Declare the source snapshot,
imported helpers, fixture and configuration as inputs; write only to the stage output and
scratch directories; emit a structured gate result. A script with fixed historical output
paths or --detach must not be placed into a recipe unchanged.

Add new architecture adapters only after checking their state, tokenizer, shape and
precision contracts against the upstream reference. Shared orchestration does not make
unknown architectures compatible automatically.
