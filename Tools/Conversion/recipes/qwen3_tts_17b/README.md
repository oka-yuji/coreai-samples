# Qwen3-TTS 1.7B CustomVoice conversion

This recipe uses the shared Qwen3-TTS adapter to convert four FP32 Core AI
components for macOS: text projection, talker/head, code predictor/heads and the
12 Hz audio decoder. The source Python host retains tokenization, prompt assembly
and embedding lookups. The [2026-10-17 validation](../../../../docs/validation/qwen3-tts-17b-conversion-20261017.md)
passed all 52 original-output comparisons and generated all three Japanese
fixtures to EOS. Subjective listening remains pending.

The source snapshot is pinned in `model.json`, including all thirteen file hashes.
The talker and raw predictor inputs have width 2048. Inside the predictor, the
original biased projection maps width 2048 to 1024. The shared adapter checks
model identity and configuration before accepting either 0.6B or 1.7B inputs.
Existing 0.6B contracts and assets remain supported; assets cannot be mixed.

Use the same isolated Python 3.11 environment and pinned Qwen checkout as the
[0.6B recipe](../qwen3_tts/README.md). `requirements.lock` records the dependencies;
install it with `uv pip install --require-hashes -r requirements.lock` and the
checkout separately with `uv pip install --no-deps -e <checkout>`.
The pretrained snapshot and generated outputs use separate directories.

Create a git-ignored bindings file:

```json
{
  "python": "/path/to/qwen3_tts/.venv/bin/python",
  "snapshot": "/path/to/0c0e3051f131929182e2c023b9537f8b1c68adfe",
  "upstream": "/path/to/pinned/qwen3-tts"
}
```

From the repository root:

```sh
/path/to/qwen3_tts/.venv/bin/python -m Tools.Conversion run \
  Tools/Conversion/recipes/qwen3_tts_17b/recipe.json \
  --bindings .local/conversion/qwen3_tts_17b.paths.json \
  --work-dir .local/conversion/work/qwen3_tts_17b \
  --lock-dir "$HOME/Library/Caches/coreai-samples/heavy-job"
```

The runner owns exclusive model execution, cancellation, output integrity and
reuse. Model stages run offline. Existing output directories are rejected.
Original FP32 reference outputs, wrapper checks and actual Core AI results remain
separate evidence. Numerical gates are unchanged from 0.6B: normalized maximum
error ≤0.001, and same-code audio cosine ≥0.999 with RMS error ≤0.001.

The Japanese fixtures use Ono_Anna, seed 42, non-streaming input, sampling in both
decoders, temperature 0.9, top-k 50, top-p 1.0, repetition penalty 1.05 and a
128-token budget. Fixed execution lengths are talker 512 and predictor 16; the
host selects the last real prefix output under causal attention. KV caching is
not used. Model loading and source hashing are outside per-utterance timing.

This is a Mac/Python research profile. The three fixtures do not qualify sustained
memory behavior, Japanese naturalness, Swift/iPhone deployment or quantization.
Source and converted weights retain the [model's Apache-2.0 terms](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).

Run the shared CPU checks in the dedicated environment:

```sh
/path/to/qwen3_tts/.venv/bin/python -m unittest discover -s Tools/Conversion/tests -v
```
