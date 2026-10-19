# Qwen3-TTS 0.6B CustomVoice conversion

This maintenance recipe converts four FP32 neural components for macOS Core AI:
text projection, talker with codec head, code predictor with its fifteen heads,
and the 12 Hz audio decoder. The pinned Qwen Python host retains the text tokenizer,
prompt preparation and embedding-table lookups. The Core AI host implements the
two nested generation loops with explicit full prefixes and no KV cache.

The [2026-10-17 validation](../../../../docs/validation/qwen3-tts-conversion-20261017.md)
passed 52 original-output comparisons and generated all three Japanese fixtures
to EOS. Listening remains pending. This is a Python research runner;
Swift integration, iPhone support, quantization and distribution are separate work.

## Fixed inputs and environment

`model.json` records the exact HF revision and all thirteen downloaded file sizes
and SHA-256 values. The initial download was checked against HF Git blob or LFS
identities. Every model stage checks those bytes again. Main model and speech
tokenizer state dictionaries are also loaded strictly after source initialization.

Use the clean Qwen3-TTS checkout at the revision recorded in `model.json`, a
dedicated Python 3.11 environment and `requirements.lock`. Install the checkout
with `uv pip install --no-deps -e <checkout>` after the lock's packages. The lock
does not contain a machine-specific editable checkout path. Qwen uses its required
Transformers 4.57.3; this environment is separate from the Magpie environment.

The [Qwen model](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) and
[source implementation](https://github.com/QwenLM/Qwen3-TTS) use Apache-2.0.
Separately installed Apple conversion tools retain their original license terms.
The original pretrained voices are used without adding voice cloning.

Download the whole model snapshot at the fixed revision and create a local,
git-ignored bindings JSON:

```json
{
  "python": "/path/to/qwen3_tts/.venv/bin/python",
  "snapshot": "/path/to/85e237c12c027371202489a0ec509ded67b5e4b5",
  "upstream": "/path/to/pinned/qwen3-tts"
}
```

From the repository root:

```sh
/path/to/qwen3_tts/.venv/bin/python -m Tools.Conversion plan Tools/Conversion/recipes/qwen3_tts/recipe.json \
  --bindings .local/conversion/qwen3_tts.paths.json
/path/to/qwen3_tts/.venv/bin/python -m Tools.Conversion run Tools/Conversion/recipes/qwen3_tts/recipe.json \
  --bindings .local/conversion/qwen3_tts.paths.json \
  --work-dir .local/conversion/work/qwen3_tts \
  --lock-dir "$HOME/Library/Caches/coreai-samples/heavy-job"
```

The shared lock, process group, timeout, failed-output handling and fingerprinted
reuse are owned by the generic runner. All model stages run offline. Existing
output directories are rejected; inputs and successful earlier artifacts are not
overwritten. Raw logs, intermediate tensors and generated audio remain private.

## Numerical and generation checks

The source generates Japanese fixtures with Ono_Anna and captures 52 reference
records, including talker positions 0, 255, 256 and 511. Each wrapper must match the
source before export; Core AI output is then compared with the same records.
All outputs must be finite and have the expected shape. The maximum absolute
error normalized by the reference maximum must be ≤0.001. Audio from the same
codes also requires cosine ≥0.999 and RMS error ≤0.001.

The first source experiment used greedy decoding and repetition penalty 1.0.
All three short utterances reached the 128-token budget. That result is retained;
it is not a conversion failure. The current fixture adds the source's recommended
sampling settings: seed 42, temperature 0.9, top-k 50, top-p 1.0 and repetition
penalty 1.05, with sampling for both generators. The original model ended at EOS
for all three with those settings. The numerical acceptance criteria did not change.

The host uses Transformers' own logits processors and warpers. CPU tests compare
native and adapted generation, including random-number consumption, forced EOS,
the minimum-token rule and token-limit termination. One talker token is completed
by fifteen code-predictor tokens before the next talker call. The final sampled
talker token has no completed frame when the budget expires; that behavior is
preserved from the source.

The runtime accepts one unpadded input, up to 512 talker positions, 2–16 predictor
positions and up to 256 codec frames. The sample generation budget is 128 talker
tokens. Position axes, predictor head order, token ranges and dtypes are checked
before model calls. Core AI functions are stateless and use GPU-preferred OS
specialization. The host right-pads the talker to 512 positions and the predictor
to 16, selecting the last real position inside each causal graph. This bounds the
number of shapes in the repeated generation loop. The profile is recorded in the
source contract, so old variable-shape assets cannot be mixed with this host.
This is not a latency-optimized or streaming runtime.

The earlier variable-shape profile passed all 52 numerical records but grew to
about 60 GiB process RSS during the second utterance, with about 24 GiB system
swap in use. That run was cancelled and retained. The fixed-shape version passed
the same 52 numerical checks and finished all three utterances with peak process
RSS 8,649,900,032 bytes. This is a three-utterance observation, not an endurance
test; the runtime's internal allocation cause has not been established.

The source PyTorch model coexists with the Core AI components to provide the
retained host operations. Process memory is therefore not a minimum deployment
requirement. Numerical fidelity and EOS completion do not replace listening and
Japanese-language quality evaluation.

Run CPU boundary tests in the dedicated environment:

```sh
/path/to/qwen3_tts/.venv/bin/python -m unittest discover -s Tools/Conversion/tests -v
```

Dependency-related skips are not evidence that the numerical checks passed.
