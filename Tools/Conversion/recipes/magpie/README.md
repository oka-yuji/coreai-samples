# Magpie TTS v2607 conversion

This maintenance recipe exports a fixed FP32 research profile of NVIDIA's Magpie
TTS Multilingual v2607 and NanoCodec to Core AI. The four components are the text
encoder, main decoder with its output projection, local codebook transformer with
its projection, and audio decoder. The host keeps tokenization, embedding-table
lookups, finite scalar dequantization, sampling, attention-prior updates and EOS
detection in the pinned NeMo implementation.

The fixed profile passed all 39 numerical records and generated three Japanese
utterances through Core AI on 2026-10-17. All eight stages passed reuse validation.
See the [validation record](../../../../docs/validation/magpie-conversion-20261017.md).
This is a macOS Python research runner; no Swift application or iPhone support is
claimed. Subjective listening acceptance remains pending.

## Fixed inputs

`model.json` pins the model and codec revisions, archive sizes and SHA-256 values,
the NeMo source revision, the ByT5 tokenizer revision and the numerical criteria.
The model's archive contains repeated configuration entries; extraction selects
the final occurrence, matching ordinary tar extraction. Archive links, escaping
paths and existing destinations are rejected.

The source models and dependencies retain their own licenses:

- [Magpie model](https://huggingface.co/nvidia/magpie_tts_multilingual_357m),
  NVIDIA Open Model License.
- [NanoCodec](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps),
  NVIDIA Open Model License.
- [NeMo Speech](https://github.com/NVIDIA-NeMo/Speech), Apache-2.0 and its included
  third-party notices. The adapter imports the pinned checkout without rewriting it.
- [ByT5](https://huggingface.co/google/byt5-small), Apache-2.0.
- [pyopenjtalk](https://github.com/r9y9/pyopenjtalk), MIT, with the separate Open
  JTalk and dictionary notices included in its distribution.
- Apple's separately installed Core AI tools retain their original licenses.

Model publication, license packaging for distribution and subjective listening
acceptance remain separate from this conversion experiment.

## Environment and local bindings

Use a dedicated Python 3.11 environment under `.local/conversion/magpie`. Install
the packages with `uv pip install --require-hashes -r requirements.lock`, then
install the clean NeMo checkout at the revision in `model.json` separately as
editable with `uv pip install --no-deps -e <checkout>`.
The lock intentionally excludes the machine-specific editable checkout path.
`peft` is included because this NeMo revision imports speechlm2 through its TTS
module initializer. No CUDA, MPS or Metal reference kernel is required for the
FP32 CPU reference.

Download the two `.nemo` files and the three ByT5 files (`config.json`,
`tokenizer_config.json`, `special_tokens_map.json`) at their pinned revisions.
Initialize pyopenjtalk's Open JTalk dictionary before starting the offline recipe.
Retain its notices. The dictionary directory is an explicit input and its contents
are fingerprinted with every model stage.

Create a git-ignored local bindings JSON with absolute paths:

```json
{
  "python": "/path/to/magpie/.venv/bin/python",
  "model": "/path/to/magpie/magpie_tts_multilingual_357m.nemo",
  "codec": "/path/to/nanocodec/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo",
  "tokenizer": "/path/to/byt5/68377bdc18a2ffec8a0533fef03b1c513a4dd49d",
  "nemo": "/path/to/pinned/nemo-speech",
  "jtalk_dictionary": "/path/to/open_jtalk_dic_utf_8-1.11"
}
```

From the repository root:

```sh
python3 -m Tools.Conversion plan Tools/Conversion/recipes/magpie/recipe.json \
  --bindings .local/conversion/magpie.paths.json
python3 -m Tools.Conversion run Tools/Conversion/recipes/magpie/recipe.json \
  --bindings .local/conversion/magpie.paths.json \
  --work-dir .local/conversion/work/magpie \
  --lock-dir "$HOME/Library/Caches/coreai-samples/heavy-job"
```

The generic runner owns stage locks, process groups, timeouts and completed
manifests. Use the shared machine lock documented in `Tools/Conversion/README.md`.
Do not run the adapters directly while another model job owns that lock.
All model stages are offline. The input archives, editable source and installed
environment are fingerprinted; successful stages can only be reused while their
identities and outputs still match. Full reports and WAV files remain private
under the selected work directory.

## Verification profile

The original NeMo model creates Japanese reference token sequences, intermediate
tensors and three WAV files. The reference disables training-only codec
discriminators/loss setup, and disables KV caching for both decoders. The inference
weights and model forward functions are retained. The wrappers are checked
against those reference tensors before export.

For export, the codec wrapper materializes weight parametrizations on a copy and
expresses its fixed causal padding as integer shape arithmetic. The original
decoder is retained for comparison. This avoids the pinned NeMo revision's legacy
weight-norm removal helper and data-dependent padding expressions during tracing.

Core AI executes the exported stateless functions through the OS runtime with GPU
preferred. Verification compares several decoder steps and local codebook
positions and requires finite output, identical shape and maximum absolute error
divided by the reference's maximum absolute value of at most 0.001. Audio decoded
from the same latent also requires cosine at least 0.999 and RMS error at most
0.001. The actual thresholds are in `model.json`.

The host enforces the exported shape bounds: text and decoder sequences up to
512 positions, local sequences 1–16, and codec sequences 2–512 frames. Decoder and
local inputs contain one conditional/unconditional CFG pair. The source model's
two-frame stacking order, all 16 local projection choices, sampling and EOS rules
are preserved. CPU tests compare the host sampler against NeMo, including forced
and forbidden EOS cases.

The final synthesis stage exercises all four Core AI components using the same
Japanese fixtures and reports its stop reason and timing. Matching intermediate
tensors does not establish Japanese naturalness. Free-running samples need not
be bit-identical when floating-point changes affect sampling. Timings include the
research Python host and do not describe an optimized Swift runtime. Prepared
assets and the original PyTorch model coexist in this process, so its memory use
is not a minimum deployment requirement.

Run model-free checks in the dedicated environment:

```sh
/path/to/magpie/.venv/bin/python -m unittest discover -s Tools/Conversion/tests -v
```

NumPy or NeMo-dependent skips are not evidence that the corresponding numerical
tests passed. Actual model conversion, fidelity and speech tests require the
shared model-job lock and the downloaded assets.
