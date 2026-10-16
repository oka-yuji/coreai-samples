# Irodori native runtime

SpeechCore defines the input/output values, frame policy and Float32 WAV encoding. SpeechRuntime owns the Core AI models in an actor and performs normalization, tokenization and generation outside the main actor. Its protocol boundary allows CPU-only lifecycle tests.

The four FP32 assets and tokenizer are fixed by SHA-256. Input uses the pinned Unicode 15 normalization and PyTorch-compatible CPU noise. Requests with more than 255 body tokens are rejected instead of silently truncated. The route is no reference voice, empty caption, four MeanFlow steps, 48 kHz mono, with no trimming or watermark.

Only one generation or model change is accepted at a time. A stop waits for the in-flight operation to return; cancellation is checked between stages. The runtime releases all tensor contexts before reloading its four models, after 60 completed generations or a frame ratio of at least two. It also drops the GPU models on failure or explicit release. A future request recreates them. The app does not run a Python interpreter or a CLI subprocess.

## Build and CPU tests

Rust/Cargo and Xcode 27 are required on an Apple silicon Mac with macOS 27.

```sh
./Tools/Irodori/build-inputs.sh
swift test --package-path CoreAISamplesKit --filter 'Speech|IrodoriRuntimeTests'
python3 Tools/Irodori/test_prepare_bundle.py
```

Do not set the model-test environment variables for this CPU-only check. Tests use temporary files and controlled fake pipelines, plus the frozen 283-case normalization corpus and 49 noise inputs. The model numerical route is adapted from the already verified lab implementation; the app's actual GPU generation and listening remain user acceptance work.

## Prepare a local model folder

Use the four verified FP32 `.aimodel` directories and the matching tokenizer JSON:

```sh
python3 Tools/Irodori/prepare-bundle.py \
  --artifacts /path/to/artifacts \
  --tokenizer /path/to/tokenizer.json \
  --output /path/to/Irodori-FP32
```

The script checks every source digest, uses APFS copy-on-write copies, and publishes the complete folder with an exclusive rename. It refuses any existing output. It does not execute a model. The app opens the resulting folder containing `irodori.json`, `tokenizer.json`, and `artifacts/`.

Input dependency provenance and upstream licenses are in `Vendor/IrodoriInputKit`. Core AI/Swift lifecycle and source changes are reviewed separately from human audio quality.
