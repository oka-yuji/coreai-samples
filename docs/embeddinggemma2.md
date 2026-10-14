# EmbeddingGemma 2 local image search

Search selected images in Japanese or English, then open the original image. The demo includes five sample images, a local model folder picker, image selection, indexing progress, search results, and Stop. Text and image preprocessing run in Swift.

The complete implementation is on `feat/113-embeddinggemma2-gallery`, based on `feat/112-embeddinggemma2-runtime`. Tracking: [runtime #112](https://github.com/oka-yuji/coreai-samples/issues/112), [demo #113](https://github.com/oka-yuji/coreai-samples/issues/113).

## Run the demo

1. Prepare the bundle below, or use the locally prepared bundle from the validation work.
2. Build and run `CoreAISamples` in Xcode on macOS 27. Select **Image search** in the sidebar.
3. Choose **Open model → Open model folder…** and select the bundle directory.
4. Choose **Use sample images** or **Choose images…**. Wait for indexing to finish.
5. Describe an image and select **Search**. Select a result to open its original.

You can preselect the demo and bundle at launch:

```sh
CoreAISamples.app/Contents/MacOS/CoreAISamples --image-search --embedding-bundle /path/to/bundle
```

For iPhone, use the `CoreAISamples-iOS` scheme and your local signing configuration. The device must run iOS 27 or later, have Developer Mode enabled, and be unlocked for installation. The simulator cannot run Core AI. Copy the bundle to the app's Documents directory as `embeddinggemma2`; the model menu then offers **Reload selected model**. File Sharing is enabled so the folder can also be transferred through Finder. Model detection does not automatically load or execute it.

The iPhone 17 Pro connected during this work runs iOS 27.0.1. A signed Release build succeeded, but installation was blocked by the device lock (`CoreDevice error 10003`), so no iPhone inference result is claimed.

The runtime accepts a local text/vision bundle, raw text, and opaque images decoded by ImageIO. It uses the same S128 / vision70 contract as the CoreLLM #27 conversion. Audio is not included. macOS 27 and iOS 27 are required; Core AI is unavailable in the iOS simulator.

## Prepare the local bundle

Use the Python environment from the conversion experiment, with NumPy installed:

```sh
python Tools/Embedding/package_bundle.py /path/to/embeddinggemma2-20261014 /path/to/new-bundle
```

The output contains the two safe FP16 assets, the original tokenizer, an exact BF16 representation of the unscaled vocabulary embeddings, the original model card, license references, and SHA256 checksums. The original assets are unchanged. Accept the source model's Gemma terms before acquiring or distributing weights.

## Runtime contract

- Queries use `task: search result | query: `; documents use `title: none | text: `.
- BOS/EOS and the prefix count toward 128 tokens. Longer text is rejected.
- Images respect EXIF orientation, use RGB and antialiased separable bicubic interpolation, then form 16×16 patches and spatial 3×3 pooling groups. Transparent and animated images are rejected. Decoded images are limited to 40 million pixels.
- Only the converted `fp16-safe-s128-v70` signature is accepted. File hashes, paths, tensor shapes, and scalar types are checked before inference.
- `EmbeddingModel` owns its models and vocabulary table in one actor. It rejects overlapping load/encode/unload operations. Cancellation does not allow replacement while an awaited runtime call is still returning.
- Embeddings have 768 finite components and must be within 0.001 of unit norm. Search uses cosine similarity, not probability.

## Validation

```sh
swift test --package-path CoreAISamplesKit
EG2_TOKENIZER=/path/to/bundle swift test --package-path CoreAISamplesKit -c release --filter Embedding
```

CPU tests do not load models. `EG2_TOKENIZER` enables the pinned twelve-text token fixture and length-limit check.

`embedding-verify BUNDLE PLAN OUTPUT --preprocess` checks token IDs and records pixel differences without loading models. Remove `--preprocess` for native text/image inference. The plan contains raw text/image paths and the original FP32 reference vectors. The low-precision gate is cosine ≥ 0.999, maximum absolute difference ≤ 0.01, finite values, and unit norm within 0.001. This fixed set is not a general retrieval-quality benchmark.

Model validation must run under the repository's shared heavy-job lock. `Tools/Embedding/run_validation.py` waits for other model processes, acquires the lock, runs the supplied validation command, and releases the lock when it exits.

All twelve native text token sequences match and all eight image token counts match. Pixel differences are at most 1/255 for the five recipe images and zero for the three synthetic gradients. Native Swift preprocessing plus safe FP16 inference passed all twenty reference inputs on the M4 Max: minimum cosine 0.999483553, maximum absolute difference 0.004896015, maximum norm difference 0.000413965.

The app's Store → runner → service → Core AI path passed five-image indexing, three searches, reading the original, cancellation before dispatch, and retry. A separate controlled CPU test covers cancellation while a service is still running. Cancellation during an actual GPU invocation has not been measured. Japanese gallery rendering was checked at 1040×760 and 390×844 points using an AppKit host. These are rendered view snapshots, not automated mouse or touch tests; the computer-use connection was unavailable.

The evidence and acceptance mapping are in [the validation report](validation/embeddinggemma2-native-20261015.md).

The application integration check uses a new output directory and the original FP32 reference fixtures:

```sh
CoreAISamples.app/Contents/MacOS/CoreAISamples --image-search-eval /path/to/new-report --embedding-bundle /path/to/bundle
```

Run it under the shared heavy-job lock. On an unlocked iPhone, `Tools/Embedding/verify_device.py` installs the signed app, copies the bundle, launches the same application check, and retrieves the results. It uses a unique remote run directory to prevent an old success report from being reused.
