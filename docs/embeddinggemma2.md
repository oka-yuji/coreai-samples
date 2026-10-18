# EmbeddingGemma 2 local image search

Search selected images in Japanese or English, then open the original image. The demo includes five sample images, a local model folder picker, image selection, indexing progress, search results, and Stop. Text and image preprocessing run in Swift.

Use `main` for the demo. Tracking: [runtime #112](https://github.com/oka-yuji/coreai-samples/issues/112), [demo #113](https://github.com/oka-yuji/coreai-samples/issues/113).

## Run the demo

1. Prepare the bundle below, or use the locally prepared bundle from the validation work.
2. Build and run `CoreAISamples` in Xcode on macOS 27. Select **EmbeddingGemma 2** in the sidebar.
3. Choose **Open model → Open model folder…** and select the bundle directory.
4. Choose **Use sample images** or **Choose images…**. Wait for indexing to finish.
5. Describe an image and select **Search**. Select a result to open its original.

You can preselect the demo and bundle at launch:

```sh
CoreAISamples.app/Contents/MacOS/CoreAISamples --image-search --embedding-bundle /path/to/bundle
```

For iPhone, use the `CoreAISamples-iOS` scheme and your local signing configuration. The device must run iOS 27 or later, have Developer Mode enabled, and be unlocked for installation. The simulator cannot run Core AI. Copy the bundle to the app's Documents directory as `embeddinggemma2`; the model menu then offers **Reload selected model**. File Sharing is enabled so the folder can also be transferred through Finder. Model detection does not automatically load or execute it.

The prepared development app and bundle have been installed on iPhone 17 Pro / iOS 27.0.1. In the Demos menu, choose **Image search** under **EmbeddingGemma 2**, then **Open model → Reload selected model → Use sample images**. The same model section also offers **Find objects with the camera**. Choose an **Example search**, then **Search**. The app does not include model weights in Git or download them automatically.

The runtime accepts a local text/vision bundle, raw text, and opaque images decoded by ImageIO. It uses the fixed S128 / vision70 conversion contract. Audio is not included. macOS 27 and iOS 27 are required; Core AI is unavailable in the iOS simulator.

## Local bundle

The locally prepared bundle is at `.local/conversion/bundles/embeddinggemma2` in the sample repository. The app accepts a verified bundle at any chosen location. It contains safe FP16 text/vision assets, the original tokenizer, an exact BF16 vocabulary table, the unmodified model card, license notices, and SHA256 checksums. The pinned source model card identifies Apache-2.0. This work does not publish weights.

Use the [EmbeddingGemma 2 conversion recipe](../Tools/Conversion/recipes/embeddinggemma2/README.md) to reproduce the bundle in an independent environment. Download the pinned source as described there, then run from this repository:

```sh
python3 -m Tools.Conversion.local setup --recipe embeddinggemma2
python3 -m Tools.Conversion.local plan --recipe embeddinggemma2
python3 -m Tools.Conversion.local run --recipe embeddinggemma2
```

Require a completed report with passed verification, then select the bundle stage's artifact directory. The convenience copy above is not automatically replaced by later runs. CoreLLM and the old `package_bundle.py ASSETS OUTPUT` experiment interface are retired; new environments, inputs and reports belong to this repository's ignored `.local` workspace.

## Search examples and limitations

Use **Example searches** to fill the query, then press **Search**. For the sample gallery, try `袋に入った細切りチーズと牛乳`, `肉と玉ねぎの近くに人参ときゅうりがある`, or `駅に停まっている青い列車`. The last query has no relevant sample image. Selecting an example does not automatically run a search, and switching the interface language keeps the query unchanged.

The default view hides scores below 0.67. **Show all candidates** reveals the complete ranking, including weak matches. This threshold was chosen on six calibration examples and frozen before six separate evaluation queries. The held-out result is **5/6**, including three correctly empty results. `トマトと白いチーズでカプレーゼを作れそうな写真` ranks the correct image first but scores about 0.660, so the default view hides it. The threshold and failed example are retained. The five refrigerator images do not establish a universal relevance threshold; similarity is not a probability.

You can choose up to 50 images. The index is held in memory and discarded when the model or selection changes or the screen closes. This is image retrieval, not OCR, image description generation, or an audio/video demo.

## Runtime contract

- Queries use `task: search result | query: `; documents use `title: none | text: `.
- BOS/EOS and the prefix count toward 128 tokens. Longer text is rejected.
- Images respect EXIF orientation, use RGB and antialiased separable bicubic interpolation, then form 16×16 patches and spatial 3×3 pooling groups. Transparent and animated images are rejected. Decoded images are limited to 40 million pixels.
- The converted `fp16-safe-s128-v70` and experimental `int4-int8-ptq-s128-v70` profiles are accepted. File hashes, paths, tensor shapes, scalar types, and vocabulary table layout are checked before inference.
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

The app's Store → runner → service → Core AI path passed five-image indexing, the original three strict retrieval cases, reading the original, cancellation before dispatch, and retry. Six additional query results are reported separately as search-quality evidence; a numerical/interaction pass does not mean all six matched. A separate controlled CPU test covers cancellation while a service is still running. Cancellation during an actual GPU invocation has not been measured. Japanese gallery rendering was checked at 1040×760 and 390×844 points using an AppKit host. The original snapshots were rendered view checks. The October 16 follow-up also exercised the Mac model picker, search, original-image sheet, weak-result toggle, examples, and language switch through the real UI. iPhone touch acceptance remains for the user on main.

The current evidence and acceptance mapping are in [the device and search report](validation/embeddinggemma2-ready-20261016.md); [the original runtime report](validation/embeddinggemma2-native-20261015.md) preserves the initial measurement conditions.

The application integration check uses a new output directory and the original FP32 reference fixtures:

```sh
CoreAISamples.app/Contents/MacOS/CoreAISamples --image-search-eval /path/to/new-report --embedding-bundle /path/to/bundle
```

Run it under the shared heavy-job lock. On an unlocked iPhone, `Tools/Embedding/verify_device.py` installs the signed app, copies the bundle, launches the same application check, and retrieves the results. It uses a unique remote run directory to prevent an old success report from being reused.


## Compare FP16 with INT4 / INT8 on iPhone

The [PTQ recipe](../Tools/Conversion/recipes/embeddinggemma2/README.md#experimental-text-int4--vision-int8) produces a second bundle. Keep FP16 at `Documents/embeddinggemma2` and PTQ at `Documents/embeddinggemma2-int4-int8`. On the camera screen, stop the search, expand **Search settings**, select a folder and start again with the same text and scene. For image search, select either installed folder from **Open model**, then re-index the same images. Selection invalidates the previous model/index. FP16 stays the launch default.

PTQ uses text/vocabulary INT4 and vision INT8; it is not the official QAT checkpoint. The [validation record](validation/embeddinggemma2-int4-int8-20261018.md) reports the smaller bundle and observed quality changes. Bundle size is not RAM usage, and matching the quantized reference is not evidence of unchanged retrieval quality. Keep the same threshold, images, text and camera settings when comparing.

For automated checks, the PTQ bundle's hash-checked `quantized-reference.json` supplies matched numeric references while the app retains the same raw text and image fixtures. The original-reference quality differences remain in the bundled quality report. The saved five-image retrieval results use the original unchanged threshold and expected examples. The iOS self-test can select PTQ with `--image-search-eval RUN --embedding-profile int4_int8`; ordinary launch remains FP16.
