# EmbeddingGemma 2 local image search

[日本語](model-reports/embeddinggemma2.ja.md) · [Home](../README.md)

Search selected images in Japanese or English, then open the original image. The demo includes five sample images, a local model folder picker, image selection, indexing progress, search results, and Stop. Text and image preprocessing run in Swift.

Use `main` for the demo. Tracking: [runtime #112](https://github.com/oka-yuji/coreai-samples/issues/112), [demo #113](https://github.com/oka-yuji/coreai-samples/issues/113).


<!-- model-report:start -->
<a id="model-report"></a>

## Evaluation summary

[Measurement protocol](benchmarking.md) · [Source record](model-reports/records/embeddinggemma2.json)

Results below retain their original dates and workloads. Different rows may use different runs; they are not a controlled comparison. Legacy records explicitly retain unrecorded conditions. Missing values are not zero or a pass.

### Demo support

| Platform | State | Verified scope / remaining work | Evidence |
|---|---|---|---|
| macos | ✅ | FP16 local-folder image-search UI recorded; this mark does not cover PTQ camera acceptance. | [E1](validation/embeddinggemma2-ready-20261016.md) |
| ios | △ | Older FP16 diagnostic passed. The current Models route acquires and validates the published FP16 and PTQ bundles (self-test on iPhone 17 Pro, no model loaded); demo execution there and PTQ device inference remain pending. | [E2](validation/embeddinggemma2-ready-20261016.md) · [E4](validation/ios-models-integration-20261018.md#device-checks) · [E3](validation/embeddinggemma2-int4-int8-20261018.md) · [E7](validation/ios-models-integration-20261018.md#acquisition) |

### Performance and quality

| Metric | Result | Statistic / samples | Scope and evidence |
|---|---|---|---|
| Bundle / payload size | 913,677,333 B | single; n=1 | [E3](validation/embeddinggemma2-int4-int8-20261018.md) · FP16 bundle; includes tokenizer and notices. |
| Bundle / payload size | 341,880,041 B | single; n=1 | [E3](validation/embeddinggemma2-int4-int8-20261018.md) · PTQ bundle; includes tokenizer and notices. |
| Prepared cache size | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| First load, including specialization | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Load from prepared cache | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Request / pipeline completion time | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Peak RSS | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Peak physical footprint | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Resident memory during inference | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Memory after release minus baseline | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Task success fraction | 0.833333333 ratio | aggregate; n=6 | [E1](validation/embeddinggemma2-ready-20261016.md) · 5/6 held-out threshold decisions. Not Recall@k or nDCG; a correct top image was hidden by the frozen threshold. |
| Task success fraction | 0.833333333 ratio | aggregate; n=6 | [E2](validation/embeddinggemma2-ready-20261016.md) · 5/6 held-out threshold decisions. Not Recall@k or nDCG; a correct top image was hidden by the frozen threshold. |
| Reference error (definition in evidence) | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Stop acknowledgement time | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Index construction time | 0.76569325 s | single; n=1 | [E1](validation/embeddinggemma2-ready-20261016.md) · One five-image indexing request; legacy route, not a controlled Mac/iPhone comparison. |
| Index construction time | 0.895115834 s | single; n=1 | [E2](validation/embeddinggemma2-ready-20261016.md) · One five-image indexing request; legacy route, not a controlled Mac/iPhone comparison. |
| Search query time | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Recall@k | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| nDCG@k | Not recorded here | — | No value in this common record; inspect historical evidence before scheduling new measurement. |
| Reference cosine similarity | 0.999483553 cosine | min; n=20 | [E1](validation/embeddinggemma2-ready-20261016.md) · 20 raw text/image inputs against original FP32. |

### Verification coverage

| Check | State | Scope / remaining work | Evidence |
|---|---|---|---|
| Conversion / reference fidelity | Passed (stated scope) | FP16 and matched-PTQ checks pass only their pinned numerical contracts; PTQ versus original precision is separate. | [E1](validation/embeddinggemma2-ready-20261016.md) · [E3](validation/embeddinggemma2-int4-int8-20261018.md) |
| Task quality | Failed (stated criterion) | Held-out false negative remains; PTQ loses original examples. Thresholds were not lowered to pass. | [E5](validation/embeddinggemma2-ready-20261016.md) · [E6](validation/embeddinggemma2-int4-int8-20261018.md) |
| Repeated use / resource release | Not measured | No completed check for this scope under the common protocol. | — |
| Stop / retry / recovery | Passed (stated scope) | Legacy image indexing stop/queued retry, not current Models lifecycle or camera interruption. | [E1](validation/embeddinggemma2-ready-20261016.md) · [E2](validation/embeddinggemma2-ready-20261016.md) |

### Evidence and conditions

| Evidence | Date / record / route level | Device / OS / tools | Workload and recorded conditions |
|---|---|---|---|
| [E1](validation/embeddinggemma2-ready-20261016.md) | 2026-10-16 / legacy / native_app | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 5 refrigerator images, 6 held-out queries; 768D, threshold 0.67; local-folder image-search UI. · variant=safe FP16 S128 / vision70; backend=Core AI; GPU preferred; route=macOS local-folder image search; source_revision=914f7f89142e33e77833254d9c9b90c3cef7303b; batch_size=1; concurrency=1; unrecorded: artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E2](validation/embeddinggemma2-ready-20261016.md) | 2026-10-16 / legacy / diagnostic | iPhone 17 Pro / iOS 27.0.1 (24A446) / Xcode 27.0 (27A266a) | Older signed app diagnostic: 5 images and 6 held-out queries. This is not the current Models entry. · variant=safe FP16 S128 / vision70; backend=Core AI; GPU preferred; route=diagnostic; batch_size=1; concurrency=1; unrecorded: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E3](validation/embeddinggemma2-int4-int8-20261018.md) | 2026-10-18 / legacy / conversion | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | Matched-PTQ 20-vector comparison; original-precision quality degradation retained; iPhone launch was locked. · variant=INT4 text / INT8 vision PTQ; backend=Core AI; GPU preferred; route=conversion; source_revision=914f7f89142e33e77833254d9c9b90c3cef7303b; artifact_revision=7c37ea823af7a4028ec3757e124b53fe8e16a75c99c46d24ca0a7007cca55b85; code_revision=71fb61afd24e1fc5324f57a1abc02744d07ae09c; batch_size=1; concurrency=1; unrecorded: warmups, power, thermal, plan_sha256 |
| [E4](validation/ios-models-integration-20261018.md#device-checks) | 2026-10-18 / legacy / plan | iPhone 17 Pro / iOS 27.0.1 (24A446) / Xcode 27.0 (27A266a) | See fixed cases in the evidence. · variant=FP16 / PTQ; backend=Core AI; GPU preferred; route=plan; batch_size=1; concurrency=1; unrecorded: source_revision, artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E5](validation/embeddinggemma2-ready-20261016.md) | 2026-10-16 / legacy / diagnostic | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | 5 refrigerator images, 6 held-out queries; 768D, threshold 0.67; local-folder image-search UI. · variant=safe FP16 S128 / vision70; backend=Core AI; GPU preferred; route=held-out retrieval quality evaluation; source_revision=914f7f89142e33e77833254d9c9b90c3cef7303b; batch_size=1; concurrency=1; unrecorded: artifact_revision, code_revision, warmups, power, thermal, plan_sha256 |
| [E6](validation/embeddinggemma2-int4-int8-20261018.md) | 2026-10-18 / legacy / diagnostic | M4 Max 128 GiB / macOS 27.0 (26A428) / Xcode 27.0 (27A266a) | Matched-PTQ 20-vector comparison; original-precision quality degradation retained; iPhone launch was locked. · variant=INT4 text / INT8 vision PTQ; backend=Core AI; GPU preferred; route=held-out retrieval quality evaluation; source_revision=914f7f89142e33e77833254d9c9b90c3cef7303b; artifact_revision=7c37ea823af7a4028ec3757e124b53fe8e16a75c99c46d24ca0a7007cca55b85; code_revision=71fb61afd24e1fc5324f57a1abc02744d07ae09c; batch_size=1; concurrency=1; unrecorded: warmups, power, thermal, plan_sha256 |
| [E7](validation/ios-models-integration-20261018.md#acquisition) | 2026-10-19 / legacy / diagnostic | iPhone 17 Pro / iOS 27.0.1 (24A446) / Xcode 27.0 (27A266a) | The library refresh migrated both variants (13 and 16 listed files; 913,677,333 B and 341,880,041 B) from the existing Documents folders after size and SHA-256 checks, then inspected them; no model loaded. · variant=safe FP16 S128 / vision70; INT4/INT8 PTQ; route=Models self-test (--acquire all); source_revision=914f7f89142e33e77833254d9c9b90c3cef7303b; artifact_revision=daab6dbfbae1610a87d9f181458419e3ea5ce682; unrecorded: backend, code_revision, batch_size, concurrency, warmups, power, thermal, plan_sha256 |

### Setup, limitations and progress

- [Search guide](embeddinggemma2.md)

- [#210](https://github.com/oka-yuji/coreai-samples/issues/210)
<!-- model-report:end -->

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

For iPhone, use the `CoreAISamples-iOS` scheme and your local signing configuration. The device must run iOS 27 or later, have Developer Mode enabled, and be unlocked for installation. The simulator cannot run Core AI. For the diagnostic launch with `--image-search` or `--legacy-demos`, copy the bundle to the app's Documents directory as `embeddinggemma2`; the model menu then offers **Reload selected model**. File Sharing is enabled so the folder can also be transferred through Finder. Model detection does not automatically load or execute it.

The prepared development app and bundle have been installed on iPhone 17 Pro / iOS 27.0.1. On that diagnostic screen, open the Demos menu and choose **Image search** under **EmbeddingGemma 2**, then **Open model → Reload selected model → Use sample images**. The same model section also offers **Find objects with the camera**. Choose an **Example search**, then **Search**. The app does not include model weights in Git or download them automatically.

## Run from Models on iPhone

The normal iPhone launch opens **Models**. **Image search demo** under **EmbeddingGemma 2** runs with the variant selected there; it has no model folder picker or **Open model** menu. **Choose images…** opens the Photos picker, which runs outside the app, so the demo asks for no photo library permission; select up to 50 photos. The chosen photos are copied to the app's temporary folder; choosing again or leaving the demo deletes the copies of photos that are no longer in the demo, while the current photos and their results stay. On the Mac, **Choose images…** still opens a Finder panel. **Use sample images**, **Choose images…** or **Index again** loads the selected bundle when needed and then indexes the images as one request. Showing the description or finishing a download loads nothing. Loading, indexing, searching and opening an original image each check, immediately before they start, that the library still permits the session (selected variant and verified lease); otherwise the demo reports that the model is not available. An image selection, or a failure to open the selection, that returns after its session stopped is ignored.

Going back, moving to the background and switching the variant stop the demo: they wait for indexing or search to end, then release the model and its index. **Stop** inside the demo only cancels the running index or search. The chosen images, the query and the results remain on screen. Results name the artifact that produced them (`model/artifact@revision`); a different variant never reuses the previous index, so **Index again** recomputes it from the retained images before the next search. A demo Store left from an earlier variant can no longer run, even after you return to that variant, and the next stop closes it again. The stop and the demo report the result of the same release attempt. If the release fails, the library reports a failed stop that can be retried; it is not shown as a successful stop or a download failure. Images that can no longer be opened ask you to choose them again. Models downloads the selected variant from [okayuji/embeddinggemma-2-coreai](https://huggingface.co/okayuji/embeddinggemma-2-coreai) (#210): **FP16** (the default) from `fp16/` or **INT4/INT8 PTQ** from `int4-int8/`, each checked file by file before the demo can run. Running this route on a device remains a user check.

The runtime accepts a local text/vision bundle, raw text, and opaque images decoded by ImageIO. It uses the fixed S128 / vision70 conversion contract. Audio is not included. macOS 27 and iOS 27 are required; Core AI is unavailable in the iOS simulator.

## Local bundle

The locally prepared bundle is at `.local/conversion/bundles/embeddinggemma2` in the sample repository. The app accepts a verified bundle at any chosen location. It contains safe FP16 text/vision assets, the original tokenizer, an exact BF16 vocabulary table, the unmodified model card, license notices, and SHA256 checksums. The pinned source model card identifies Apache-2.0. The same FP16 bundle and the PTQ bundle below are published in [okayuji/embeddinggemma-2-coreai](https://huggingface.co/okayuji/embeddinggemma-2-coreai) under `fp16/` and `int4-int8/`; the iPhone's Models downloads them from there.

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

You can choose up to 50 images. The index is held in memory and discarded when the model or selection changes or the screen closes. The selected images, the query and the results stay; search again after **Index again**, which loads the model folder first when it is not loaded. On the Mac, **Reload selected model** also keeps them; opening a different folder clears them. This is image retrieval, not OCR, image description generation, or an audio/video demo.

## Video moment search

Find moments in a video with your own words, then play the video from the moment you select (#129). The demo samples still frames, embeds each frame with the same image encoder and preprocessing as image search, and ranks the frames against the query. Audio, motion between frames and captions are not used.

- **iPhone:** in **Models**, open **Video search demo** under **EmbeddingGemma 2**. **Choose video…** opens the Photos picker, which runs outside the app, so the demo asks for no photo library permission. The chosen video is copied to the app's temporary folder; the next choice replaces the copy. Indexing loads the selected variant when needed. Going back, moving to the background and switching the variant stop the demo and release the model and its index as in image search; the video, the query, the results and the threshold stay, and **Index again** rebuilds the index.
- **Mac:** select **Find moments in a video with your own words** under **EmbeddingGemma 2** in the sidebar, open the model folder, then choose a movie file. You can preselect the demo and bundle at launch: `CoreAISamples.app/Contents/MacOS/CoreAISamples --embedding-bundle /path/to/bundle --video-search`. Keep `--video-search` last: AppKit reads the arguments as option-value pairs, so a flag placed before `--embedding-bundle` leaves the path on its own, AppKit opens it as a document, and SwiftUI then opens no window.

One frame is taken at the center of each one-second interval, up to 300 frames; a longer video is sampled at 300 evenly spaced times. Frames are decoded with the video's orientation applied, at most 640 pixels on the long side, within 0.5 seconds of the requested time, and a result shows the time of the decoded frame. Frames are requested a few at a time so that decoded frames do not pile up while the model is busy. A time whose frame cannot be decoded is skipped, and a video with no decodable frame is rejected. The index is held in memory only.

Results are ranked by similarity. The default view hides frames below 0.67 and folds frames within 3 seconds of a better match into it; **Show all candidates** lists every indexed frame. The 0.67 default comes from the image demo's five-image calibration and is not calibrated for video. While nothing runs, **Search settings → Match threshold** adjusts it between 0.3 and 0.9. Similarity compares meaning; it is not a probability.

Validation: `swift test --package-path CoreAISamplesKit --filter VideoSearchTests` covers sampling, the order of indexing, search, stop and close, neighbor folding and the threshold, the managed gate and release, and frames read from synthetic videos (colors in order, a rotated video, cancellation, a file without a video track) without loading a model. The model check writes the five bundled images, three seconds each, into a fifteen-second video and requires the best frame for the two tomato and white cheese queries of `image-search-queries.json` to fall between 12 and 15 seconds, with neighbors folded. It loads the model, so run it under the shared heavy-job lock:

```sh
COREAI_TEST_BUNDLES=/path/to/embeddinggemma2 swift test --package-path CoreAISamplesKit -c release --filter VideoSearchIntegration
```

It runs only when a path in `COREAI_TEST_BUNDLES` contains `embedding-manifest.json`, so language-model bundles passed to the same variable skip it. Choosing, playing and searching a video on an iPhone remain user checks.

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

CPU tests do not load models. `EG2_TOKENIZER` enables the pinned twelve-text token fixture and length-limit check. The Models route is covered by `ImageSearchTests` with fake services: loading only the selected artifact, closed gates and replaced artifact bindings, late image selections, retention across stop and re-entry, a standalone screen shown again, variant switches without the old index, returning to an earlier variant, late selection failures, release failures and retries, releases that complete at once, releases from cancelled tasks or before any content, cancellation during indexing, invalid images, and expired image URLs. Device acceptance remains a manual check.

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

In Models, choose **FP16** or **INT4/INT8 PTQ** under Model information. Each variant is downloaded and validated on its own, and switching stops the running demo before the other variant is used. The diagnostic screens below use local folders instead. The [PTQ recipe](../Tools/Conversion/recipes/embeddinggemma2/README.md#experimental-text-int4--vision-int8) produces a second bundle. Keep FP16 at `Documents/embeddinggemma2` and PTQ at `Documents/embeddinggemma2-int4-int8`. On the camera screen, stop the search, expand **Search settings**, select a folder and start again with the same text and scene. For image search, select either installed folder from **Open model**, then re-index the same images. Selection invalidates the previous model/index. FP16 stays the launch default.

PTQ uses text/vocabulary INT4 and vision INT8; it is not the official QAT checkpoint. The [validation record](validation/embeddinggemma2-int4-int8-20261018.md) reports the smaller bundle and observed quality changes. Bundle size is not RAM usage, and matching the quantized reference is not evidence of unchanged retrieval quality. Keep the same threshold, images, text and camera settings when comparing.

For automated checks, the PTQ bundle's hash-checked `quantized-reference.json` supplies matched numeric references while the app retains the same raw text and image fixtures. The original-reference quality differences remain in the bundled quality report. The saved five-image retrieval results use the original unchanged threshold and expected examples. The iOS self-test can select PTQ with `--image-search-eval RUN --embedding-profile int4_int8`; ordinary launch remains FP16.
