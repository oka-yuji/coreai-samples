# Object finder for iPhone

Enter an object name, start the camera, and look for matching regions. Japanese examples include eyeglasses, keys, a wallet and a mug. The demo reuses the local EmbeddingGemma 2 text/vision bundle. Camera images stay in memory and are not saved or sent to a server.

Tracking: [image regions #147](https://github.com/oka-yuji/coreai-samples/issues/147), [camera #149](https://github.com/oka-yuji/coreai-samples/issues/149), [live preview and typed targets #153](https://github.com/oka-yuji/coreai-samples/issues/153), [acceptance #150](https://github.com/oka-yuji/coreai-samples/issues/150).

## Run on iPhone

1. Use an iPhone running iOS 27 or later and the `CoreAISamples-iOS` scheme with local signing configured.
2. Prepare the [EmbeddingGemma 2 bundle](embeddinggemma2.md) and copy it to the app's `Documents/embeddinggemma2` directory. A bundle already used by Image search can be reused. Alternatively choose its folder in Search settings.
3. Open **Demos → Object finder**, type a target in **Object to find**, and select **Start searching**. **Fill an example** optionally fills the same editable field.
4. Grant camera permission. A native camera preview starts before the model finishes preparing. Capture is configured for 30 fps; preview display does not wait for model inference. Sustained frame rate on the user's device remains a manual check.
5. Matching candidates receive boxes that Vision tracks between model results. A stable candidate produces one short haptic. **Stop** releases the camera and model before another search can start.
6. To change the target during a search, edit the field and tap **Search this text**, or use the keyboard's search key. Edits remain a draft until applied. The old search finishes shutting down before the new target starts. **Cancel change** cancels a pending restart.

The `--object-finder` launch argument opens the screen. `--embedding-bundle /path/to/bundle` selects a local bundle without starting inference. Model weights are not included in the app or downloaded automatically. Core AI cannot run in the iOS simulator.

Camera permission denial shows a Settings link. Leaving the screen or moving the app to the background stops the search. A camera interruption produces an error with an explicit retry. Camera rotation clears old results and restarts capture in the new orientation.

## What the boxes mean

Vision objectness saliency proposes up to three regions. EmbeddingGemma 2 compares each cropped region with the target text. Boxes are candidates, not guaranteed detections. If saliency misses an object, text similarity cannot recover it. Small, obscured or crowded objects are particularly important device checks.

Preview frame rate and model analysis frequency are separate. Vision tracking only moves existing candidates; it does not confirm their identity again. A candidate expires two seconds after its source frame, and tracking loss or invalid geometry clears its box sooner. Rotation, stopping and target changes also clear old boxes. If inference takes longer than this window, boxes can be temporarily absent while the camera stays live.

Candidates below the selected similarity threshold are hidden; an empty result is normal. The threshold is adjustable while stopped. Similarity is not a probability. The initial value remains **0.67 pending this demo's static-photo validation**; the earlier refrigerator-image search calibration does not establish a threshold for object crops and short nouns.

## Contracts

- Images passed to `ObjectRegionFinder` are opaque, orientation-corrected `CGImage` values up to 4096 pixels per side. There is no file-write step in the camera path.
- Regions use normalized `x`, `y`, `width`, `height` with the origin at the top left. Invalid or non-finite coordinates/scores cannot become selected matches. Overlapping candidates are suppressed.
- The finder owns one model and rejects overlapping preparation, inference and close operations. Cancellation does not release that gate while a runtime invocation is still returning.
- The Feature owns target text, session, display and orientation revision. The service receives versioned orientation commands; it does not independently infer orientation history. Late updates cannot override a newer search or rotation.
- Capture configuration, start, stop, tracking and delegate handling use one serial executor. Every capture start/stop invalidates old callbacks. Capture buffers retain only the latest frame for model analysis, sampled no more often than every 0.15 seconds. Frame durations are set to 1/30 second after session configuration; unsupported formats fail explicitly.
- The native preview layer and box drawing belong to the MainActor. Preview orientation follows the current UI pose immediately, independently of an in-flight model call. Boxes require the Feature's expected capture generation and expire even if capture callbacks stop. No JPEG encoding or decoding is used for preview display.
- Editing the target changes only the draft. Applying it cancels the old search and waits for resource cleanup. Stop, backgrounding and dismissal cancel any pending restart.
- The service releases its resources on every search exit. The explicit stop effect waits for that cleanup. The runner must not issue a second stop after an old search returns, because a newer search may then exist.

## Verification

CPU checks do not open a camera or load model weights:

```sh
./Tools/Irodori/build-inputs.sh
swift test --package-path CoreAISamplesKit --filter ObjectFinder
```

Static-photo checking uses the same region inference as the camera, under the shared heavy-job lock:

```sh
swift run --package-path CoreAISamplesKit object-finder-check BUNDLE PLAN.json OUTPUT.json
```

The output must not already exist. A plan contains `minimumSimilarity` and `samples`, each with `id`, `image` and `query`. Output records contain every proposed box and raw similarity, selected candidates, image dimensions and the region-inference duration. Exit success means the check ran; it does not assert retrieval accuracy. Compare the boxes with separately recorded annotations and keep positive and negative results separate.

Implementation review and build evidence are in [the original validation record](validation/object-finder-20261017.md) and [the live-preview follow-up](validation/object-finder-live-20261017.md). Static model verification and the user's device acceptance are distinct gates.
