# Object finder for iPhone

Enter an object name, start the camera, and look for matching regions. Japanese examples include eyeglasses, keys, a wallet and a mug. The demo reuses the local EmbeddingGemma 2 text/vision bundle. Camera images stay in memory and are not saved or sent to a server.

Tracking: [image regions #147](https://github.com/oka-yuji/coreai-samples/issues/147), [camera #149](https://github.com/oka-yuji/coreai-samples/issues/149), [live preview and typed targets #153](https://github.com/oka-yuji/coreai-samples/issues/153), [acceptance #150](https://github.com/oka-yuji/coreai-samples/issues/150).

## Run on iPhone

1. Use an iPhone running iOS 27 or later and the `CoreAISamples-iOS` scheme with local signing configured.
2. Prepare the [EmbeddingGemma 2 bundle](embeddinggemma2.md) and copy it to the app's `Documents/embeddinggemma2` directory. A bundle already used by Image search can be reused. Alternatively choose its folder in Search settings.
3. Open **Demos → Object finder**, type a target in **Object to find**, and select **Start searching**. **Fill an example** optionally fills the same editable field.
4. Grant camera permission. A native camera preview starts before the model finishes preparing. Capture is configured for 30 fps; preview display does not wait for model inference. Sustained frame rate on the user's device remains a manual check.
5. Matching candidates receive boxes that Vision tracks between model results. A stable candidate produces one short haptic. **Stop** releases the camera and model before another search can start.
6. To change the target during a search, edit the field and tap **Search this text**, or use the keyboard's search key. Edits remain a draft until applied. The old operation and camera stop before the new target starts; a successfully prepared model is reused for the same folder. **Cancel change** cancels a pending restart and releases the model.

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
- Capture configuration, start, stop, tracking and delegate handling use one serial executor. Every capture start/stop invalidates old callbacks and pending image requests. Capture retains only the latest camera input. The analysis service requests one image after preparation or its previous analysis finishes; no analysis image is created without that request. Source-frame timestamps are gated by the existing 0.15-second interval; a camera timestamp restart is allowed. After session configuration, capture selects a supported native 8-bit YUV format when recognized, or falls back to BGRA for other pixel subtypes. Analysis images are rendered immediately as RGBA8/sRGB. Frame durations are then set to 1/30 second; unsupported configurations fail explicitly.
- Tracking receives only valid candidates meeting the active search threshold. Candidate IDs, scores and order are preserved; overlap suppression still uses the tracked positions. An empty candidate set clears the tracker and boxes.
- The native preview layer and box drawing belong to the MainActor. Preview orientation follows the current UI pose immediately, independently of an in-flight model call. Boxes require the Feature's expected capture generation and expire even if capture callbacks stop. A one-shot timer is scheduled only for visible candidates with a deadline; stopped and empty previews do not poll for expiry. No JPEG encoding or decoding is used for preview display.
- Editing the target changes only the draft. Applying it requests a stop and waits for the old operation to finish. The finder reuses its prepared model for the same folder and encodes the new text. Model-operation failure or cancellation discards retention; a different folder or a completed close requires another load. Updating model files in place during a search is not a supported reload mechanism.
- The service stops the camera on every search exit. Only a target-replacement stop may retain a successfully prepared model until the next search. Stop, backgrounding, dismissal and cancelling a replacement revoke retention and wait for model cleanup. Cleanup failure has a separate stop-failure action so an old search failure cannot unlock a pending stop. The runner suppresses cancelled control acknowledgments, and stops carry a session so old requests cannot stop a newer search.

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

Implementation review and build evidence are in [the original validation record](validation/object-finder-20261017.md), [the live-preview follow-up](validation/object-finder-live-20261017.md), [the image-work optimization](validation/object-finder-performance-20261017.md), [expiry scheduling](validation/object-finder-expiry-20261017.md), [native pixel-format comparison](validation/object-finder-native-format-20261017.md) and [target-replacement reuse](validation/object-finder-model-reuse-20261017.md). Static model verification and the user's device acceptance are distinct gates.
