# Irodori Speech on iPhone

## Use Irodori from Models

The normal **Core AI Demos** iOS app now includes **Models → Irodori → Speech demo**. Build and run the `CoreAISamples-iOS` scheme from **main**, with the local signing configuration and the `aarch64-apple-ios` Rust target installed. Its build script prepares the native input library. Device generation and listening acceptance are still pending; the host build is not evidence of device support.

1. Prepare the known FP32 folder using [the existing preparation command](runtime.md#prepare-a-local-model-folder). Keep the complete folder accessible in Files on the iPhone. An existing folder under **On My iPhone → Irodori Speech** can be selected without moving or deleting it.
2. Open **Models → Irodori → Speech demo**, choose **Import model**, and select the complete prepared folder. Wait for validation. The app copies it into managed storage; it does not modify the original. The 3.38 GB source and managed copy both need space, plus room for specialization.
3. Enter a short Japanese passage and a whole-number seed, or choose an example. Press **Generate speech**. Merely opening Models, importing, or viewing the speech screen does not generate audio.
4. Use **Play**, **Stop**, and **Save WAV…**. The exported file is 48 kHz mono Float32 WAV. Generation time and audio length are shown on the screen.
5. Wait for a stop to finish before starting again. Leaving the demo, changing models, or moving to the background cancels generation/playback and waits for audio-session deactivation and runtime release. Reopening retains text and seed, but clears old audio and save tickets; generate again to obtain a new result.

The [managed bundle contract](managed-bundles.md) and [validation record](managed-validation.md) distinguish host checks from device results. Only the fixed 14-file inventory is accepted; incomplete copies, changed files, extra files (including `.DS_Store`), and symbolic links are rejected. Do not add files to the prepared folder.

### Acceptance on main

The user requested device confirmation after merging. On iPhone 17 Pro / iOS 27, check a short generation, reading/naturalness, playback, WAV export, cancellation during generation and playback, background/foreground, and a switch to another model followed by return. Record the main commit, device model/identifier, OS build, text and seed, displayed generation time and audio length, memory measurement method and peak, thermal state and power connection. Keep the WAV and record failures as well as successes. The existing SpeechPhone diagnostic below can supply separate measurements, but does not replace normal-Models acceptance. #329 IOS-IR-01/03/05 and #95 M4/M5 remain pending until their evidence is recorded.

## Separate SpeechPhone app

`SpeechPhone` is a separate iOS 27 app, displayed as **Irodori Speech**, with bundle identifier `com.coreai-samples.SpeechPhone`. Its demo screen is titled **Irodori**. It reuses the pinned FP32 speech runtime and can coexist with the Mac demos and the existing Extract iPhone app. Its source is in SpeechPhone, with a dedicated target and scheme alongside the Mac and existing iPhone app.

The integration check on October 16, 2026 passed the signed SpeechPhone Release build, Mac and existing iPhone regression builds, CPU tests, lint and format. The generated app contains both file-sharing keys and the increased-memory-limit entitlement. A fresh device-info read reported iOS 27.0.1 and a connected tunnel. Model execution, audible playback, WAV export on the device and personal listening acceptance remain unverified. The October 13 connection failure is retained in the validation record as historical evidence.

## Build and install

Use Xcode 27, Rust/Cargo and a physical iPhone running iOS 27 or later. The simulator is unsupported. Build the input library and configure your development team in `Config/Local.xcconfig` using the existing template. This local signing file is git-ignored.

```sh
rustup target add aarch64-apple-ios
./Tools/Irodori/build-inputs.sh iphoneos
xcodebuild -project CoreAISamples.xcodeproj -scheme SpeechPhone \
  -configuration Release -destination 'generic/platform=iOS' \
  -derivedDataPath .build/SpeechPhone -allowProvisioningUpdates build
```

In Xcode, choose **SpeechPhone**, then the iPhone, and Run. The new app requests the increased-memory-limit entitlement. The built app and development profile were checked for that entitlement; the available memory must still be measured on the device.

For command-line deployment, replace the device placeholder with the intended iPhone's identifier:

```sh
phone_device=YOUR_DEVICE_IDENTIFIER
xcrun devicectl device install app --device "$phone_device" \
  .build/SpeechPhone/Build/Products/Release-iphoneos/SpeechPhone.app
```

Unlock the iPhone, keep Developer Mode enabled, and connect it to the Mac by USB if wireless discovery is unavailable. If install reports error 4016 or the device reports no tunnel/DDI services, restore the device connection before retrying. Do not replace this with an OS-version override or lower the deployment target: the Core AI APIs require iOS 27.

## Put the model on the phone

Use [the prepared FP32 folder](runtime.md#prepare-a-local-model-folder). The four assets total 3,368,343,098 bytes; the folder including tokenizer and metadata totals **3,375,061,725 bytes (3.38 GB)**. That is storage size, not the app's working-memory requirement. Specialization needs additional storage and memory. No model weights are embedded in the app.

Copy the entire `sampler-irodori-fp32` folder into **Files → On My iPhone → Irodori Speech**, or use:

```sh
xcrun devicectl device copy to --device "$phone_device" \
  --domain-type appDataContainer --domain-identifier com.coreai-samples.SpeechPhone \
  --source /path/to/sampler-irodori-fp32 \
  --destination Documents/sampler-irodori-fp32
```

Let the copy finish before opening the model. Do not use `--remove-existing-content true`. `SpeechChecks` is reserved for diagnostic output; give the model a different folder name. The app lists visible local folders containing `irodori.json`, then verifies all asset sizes/digests and the pinned tokenizer before inference. An incomplete or changed copy is rejected.

## Use the app

1. Launch Irodori Speech normally and tap **Refresh model folders**.
2. Choose the prepared folder. Wait for file verification to complete.
3. Enter a short passage or choose an example; seed `1708` is the default. Input above 255 body tokens is rejected.
4. Tap **Generate speech**, then **Play** or **Save WAV…**. The result shows audio duration, generation duration and normalized text. The WAV is 48 kHz mono Float32.
5. **Stop** waits for resource cleanup before allowing another operation. Leaving the foreground also ends the operation. Audio-session interruption returns playback to idle without discarding the completed WAV.

Keep the app in the foreground during model preparation and generation. Editing text or seed invalidates the previous audio. First specialization can take longer than later runs. Reading accuracy and naturalness still require listening; this app preserves the upstream normalization and uses no reference voice, empty caption, four steps, no trim and no watermark.

## Explicit device diagnostic

Normal launches perform no inference. The following opt-in launch generates greeting, announcement and greeting again with seed 1708, saves each WAV, plays the first to completion, explicitly stops/releases resources, and writes a report. It keeps the screen awake for the check and restores the previous setting afterward. Use a fresh run ID each time; existing outputs are never overwritten.

```sh
xcrun devicectl device process launch --device "$phone_device" --terminate-existing \
  com.coreai-samples.SpeechPhone \
  --speech-check-run check-001 --speech-model-folder sampler-irodori-fp32
```

The screen reports completion or failure. Results are under `Documents/SpeechChecks/check-001` in the app. `status.json` records the most recent stage, including an interrupted/failed attempt; `report.json` is written only after a successful complete run. Retrieve the directory after completion:

```sh
xcrun devicectl device copy from --device "$phone_device" \
  --domain-type appDataContainer --domain-identifier com.coreai-samples.SpeechPhone \
  --source Documents/SpeechChecks/check-001 --destination /tmp/irodori-phone-check-001
```

The report records OS/build, hardware, Core AI architecture, thermal/power state, per-sample generation time, waveform size and process memory. `availableBytes` is the runtime process allowance, `footprintBytes` is the current physical footprint, and `peakFootprintBytes` is the process-lifetime peak, not a separately reset peak for each utterance. This three-call check is not an endurance or production benchmark. The first two texts correspond to the earlier lab CLI's `a00-s1708` and `a01-s1708`; the third repeats the first. Actual phone output comparison remains pending.

The diagnostic plays to completion, then calls stop/release. Stopping partway through audible playback, real system interruption, save-panel interaction and subjective listening remain separate manual checks. To return to the normal screen, launch the app without the diagnostic arguments.

## Acceptance still required on the device

- [ ] Install on the updated iPhone 17 Pro and verify the visible model/text/seed controls.
- [ ] Copy the model, complete file validation, generate and listen to the samples; inspect the saved WAVs and device report.
- [ ] Play, stop partway, replay and export a WAV through the actual save panel.
- [ ] Stop generation and switch to the background; after cleanup, return and generate again without an old result appearing.
- [ ] Confirm pronunciation, naturalness and usability personally. No automated result substitutes for this check.

See [implementation validation](iphone-validation.md) for the completed CPU/build/review evidence and the outstanding device connection.
