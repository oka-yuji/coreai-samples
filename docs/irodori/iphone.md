# Irodori Speech on iPhone

## Use Irodori from Models

Build and run **CoreAISamples-iOS** on a physical iPhone with iOS 27 or later. In **Models → Irodori → Speech demo**, select **Reference voice (FP32)**. The legacy FP32 variant remains available for text-only generation.

### Model preparation

Tap **Download model** and wait for file verification. The app downloads the fixed [Core AI reference release](https://huggingface.co/okayuji/irodori-v4.1-small-mf-coreai-fp32), including six model components, tokenizer, manifest and license notices: **3,835,396,408 bytes (3.84 GB)**. Allow additional space for model preparation and specialization caches. Keep the app open during download and generation.

The reference model can generate with or without a reference voice. The legacy **FP32** variant uses **Import model** and a separately prepared text-only folder. The original PyTorch checkpoint and a renamed legacy folder cannot be used as the reference model.

### Record or choose a voice

1. Download the **Reference voice** model and wait for **1. Model ready**.
2. Under **2. Choose a reference voice**, tap **Record voice**, speak clearly, and tap **Finish recording**. Recording stops automatically after 30 seconds. **Cancel recording** discards that attempt. If microphone permission is denied, use **Open microphone settings** or choose a file instead.
3. Alternatively, tap **Choose audio file** and select a WAV, M4A, MP3 or AIFF file. Use a clear recording from 0.5 to 30 seconds. The app converts supported files to 48 kHz mono; it does not silently truncate a long file. The selected name and duration appear after validation.
4. Use **Listen to reference** to check the input, or **Remove reference** to return to a generated voice. Reference audio is optional. Use your own voice or obtain the voice owner's explicit permission for the intended AI synthesis and use of its outputs. Owning a recording or finding it online does not establish that permission. Do not present generated speech as something that person actually said.
5. Under **3. Enter what to say**, choose **Try an example** or enter a short Japanese passage. The seed is inside **Voice options** and can stay unchanged.
6. Under **4. Generate and listen**, tap **Generate speech** and keep the app open. First specialization takes longer. Then use **Play**, **Stop** and **Save WAV…**.

Opening the screen, importing a model, or choosing an example does not start inference. Leaving the managed demo or moving to the background stops recording, loading, generation and playback before the resource lease is released. Text and seed are retained; reference/output audio from a closed managed session are cleared. Import failure or cancellation keeps an existing reference instead of replacing it with invalid data.

Generation and reference processing run on the device. Reference audio is not uploaded by the app; an exported WAV follows the destination selected by the user. Recorded temporary files are removed after decoding or failure.

### Verification status

The October 29, 2026 preflight on **iPhone 17 Pro / iOS 27.0.1 (24A446)** completed reference-file decoding, three generations, WAV saving and resource release through the shared runtime. Greeting → announcement → greeting produced 4.00 / 4.24 / 4.00-second outputs; the repeated greeting WAVs matched. See [the measured report](reference-input-device.json).

The initial device attempt aborted in the GPU codec decoder with a dynamic-allocation assertion. Disabling frequent-reshape specialization for the iOS decoder resolved that attempt. The Mac policy stays separate. The diagnostic used a synthetic reference file and did not record a microphone or play sound. This diagnostic does not establish normal-screen operation, real microphone capture, audible playback or subjective speaker quality. The earlier Python waveform-fidelity miss is preserved in [the runtime record](reference-runtime.md).

### Explicit reference diagnostic in the normal app

For a prepared folder and a reference file directly in the app's Documents directory, the normal iOS app accepts these development launch arguments:

```text
--speech-reference-check reference-check-01
--speech-model-folder Irodori-Reference
--reference-audio reference.wav
--exit-after-speech-check
```

Use a new check ID each time. Results are written to `Documents/SpeechChecks/<ID>/` without overwriting prior runs. The diagnostic generates and saves three examples, records memory and repeat equality, and skips playback and microphone capture. A normal launch without the flag opens Models and performs no generation. The separate SpeechPhone product uses `--speech-check-run <ID>` with the same folder/reference flags and `--no-playback`.

## Separate SpeechPhone app

`SpeechPhone` is a separate iOS 27 app, displayed as **Irodori Speech**, with bundle identifier `com.coreai-samples.SpeechPhone`. Its demo screen is titled **Irodori**. It reuses the pinned FP32 speech runtime and can coexist with the Mac demos and the existing Extract iPhone app. Its source is in SpeechPhone, with a dedicated target and scheme alongside the Mac and existing iPhone app.

The historical integration check on October 16, 2026 passed the signed SpeechPhone Release build, Mac and existing iPhone regression builds, CPU tests, lint and format. The generated app contains both file-sharing keys and the increased-memory-limit entitlement. That check did not execute a model. The October 29 reference preflight above adds model execution and WAV saving; audible playback and subjective listening quality were not evaluated by that diagnostic. The October 13 connection failure is retained as historical evidence.

## Build and install

Use Xcode 27, Rust/Cargo and a physical iPhone running iOS 27 or later. The simulator is unsupported. Build the input library and configure your development team in `Config/Local.xcconfig` using the existing template. This local signing file is git-ignored.

```sh
rustup target add aarch64-apple-ios
./Tools/Irodori/build-inputs.sh iphoneos
xcodebuild -project CoreAISamples.xcodeproj -scheme SpeechPhone \
  -configuration Release -destination 'generic/platform=iOS' \
  -derivedDataPath .build/SpeechPhone -allowProvisioningUpdates build
```

For the normal Models download flow, choose **CoreAISamples-iOS** in Xcode, then the iPhone, and Run. Choose **SpeechPhone** only for the separate local-folder demo described below. The new app requests the increased-memory-limit entitlement. The built app and development profile were checked for that entitlement; the available memory must still be measured on the device.

For command-line deployment, replace the device placeholder with the intended iPhone's identifier:

```sh
phone_device=YOUR_DEVICE_IDENTIFIER
xcrun devicectl device install app --device "$phone_device" \
  .build/SpeechPhone/Build/Products/Release-iphoneos/SpeechPhone.app
```

Unlock the iPhone, keep Developer Mode enabled, and connect it to the Mac by USB if wireless discovery is unavailable. If install reports error 4016 or the device reports no tunnel/DDI services, restore the device connection before retrying. Do not replace this with an OS-version override or lower the deployment target: the Core AI APIs require iOS 27.

## Put the model on the phone

SpeechPhone uses a local model folder rather than the Models downloader. For recording and file input, download the reference v2 release from [Hugging Face](https://huggingface.co/okayuji/irodori-v4.1-small-mf-coreai-fp32) and copy it to the phone. Its 20 model files total **3,835,379,527 bytes**; license notices and documentation are additional. SpeechPhone also supports [the legacy FP32 folder](runtime.md#prepare-a-local-model-folder) for text-only generation: four assets totaling 3,368,343,098 bytes, or **3,375,061,725 bytes (3.38 GB)** including tokenizer and metadata. These are storage sizes, not working-memory requirements. Specialization needs additional storage and memory. No model weights are embedded in the app.

Copy the complete model folder into **Files → On My iPhone → Irodori Speech**. For example, copy a prepared reference folder with:

```sh
xcrun devicectl device copy to --device "$phone_device" \
  --domain-type appDataContainer --domain-identifier com.coreai-samples.SpeechPhone \
  --source /path/to/irodori-reference-fp32-v2 \
  --destination Documents/irodori-reference-fp32-v2
```

Let the copy finish before opening the model. Do not use `--remove-existing-content true`. `SpeechChecks` is reserved for diagnostic output; give the model a different folder name. The app lists visible local folders containing `irodori.json`, then verifies all asset sizes/digests and the pinned tokenizer before inference. An incomplete or changed copy is rejected.

## Use the app

1. Launch Irodori Speech normally and tap **Refresh model folders**.
2. Choose the prepared folder. Wait for file verification to complete.
3. With a reference v2 folder, use **Record voice** or **Choose audio file** as described above. Enter a short passage or choose an example; seed `1708` is the default. Input above 255 body tokens is rejected.
4. Tap **Generate speech**, then **Play** or **Save WAV…**. The result shows audio duration, generation duration and normalized text. The WAV is 48 kHz mono Float32.
5. **Stop** waits for resource cleanup before allowing another operation. Leaving the foreground also ends the operation. Audio-session interruption returns playback to idle without discarding the completed WAV.

Keep the app in the foreground during model preparation and generation. Editing text or seed invalidates the previous audio. First specialization can take longer than later runs. Reading accuracy and naturalness still require listening. The app preserves upstream normalization, empty caption, four steps, no trim and no watermark. A selected reference conditions the voice with v2; without a reference, generation uses the model's default voice.

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

The report records OS/build, hardware, Core AI architecture, thermal/power state, per-sample generation time, waveform size and process memory. `availableBytes` is the runtime process allowance, `footprintBytes` is the current physical footprint, and `peakFootprintBytes` is the process-lifetime peak, not a separately reset peak for each utterance. This three-call check is not an endurance or production benchmark. The first two texts correspond to the earlier lab CLI's `a00-s1708` and `a01-s1708`; the third repeats the first. Repeat equality passed in the October 29 reference preflight; comparison with original PyTorch waveforms remains separate and incomplete.

The diagnostic plays to completion, then calls stop/release. Stopping partway through audible playback, real system interruption, save-panel interaction and subjective listening remain separate manual checks. To return to the normal screen, launch the app without the diagnostic arguments.

## Validation limits

The diagnostic results cover the recorded assets, inputs and device conditions. Microphone capture, audible playback, speaker similarity, naturalness and long-running iPhone memory behavior have not been established by those measurements. Original-PyTorch waveform parity remains unresolved for the native reference path. See the [model card](model-card.md) for numerical results and upstream usage conditions.

See [reference-input validation](reference-input-validation.md) for current CPU/build/device evidence. [Earlier implementation validation](iphone-validation.md) retains the previous connection and verification history.
