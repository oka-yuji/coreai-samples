# Irodori speech demo

The macOS app has an **Irodori** screen for the verified Irodori v4.1 Small MF FP32 assets. Download the model from the **Irodori** heading in the sidebar, select the Irodori row, enter text and a seed, generate speech, then play it or save a 48 kHz mono Float32 WAV. This model is separate from the Chat model selection.

Native generation is covered by the [release asset checks](release-asset-validation.json). Those checks do not evaluate GUI playback or subjective pronunciation, naturalness and speaker similarity. For the iPhone download and reference-input flow, see [the iPhone guide](iphone.md).

## Build and prepare

Use an Apple silicon Mac with macOS 27, Xcode 27 and Rust/Cargo. Xcode invokes `Tools/Irodori/build-inputs.sh` before compiling the Mac app. That script also finds Cargo in `~/.cargo/bin` when Xcode has a restricted PATH. Changes to build settings belong in `project.yml`; regenerate with `xcodegen generate`.

```sh
xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples \
  -configuration Release -destination 'platform=macOS' \
  -derivedDataPath .build/SpeechDemo build
open .build/SpeechDemo/Build/Products/Release/CoreAISamples.app
```

Quit an older running copy of Core AI Samples before opening this build, so the shared bundle identifier does not bring the older process forward.

The download button on the **Irodori** heading fetches the pinned reference model (22 files, about 3.84 GB) from [Hugging Face](https://huggingface.co/okayuji/irodori-v4.1-small-mf-coreai-fp32) into the app's model library and verifies every file. No weights are bundled with the app. The screen verifies the fixed asset sizes and SHA-256 values again when it opens the downloaded folder; this stage does not execute a model. While the screen is shown, the heading's delete button is unavailable; leave the screen to delete the model.

## Use the screen

1. Select the **Irodori** row after the download finishes, or stay on the screen until it does. The screen opens the downloaded model by itself; after validation, the status says that the files are verified. Without a download, the screen asks you to download the model from the sidebar heading.
2. Use an example or enter a short passage. The fixed tokenizer permits at most 255 body tokens. Longer input is rejected without truncation. The seed is an unsigned 64-bit integer; `1708` is the initial value.
3. Press **Generate speech**. The first request loads and specializes the models. The status remains busy during this work. When ready, the screen shows the audio duration, generation duration and the normalized text used by the model.
4. Press **Play**, then **Stop** to interrupt playback, or let it finish. The result remains available for replay and saving. Editing text or seed discards that result so it cannot be mistaken for the new input.
5. Press **Save WAV…** and choose a destination. Cancelling the save panel preserves the generated audio. A save error is displayed without discarding it.

The generation duration measures model loading/specialization if needed, the inference stages and WAV encoding; it starts after text preprocessing. It is not total time since the button press, and it is not a memory measurement. The result uses no reference voice, empty caption, four MeanFlow steps, no trim and no watermark. The upstream normalization behavior, including its known removal of a Unicode minus sign, is preserved.

**Stop during generation** cancels the request cooperatively. It waits for the current model operation to return before releasing model resources and enabling another request. Switching away from Irodori also cancels and releases resources. Returning while cleanup is in progress keeps generation disabled. Later generation loads the models again. A failed generation clears its runtime resources so a subsequent request can retry. After the model is deleted, also in another window, the screen forgets it and asks you to download it again from the sidebar heading. **Generate speech** is available only while the screen holds the downloaded model. If opening the model fails, select another demo and return to try again.

## Validation limits

The native release checks use fixed synthetic inputs and verify generated WAVs. They do not establish pronunciation, naturalness, speaker similarity, normal-screen interaction or long-running memory stability. The original-PyTorch reference waveform difference remains documented in [the reference runtime](reference-runtime.md).

For a reproducible issue report, include the input text, seed, operation and displayed error. The implementation's automated evidence is recorded in [demo-validation.md](demo-validation.md).
