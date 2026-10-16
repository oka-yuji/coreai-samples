# Irodori in the Speech demo

The macOS app has a **Speech** screen for the verified Irodori v4.1 Small MF FP32 assets. Open a local model folder, enter text and a seed, generate speech, then play it or save a 48 kHz mono Float32 WAV. **Models → Open speech model…** opens the same screen. This model is separate from the Chat model selection.

Implementation and CPU verification are complete. Actual app interaction, GPU generation and listening are **pending user acceptance** under [issue #92, U6](https://github.com/oka-yuji/coreai-samples/issues/92). The earlier lab CLI verification is evidence for the reused numerical route, not a completed test of this app. This integration is not a production release or an iPhone TTS implementation.

## Build and prepare

Use an Apple silicon Mac with macOS 27, Xcode 27 and Rust/Cargo. Xcode invokes `Tools/Irodori/build-inputs.sh` before compiling the Mac app. That script also finds Cargo in `~/.cargo/bin` when Xcode has a restricted PATH. Changes to build settings belong in `project.yml`; regenerate with `xcodegen generate`.

```sh
xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples \
  -configuration Release -destination 'platform=macOS' \
  -derivedDataPath .build/SpeechDemo build
open .build/SpeechDemo/Build/Products/Release/CoreAISamples.app
```

Quit an older running copy of Core AI Samples before opening this build, so the shared bundle identifier does not bring the older process forward.

Prepare the model folder using [the runtime instructions](runtime.md#prepare-a-local-model-folder). Select the resulting parent folder containing `irodori.json`, `tokenizer.json` and `artifacts/`, rather than an individual `.aimodel`. No weights are bundled with the app or downloaded by this feature. The app verifies the fixed asset sizes and SHA-256 values when opening the folder; this stage does not execute a model.

## Use the screen

1. Choose **Speech → Open speech model…** and select the prepared folder. After validation, the status says that the files are verified. The last successful folder is remembered; **Open last model** reopens it explicitly on the next launch.
2. Use an example or enter a short passage. The fixed tokenizer permits at most 255 body tokens. Longer input is rejected without truncation. The seed is an unsigned 64-bit integer; `1708` is the initial value.
3. Press **Generate speech**. The first request loads and specializes the models. The status remains busy during this work. When ready, the screen shows the audio duration, generation duration and the normalized text used by the model.
4. Press **Play**, then **Stop** to interrupt playback, or let it finish. The result remains available for replay and saving. Editing text or seed discards that result so it cannot be mistaken for the new input.
5. Press **Save WAV…** and choose a destination. Cancelling the save panel preserves the generated audio. A save error is displayed without discarding it.

The generation duration measures model loading/specialization if needed, the inference stages and WAV encoding; it starts after text preprocessing. It is not total time since the button press, and it is not a memory measurement. The result uses no reference voice, empty caption, four MeanFlow steps, no trim and no watermark. The upstream normalization behavior, including its known removal of a Unicode minus sign, is preserved.

**Stop during generation** cancels the request cooperatively. It waits for the current model operation to return before releasing model resources and enabling another request. Switching away from Speech also cancels and releases resources. Returning while cleanup is in progress keeps generation disabled. Later generation loads the models again. A failed generation clears its runtime resources so a subsequent request can retry. A deleted or replaced model folder should be selected and validated again.

## User acceptance checklist (U6, not executed by the implementation agent)

- [ ] Open the prepared FP32 folder from both Speech and Models. Confirm the model name, example text and successful validation status.
- [ ] Generate the first example with seed `1708`. Confirm nonempty audio, plausible duration, displayed normalized text and intelligible speech. Listen to the complete result; waveform compatibility alone does not establish reading accuracy or naturalness.
- [ ] Play to completion, play again and stop, then play again. Confirm there is no overlapping playback and the buttons recover.
- [ ] Save a WAV, play the saved file in another player, cancel another save and confirm the original audio remains usable.
- [ ] Stop a generation, wait for the stopping state to finish, then generate again. Repeat while switching from Speech to another screen and back. Confirm that an older result never replaces the new request.
- [ ] Check missing/wrong model files, empty or whitespace input, invalid seed and input exceeding 255 tokens. Confirm the app explains the problem and permits correction.

Report the input text, seed, operation and displayed error for any failure. The unit's automated evidence and review scope are recorded in [demo-validation.md](demo-validation.md).
