# VoxCPM2 speech demo

[Home](../README.md) · [日本語](voxcpm2.ja.md)

The macOS app has a **VoxCPM2** screen that generates speech from text, optionally with a written description of the voice, using the FP16 Core AI bundle of [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (Apache-2.0). Download the model from the **VoxCPM2** heading in the sidebar, select the row under it, then generate, play, stop and save a 48 kHz mono Float32 WAV. There is no iPhone version.

The app downloads the `fp16` folder of the published bundle [okayuji/voxcpm2-coreai](https://huggingface.co/okayuji/voxcpm2-coreai) at the pinned revision `afe11d61de9dd7c3609140aa040b777ef77b2555`. The bundle was made with the [conversion recipe](../Tools/Conversion/recipes/voxcpm2/README.md). Measured fidelity, times and memory are in the [conversion validation record](validation/voxcpm2-conversion-20261117.md). Listening quality and use of the normal screen have not been accepted yet.

## Download the model

1. Use **Download model** on the **VoxCPM2** heading. The app downloads 22 pinned files, 4,677,413,664 bytes in total (about 4.7 GB), without signing in. It checks the size and SHA-256 of every file and the configuration in the manifest, then shows the model as ready. Opening the model later makes one more verified copy in the app's caches, so allow space for both.
2. Only this FP16 configuration can be downloaded. The FP32 bundle and the FP16 bundle with an FP32 projection are not offered, and the heading has no folder import.
3. If you imported the FP16 bundle with an earlier version of the app, the heading shows it as ready without downloading again. The download uses the same identity, folder and model-library record as that import. The identity does not follow the published revision: a later revision with the same configuration and manifest uses the same folder.

If a later version of the app pins a different inventory, an earlier copy can still appear ready but fail to open. The regular check of a copy that is already recorded looks at its files and their sizes, not their contents, so a copy whose contents changed at the same sizes also appears ready and fails the full check of every file when the screen opens it. Delete the copy from the heading and download it again.

## Use the screen

1. Select the **VoxCPM2** row. The screen opens the downloaded model by itself and verifies all files again; this does not run the model. Without the model, the screen asks you to download it from the sidebar heading.
2. Choose an example or enter text. To describe the voice, write the description in half-width parentheses at the start of the text, for example `(A young woman, gentle and calm voice)確認が終わりました。`. There is no separate field for it. The examples are a sentence without a description, a Japanese sentence with a description and an English sentence with a description.
3. The text may contain at most 144 text tokens after whitespace normalization and the split of multi-character kanji tokens. Longer text is rejected without truncation. Text with characters that the model cannot read is also rejected, with a message that asks you to remove or replace them.
4. Press **Generate speech**. The first generation after the model is opened, after you return to the screen or after a failure also loads the model assets. A generation that reaches the model's length limit before the speech ends is reported as an error and returns no audio. The seed is an unsigned 64-bit integer; `1708` is the initial value.
5. **Play**, **Stop** and **Save WAV…** work as on the Irodori screen. The default file name is `voxcpm2-speech.wav`.

Leaving the screen stops the current operation and releases the model. While the screen is shown, the heading cannot delete the model. The VoxCPM2 and Irodori screens keep their own text, seed, audio and model folder.

## Not included

Reference audio (voice cloning), streaming playback, a separate field for the voice description, volume normalization, other bundle configurations, folder import from the screen and iPhone.

## Checks

- `swift test --package-path CoreAISamplesKit` covers the catalog and the download definition (`MacDemoCatalogTests`, `VoxCPMDownloadTests`), the generate button condition (`SpeechGenerationGateTests`), the screen's store (`SpeechStoreTests`) and the text errors (`VoxCPMTextErrorTests`). The app's `--check-localization` compares the sidebar with the catalog and checks the Japanese strings.
- `VOXCPM2_TOKENIZER=/path/to/fp16 swift test --package-path CoreAISamplesKit --force-resolved-versions --filter SpeechRuntimeTests.VoxCPMTokenizerTests` also counts the text tokens of the examples. It reads only `tokenizer.json`.
- The download test in `AGENTS.md` runs under the shared heavy-job lock. It downloads the pinned files without credentials into an empty temporary model library, checks every file, takes the usage lease, generates one example through the production service and store, stops, returns the lease and deletes the copy.
- The end-to-end import test in `AGENTS.md` imports a local FP16 bundle through the same model library, which the screen no longer offers, and generates every example.
- These checks do not establish listening quality or use of the normal screen; those remain the owner's acceptance checks.
