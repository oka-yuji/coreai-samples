# VoxCPM2 speech demo

[Home](../README.md) · [日本語](voxcpm2.ja.md)

The macOS app has a **VoxCPM2** screen that generates speech from text, optionally with a written description of the voice, using the FP16 Core AI bundle of [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (Apache-2.0). Import the bundle from the **VoxCPM2** heading in the sidebar, select the row under it, then generate, play, stop and save a 48 kHz mono Float32 WAV. There is no iPhone version.

The bundle is not published and the app does not download it. Create it with the [conversion recipe](../Tools/Conversion/recipes/voxcpm2/README.md). Measured fidelity, times and memory are in the [conversion validation record](validation/voxcpm2-conversion-20261117.md). Listening quality and use of the normal screen have not been accepted yet.

## Import the bundle

1. The recipe writes three configurations. Choose the `fp16` folder. The app accepts only this configuration: 22 pinned files, 4,677,413,664 bytes in total. The FP32 folder and the FP16 folder with an FP32 projection are rejected.
2. Remove every other item from the folder before you choose it, such as a `.DS_Store` that Finder created. Extra, changed, missing or linked files are rejected. For these and for another configuration, the sidebar shows the same message: “The model files do not match the required bundle.”
3. Use **Import model** (the folder icon with a plus sign) on the **VoxCPM2** heading and choose the folder. Models copies it into the app's model library (about 4.7 GB), checks the size and SHA-256 of every file and leaves the chosen folder unchanged. Opening the model later makes one more verified copy in the app's caches, so allow space for both.

If a later version of the app pins a different inventory, an earlier import can still appear as imported but fail to open. Delete it from the heading and import the new folder.

## Use the screen

1. Select the **VoxCPM2** row. The screen opens the imported model by itself and verifies all files again; this does not run the model. Without an import, the screen asks you to import the model from the sidebar heading.
2. Choose an example or enter text. To describe the voice, write the description in half-width parentheses at the start of the text, for example `(A young woman, gentle and calm voice)確認が終わりました。`. There is no separate field for it. The examples are a sentence without a description, a Japanese sentence with a description and an English sentence with a description.
3. The text may contain at most 144 text tokens after whitespace normalization and the split of multi-character kanji tokens. Longer text is rejected without truncation. Text with characters that the model cannot read is also rejected, with a message that asks you to remove or replace them.
4. Press **Generate speech**. The first generation after the model is opened, after you return to the screen or after a failure also loads the model assets. A generation that reaches the model's length limit before the speech ends is reported as an error and returns no audio. The seed is an unsigned 64-bit integer; `1708` is the initial value.
5. **Play**, **Stop** and **Save WAV…** work as on the Irodori screen. The default file name is `voxcpm2-speech.wav`.

Leaving the screen stops the current operation and releases the model. While the screen is shown, the heading cannot delete the model. The VoxCPM2 and Irodori screens keep their own text, seed, audio and model folder.

## Not included

Reference audio (voice cloning), streaming playback, a separate field for the voice description, volume normalization, downloads, other bundle configurations and iPhone.

## Checks

- `swift test --package-path CoreAISamplesKit` covers the catalog (`MacDemoCatalogTests`), the generate button condition (`SpeechGenerationGateTests`), the screen's store (`SpeechStoreTests`) and the text errors (`VoxCPMTextErrorTests`). The app's `--check-localization` compares the sidebar with the catalog and checks the Japanese strings.
- `VOXCPM2_TOKENIZER=/path/to/fp16 swift test --package-path CoreAISamplesKit --force-resolved-versions --filter SpeechRuntimeTests.VoxCPMTokenizerTests` also counts the text tokens of the examples. It reads only `tokenizer.json`.
- The end-to-end test in `AGENTS.md` runs under the shared heavy-job lock. It imports the FP16 bundle into a temporary model library, takes the usage lease, generates every example through the production service and store, checks the WAV files, stops, returns the lease and confirms that the chosen folder is unchanged.
- These checks do not establish listening quality or use of the normal screen; those remain the owner's acceptance checks.
