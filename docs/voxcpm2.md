# VoxCPM2 speech demo

[Home](../README.md) · [日本語](voxcpm2.ja.md)

The macOS app has a **VoxCPM2** screen that generates speech from text, optionally with a written description of the voice, using the FP16 Core AI bundle of [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (Apache-2.0). Download the model from the **VoxCPM2** heading in the sidebar, select the row under it, then generate, play, stop and save a 48 kHz mono Float32 WAV. There is no iPhone version.

The app downloads the `fp16` folder of the published bundle [okayuji/voxcpm2-coreai](https://huggingface.co/okayuji/voxcpm2-coreai) at the pinned revision `afe11d61de9dd7c3609140aa040b777ef77b2555`. The bundle was made with the [conversion recipe](../Tools/Conversion/recipes/voxcpm2/README.md). Measured fidelity, times and memory are in the [conversion validation record](validation/voxcpm2-conversion-20261117.md). On 2026-11-18 the owner reported that download, generation, playback and saving worked on the normal screen of a Mac and that the voice had no problem ([acceptance record](validation/voxcpm2-mac-acceptance-20261118.md)). No score for the audio quality has been recorded.

## Demo recording

The Mac recording uses the same Japanese greeting for both takes. The first voice description asks for a middle-aged man with a low, steady voice. The second asks for a young woman with a gentle, calm voice.

[Watch the silent preview (1× speed)](media/voxcpm2-mac.gif) · [Listen to the full recording](https://x.com/oka_yuuji/status/2123952954755752189)

## Download the model

1. Use **Download model** on the **VoxCPM2** heading. The app downloads 22 pinned files, 4,677,413,664 bytes in total (about 4.7 GB), without signing in. It checks the size and SHA-256 of every file and the configuration in the manifest, then shows the model as ready. Opening the model later makes one more verified copy in the app's caches, so allow space for both.
2. Only this FP16 configuration can be downloaded. The FP32 bundle and the FP16 bundle with an FP32 projection are not offered, and the heading has no folder import.
3. If you imported the FP16 bundle with an earlier version of the app, the heading shows it as ready without downloading again. The download uses the same identity, folder and model-library record as that import. The identity does not follow the published revision: a later revision with the same configuration and manifest uses the same folder.

If a later version of the app pins a different inventory, an earlier copy can still appear ready but fail to open. The regular check of a copy that is already recorded looks at its files and their sizes, not their contents, so a copy whose contents changed at the same sizes also appears ready and fails the full check of every file when the screen opens it. Delete the copy from the heading and download it again.

## Use the screen

1. Select the **VoxCPM2** row. The screen opens the downloaded model by itself and verifies all files again; this does not run the model. Without the model, the screen asks you to download it from the sidebar heading.
2. Choose an example or enter text. To describe the voice, write a few words in the **Voice description** field below the text, for example `A young woman, gentle and calm voice`. The screen removes the whitespace at both ends and every half-width or full-width parenthesis from the description, then passes the rest in half-width parentheses before the text; the text itself is not changed. If no description remains, the model receives only the text. If the description contains hiragana or katakana, the screen first translates it to English on this Mac and adds the translation before the text in the same way. A description in English or Chinese, or one with only kanji, is passed as it is. The text to speak is never translated. After generation, **Text used for generation** shows the text that reached the model. If the Japanese and English translation languages are not installed on this Mac, the screen does not generate and asks you to download them in System Settings or to write the description in English. The translation can change with the OS version, so the same input and seed can then produce different audio. With an empty field, a description written in half-width parentheses at the start of the text, such as `(A young woman, gentle and calm voice)確認が終わりました。`, also works. With no description in the field or at the start of the text, the model chooses the voice. Choosing an example replaces both the text and the voice description, and **Clear** clears only the text. The examples are a sentence without a description, a Japanese sentence with a description and an English sentence with a description.
3. The text and the voice description share a limit of 144 text tokens, counted on the combined input, including the added parentheses, after whitespace normalization and the split of multi-character kanji tokens. Longer input is rejected without truncation. Input with characters that the model cannot read is also rejected, with a message that asks you to remove or replace them.
4. Press **Generate speech**. The first generation after the model is opened, after you return to the screen or after a failure also loads the model assets. A generation that reaches the model's length limit before the speech ends is reported as an error and returns no audio. The seed is an unsigned 64-bit integer; `1708` is the initial value. The same input and seed produce the same audio (with a translated voice description, as long as the translation stays the same); change the seed to try a different take.
5. **Play**, **Stop** and **Save WAV…** work as on the Irodori screen. The default file name is `voxcpm2-speech.wav`.

Leaving the screen stops the current operation and releases the model. While the screen is shown, the heading cannot delete the model. The VoxCPM2 and Irodori screens keep their own text, seed, audio and model folder.

## Not included

Reference audio (voice cloning), streaming playback, volume normalization, other bundle configurations, folder import from the screen and iPhone.

## Checks

- `swift test --package-path CoreAISamplesKit` covers the catalog and the download definition (`MacDemoCatalogTests`, `VoxCPMDownloadTests`), the generate button condition (`SpeechGenerationGateTests`), the screen's store (`SpeechStoreTests`), the composition of the voice description and the text (`SpeechVoiceTests`), the translation path for a voice description with kana (`VoiceTranslationFeatureTests`, `VoiceTranslationRunnerTests`) and the text errors (`VoxCPMTextErrorTests`). The app's `--check-localization` compares the sidebar with the catalog and checks the Japanese strings.
- `VOXCPM2_TOKENIZER=/path/to/fp16 swift test --package-path CoreAISamplesKit --force-resolved-versions --filter SpeechRuntimeTests.VoxCPMTokenizerTests` also counts the text tokens of the examples. It reads only `tokenizer.json`.
- `COREAI_VOICE_TRANSLATION_TEST=1 swift test --package-path CoreAISamplesKit --force-resolved-versions --filter VoiceTranslationRunnerTests` calls the translation on this Mac once and checks that the result is not empty and has no kana. Run it on a Mac with the Japanese and English translation languages installed. It does not read the model. The default run skips it.
- The download test in `AGENTS.md` runs under the shared heavy-job lock. It downloads the pinned files without credentials into an empty temporary model library, checks every file, takes the usage lease, generates one example through the production service and store, stops, returns the lease and deletes the copy.
- The end-to-end import test in `AGENTS.md` imports a local FP16 bundle through the same model library, which the screen no longer offers, and generates every example.
- These checks do not establish listening quality or use of the normal screen. On 2026-11-18 the owner reported download, generation, playback and saving in the app with no problem in the voice; the [acceptance record](validation/voxcpm2-mac-acceptance-20261118.md) gives the scope of that report. On 2026-11-19, after the screen changes on main (separate voice-description field, seed explanation, on-device translation of a voice description that contains hiragana or katakana, fixed-height text field), the owner answered that the updated screen was very good and showed a 36.8 s screen recording without an audio track, in which the three bundled examples are selected, generated and played in the Japanese display. The recording does not show a voice description written in Japanese, a changed seed, Stop or saving.
