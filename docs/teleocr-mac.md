# TeleOCR in the Mac app

The OCR demo reads a single document image and displays text and tables. It uses the shared page reader and supports saving Markdown/JSON files and sharing their text. This guide covers the evaluated local FP16 bundle; it does not describe a published Hugging Face download.

## Open the demo

1. Build `CoreAISamples` from the repository's `main` branch with Xcode 27 on an Apple silicon Mac running macOS 27.
2. Open **Models → Documents → TeleOCR** and choose **Import Model…**.
3. Select the evaluated FP16 bundle folder. The app verifies its metadata and all file sizes/hashes before registering its own copy. The accepted [inventory](../CoreAISamplesKit/Sources/ModelLibraryStorage/Resources/teleocr-fp16-files.json) is fixed; a different conversion run is not automatically accepted merely because it contains the same model name.
4. Choose **Open OCR**, then select or drop one image file. The preview uses the same decoded pixels as the reader.
5. Choose **Read**. The first model preparation may take longer than later loads. Progress describes preparation or reading; the existing runtime does not provide a block-by-block progress stream.

PNG/JPEG with supported 8-bit RGB/grayscale decoding are verified. Stored pixels are used without EXIF rotation. If necessary, rotate and export a copy in an image editor before choosing it. PDF input, multiple pages and editing recognition results are not included.

## Stop, leave and export

**Stop** cancels the request and waits for its completion and model release. Wait for the stopped state before choosing or reading another image. Leaving OCR, closing its window or moving its scene to the background also closes the OCR session. Open OCR again after returning; the selected image may be retained, while the model and completed result are not transferred to a new owner.

OCR windows share one model owner. Close the OCR demo in the other window before retrying an occupied session. This does not migrate the app's other Mac demos into the OCR ownership group.

Use **Save → Save Markdown… / Save JSON…** to choose a file destination. **Share Markdown / Share JSON** shares the corresponding text, not a separately generated file attachment. A result stopped by the output limit remains marked as incomplete. Unparsed layout and skipped regions are also shown; generation completion is not a guarantee that every region is correct.

## Verification scope

The shared native reader matches the fixed F2/F4 CLI page outputs, including text/table structure, token counts, EOS and Markdown. The Models integration check uses real import, preparation, lease ownership and the production OCR session/service with a test rendering adapter. Saved synthetic results were separately rendered in Japanese/light and English/dark layouts.

These checks do not replace the owner's normal-Models real-photo and interactive stop/retry acceptance on `main`. iPhone acceptance and publication are separate tasks. Native support marks are generated from their evidence and are not promoted merely because a build succeeds.
