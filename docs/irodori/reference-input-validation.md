# Reference input validation

Scope: #368 / #370, October 29, 2026. The reference runtime from #362 / #363 is the dependency. This change connects recording, file selection, preview and generation in the normal iOS Models screen and SpeechPhone.

| Acceptance | Evidence | State |
|---|---|---|
| REF-UI-01: Recording/file success, failure, cancellation and stale completions remain distinct; the selected reference reaches generation. | SpeechReferenceFeatureTests, SpeechReferenceRunnerTests; existing service-forwarding tests. | Passed |
| REF-UI-02: Input shutdown and stale-result rejection survive background/exit and restart. | SpeechReferenceControlTests reproduces delayed finish → stop → new recording; session tickets reach the recorder. Managed lifecycle tests and independent review cover cleanup and lease joining. | Passed in CPU tests/source review; actual microphone operation remains a personal check. |
| REF-UI-03: File conversion and invalid-input rejection. | SpeechReferenceReaderTests: mono averaging, 24/44.1/96 kHz conversion, minimum duration, long/silent/broken files. Independent CPU boundary check covered 8/192 kHz × 0.5/30 seconds. | Passed |
| REF-UI-04: Models registration, signed app connection and required regressions. | LocalSpeechCatalogTests.referenceVariant; all CPU tests; strict lint and format; three Release products and compiled translations. | Passed for local acquisition. Public downloading awaits the owner's publication choice. |
| REF-UI-05: Make the merged main version available on iPhone 17 Pro. | Shared-runtime preflight generated and saved three reference-conditioned WAVs; the main app's explicit diagnostic and local import use the same implementation. Main deployment and acquisition are checked after merging. | Preflight passed; post-merge preparation follows. |
| REF-UI-06: Personal recording/file operation, listening and usability. | The user requested checking the completed interface on main. | Pending user confirmation |

The final host check reported **980 tests**, including **47 conditional skips**, with zero failures. Strict lint reported zero violations; changed Swift formatting passed. CoreAISamples, CoreAISamples-iOS and SpeechPhone Release builds succeeded. Compiled localization checks found zero mismatches across the normal iOS app (1,780 lookups), its ModelLibraryUI bundle (142), SpeechPhone (1,774), and Mac app (1,772). The Mac source-key checker reported 438 keys from 100 files without missing entries.

The iPhone preflight is retained in [reference-input-device.json](reference-input-device.json). It used iPhone 17 Pro / iOS 27.0.1 (24A446), charging over USB, thermal state nominal, a synthetic 3.88-second reference, seed 1708 and a greeting → announcement → greeting sequence. It saved 4.00 / 4.24 / 4.00-second WAVs and the repeated greetings matched. The first timed generation was 13.734 seconds and the next two were 3.904 / 3.684 seconds; these values exclude prior model hashing and input-file decoding. Observed kernel peak footprint was 1,144,576,904 bytes. Playback and microphone capture were deliberately not executed by the diagnostic.

## Issues found and fixed

- A delayed finish command could end a new recording after stop. Independent CPU reproduction confirmed it. Stop now cancels the recording-control effect, and the same session ticket is checked inside the recorder before sending the finish event. Both cancelled and uncancelled stale-ticket paths were checked.
- The iOS GPU codec decoder aborted on latent `[1,100,32]` with `Failed to resolve dynamic dimensions for memref.alloc`. Stage tracing showed all preceding components and four DiT steps completing. Turning EFR off for the iOS decoder resolved the three-generation check. Mac decoder behavior is unchanged. The earlier attempt remains a failure; the crash's large virtual Guard region is not treated as required physical RAM.

Independent review found no remaining confirmed code defect after these fixes. It did not execute a model or record a microphone. The unresolved Python waveform-fidelity screen and the limits of native smoke testing remain in [reference-runtime.md](reference-runtime.md). Tests and build success do not establish speaker similarity, naturalness, maximum-length iPhone behavior or long-run memory stability.

The operation guide and model-report records were updated, retaining pending personal acceptance and the earlier evidence. The public model card is a publication draft; no placeholder download source has been inserted. Existing source-model folders and unrelated working-tree changes were kept intact.

After integrating concurrent main changes through `b1e27bf`, the full host suite reported 995 tests with 47 conditional skips and no failures. All three Release products, strict lint (514 files), changed-file format (33 Swift files) and model-report checks passed again. The earlier 980-test count above refers to the pre-integration check.
