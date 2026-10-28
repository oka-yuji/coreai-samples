# Reference input validation

Scope: #368 / #370, October 29, 2026. The reference runtime from #362 / #363 is the dependency. This change connects recording, file selection, preview and generation in the normal iOS Models screen and SpeechPhone.

| Acceptance | Evidence | State |
|---|---|---|
| REF-UI-01: Recording/file success, failure, cancellation and stale completions remain distinct; the selected reference reaches generation. | SpeechReferenceFeatureTests, SpeechReferenceRunnerTests; existing service-forwarding tests. | Passed |
| REF-UI-02: Input shutdown and stale-result rejection survive background/exit and restart. | SpeechReferenceControlTests reproduces delayed finish → stop → new recording; session tickets reach the recorder. Managed lifecycle tests and independent review cover cleanup and lease joining. | Passed in CPU tests/source review; actual microphone operation remains a personal check. |
| REF-UI-03: File conversion and invalid-input rejection. | SpeechReferenceReaderTests: mono averaging, 24/44.1/96 kHz conversion, minimum duration, long/silent/broken files. Independent CPU boundary check covered 8/192 kHz × 0.5/30 seconds. | Passed |
| REF-UI-04: Models registration, signed app connection and required regressions. | LocalSpeechCatalogTests.referenceVariant; all CPU tests; strict lint and format; three Release products and compiled translations. | Passed for local acquisition. Public downloading awaits the owner's publication choice. |
| REF-UI-05: Make the merged main version available on iPhone 17 Pro. | Main `86e0499` installed; normal Models acquisition verified the v2 inventory and reached ready. The normal app's explicit diagnostic generated/saved three WAVs and passed repeat equality. Ordinary launch then returned to Models. | Passed; see post-merge evidence below. |
| REF-UI-06: Personal recording/file operation, listening and usability. | The user requested checking the completed interface on main. | Pending user confirmation |

The final host check reported **980 tests**, including **47 conditional skips**, with zero failures. Strict lint reported zero violations; changed Swift formatting passed. CoreAISamples, CoreAISamples-iOS and SpeechPhone Release builds succeeded. Compiled localization checks found zero mismatches across the normal iOS app (1,780 lookups), its ModelLibraryUI bundle (142), SpeechPhone (1,774), and Mac app (1,772). The Mac source-key checker reported 438 keys from 100 files without missing entries.

The iPhone preflight is retained in [reference-input-device.json](reference-input-device.json). It used iPhone 17 Pro / iOS 27.0.1 (24A446), charging over USB, thermal state nominal, a synthetic 3.88-second reference, seed 1708 and a greeting → announcement → greeting sequence. It saved 4.00 / 4.24 / 4.00-second WAVs and the repeated greetings matched. The first timed generation was 13.734 seconds and the next two were 3.904 / 3.684 seconds; these values exclude prior model hashing and input-file decoding. Observed kernel peak footprint was 1,144,576,904 bytes. Playback and microphone capture were deliberately not executed by the diagnostic.

## Issues found and fixed

- A delayed finish command could end a new recording after stop. Independent CPU reproduction confirmed it. Stop now cancels the recording-control effect, and the same session ticket is checked inside the recorder before sending the finish event. Both cancelled and uncancelled stale-ticket paths were checked.
- The iOS GPU codec decoder aborted on latent `[1,100,32]` with `Failed to resolve dynamic dimensions for memref.alloc`. Stage tracing showed all preceding components and four DiT steps completing. Turning EFR off for the iOS decoder resolved the three-generation check. Mac decoder behavior is unchanged. The earlier attempt remains a failure; the crash's large virtual Guard region is not treated as required physical RAM.

Independent review found no remaining confirmed code defect after these fixes. It did not execute a model or record a microphone. The unresolved Python waveform-fidelity screen and the limits of native smoke testing remain in [reference-runtime.md](reference-runtime.md). Tests and build success do not establish speaker similarity, naturalness, maximum-length iPhone behavior or long-run memory stability.

The operation guide and model-report records were updated, retaining pending personal acceptance and the earlier evidence. The public model card is a publication draft; no placeholder download source has been inserted. Existing source-model folders and unrelated working-tree changes were kept intact.

After integrating concurrent main changes through `b1e27bf`, the full host suite reported 995 tests with 47 conditional skips and no failures. All three Release products, strict lint (514 files), changed-file format (33 Swift files) and model-report checks passed again. The earlier 980-test count above refers to the pre-integration check.

## Post-merge main deployment

On October 29, 2026, the signed **CoreAISamples-iOS** product was built from merged main `86e049964ea4f7790db1c70bf735a8db8d41cc06` and installed on the requested iPhone 17 Pro. The normal app has bundle identifier `com.coreai-samples.DemoApp.ios`. Its built Info.plist contains both file-sharing keys and the microphone purpose string.

The model-library acquisition check used the normal app's managed store and `irodori-v4.1-small-mf-reference-fp32`. It reported `catalogValidated: true`, no failures, `route: local-import`, and both `state` and `inspection` as `ready`. Defined and measured payload size matched at **3,836,181,569 bytes**. The imported `irodori-coreai-v2` manifest revision was `8dca0128ec50d8b186a2f9acb6e6d658f65e9e21aae0ac766bad650e5ede4ea9`.

The normal app's explicit reference diagnostic then ran on iOS 27.0.1 (24A446), charging over USB, nominal thermal state, low-power mode off, seed 1708 and the same synthetic 3.88-second reference. Run ID: `main-86e0499-20261029T070944`. This is distinct from the earlier SpeechPhone preflight.

| Example | Output seconds | Generation seconds | WAV SHA-256 |
|---|---:|---:|---|
| greeting | 4.00 | 32.145860125 | `5ec98369d2abeeb4d48b8773bf6de56916b8f08c1861cdbee86924dbd4adf792` |
| announcement | 4.24 | 3.970625041 | `80b83455e8f0fd4a08d0ae5c862b72f6ffa804b09945889edd289433d8f71db5` |
| greeting repeat | 4.00 | 3.819377833 | `5ec98369d2abeeb4d48b8773bf6de56916b8f08c1861cdbee86924dbd4adf792` |

The process exited successfully, saved all three WAVs, and reported `repeatedWAVMatches: true`. Kernel lifetime peak footprint was **1,194,105,808 bytes**; final footprint after release was **156,879,168 bytes**. Generation intervals exclude prior folder hashing and audio-file decoding. The first generation includes initial specialization and must not be compared as a steady-state measurement with the next two.

The acquisition report remains in the app's Documents directory as `main-86e0499-20261029T070944-acquisition.json`; the diagnostic report, status and WAVs remain under `SpeechChecks/main-86e0499-20261029T070944/`. Reports and WAV hashes were retrieved and checked after execution. The app was then launched without diagnostic arguments, leaving the model available from normal Models. Existing device folders were retained.

No microphone capture or sound playback was performed by these commands. REF-UI-06, personal voice quality, the maximum-length iPhone check and the earlier PyTorch fidelity miss remain unresolved. The prepared local model makes this phone usable while external model publication and in-app downloading await the owner's decision.

The follow-up updates only this evidence and the iPhone guide. It reuses the passing code checks for the unchanged app revision; document links, recorded values and `git diff --check` are checked separately. The retrospective finding is to verify model acquisition and generation in the normal app after merging, and to label historical text-only instructions so they do not contradict the reference input controls.
