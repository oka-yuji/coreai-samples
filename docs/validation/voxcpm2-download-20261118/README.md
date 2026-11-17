# VoxCPM2 public download check (2026-11-18)

[Demo guide](../../voxcpm2.md) · [Model report](../../model-reports/voxcpm2.md) · [Results](results.json)

Issue #593. This record covers one fresh download of the published FP16 bundle through the production Mac model library, followed by one generation through the production speech service. It does not cover the app's normal screen, playback, listening, iPhone or performance.

## Published bundle

| Item | Value |
|---|---|
| Repository | [okayuji/voxcpm2-coreai](https://huggingface.co/okayuji/voxcpm2-coreai) |
| Revision | `afe11d61de9dd7c3609140aa040b777ef77b2555` |
| Folder | `fp16/` |
| Files | 22 files, 4,677,413,664 bytes. Sizes and SHA-256 equal the bundled inventory `CoreAISamplesKit/Sources/SpeechAssets/Resources/voxcpm2-fp16.inventory.json`. |
| Models identity | Artifact `voxcpm2-fp16` with the local manifest SHA-256 `fbe90a07bcdb4736b2b8b9dcd0b1f11b71857c28e4d506aad216bcda6c76754c`, the same identity as the import definition |

## Anonymous download

`COREAI_VOXCPM2_DOWNLOAD_OUTPUT=/new/output swift test --package-path CoreAISamplesKit --force-resolved-versions -c release --filter VoxCPMDownloadNativeTests` (VDL-04) ran under the shared heavy-job lock with the Hugging Face token variables removed from the environment. The test's file loader accepted only the 22 expected URLs under `fp16/` at the pinned revision and required every request to carry no header fields. The transfer session uses no cookies, credential storage or URL cache. All 22 responses were HTTP 200. The model library started empty and had no migration candidates.

| Step | Result |
|---|---|
| Download of 22 files, each checked for size and SHA-256 | 138.3 s |
| Validation before the model became ready: every file and the manifest configuration | 1.51 s |
| Separate full inspection | 1.60 s |

The times depend on the network and the machine and are not performance measurements.

## Generation and release

| Step | Result |
|---|---|
| Usage leases | 0 after the download, 1 in use, 1 after the speech store stopped, 0 after use ended |
| Open with the full hash check | 1.77 s |
| Example `plain`, seed 1708 | 2.40 s of audio (15 patches, 115,200 samples, 48 kHz mono Float32) in 9.28 s including the asset load |
| Stop, end of use, deletion | 0.08 s, under 0.001 s, 0.05 s |

The downloaded copy was deleted after the leases returned to zero, and the model library's record was then empty. The open reused the existing verified copy in the test runner's caches. No audio was played or listened to.

## Conditions

Apple M5 Max with 137,438,953,472 bytes of memory, macOS 27.2.1 (26B131), Xcode 27.2 (27B52) and Swift 6.4.2. Swift package tests in release, built from the #593 working tree based on `ccc26fd`.

## Not covered

Download and generation on the app's normal screen, playback and listening (the owner's acceptance, VDL-07), iPhone, the FP32 and FP16-with-FP32-projection configurations, interrupted or resumed downloads, and the performance of downloads or generation.
