# VoxCPM2 Mac app acceptance (2026-11-18)

[Demo guide](../voxcpm2.md) · [Model report](../model-reports/voxcpm2.md) · [Download check](voxcpm2-download-20261118/README.md)

Issue #595 (parent #573). On 2026-11-18 the owner checked the VoxCPM2 demo in the normal Mac app and reported the result. This is a manual check, not an automated test. The owner's report and the facts observed on the Mac afterwards are recorded separately.

## Conditions

| Item | Value |
|---|---|
| Date | 2026-11-18 |
| Device | Apple M5 Max (Mac17,6) |
| OS | macOS 27.2.1 (26B131) |
| Repository version at the download (observed) | `main` at `c90a4aad96c75d671c0f4ef65f575972f4d5df41` (#594), the commit of the repository's working tree when the model was downloaded. How the app was built and launched was not observed. |
| Configuration | `fp16`, all components in FP16, observed in the app's model library as artifact `voxcpm2-fp16`, contract `voxcpm2-coreai-v1`: 22 files, 4,677,413,664 bytes. In the repository version above, the app downloads only this configuration. |
| Download source set in that repository version | [okayuji/voxcpm2-coreai](https://huggingface.co/okayuji/voxcpm2-coreai) at the pinned revision `afe11d61de9dd7c3609140aa040b777ef77b2555`, folder `fp16/`. This is the app's setting in the repository version above, not an observation of the commit that the owner's download used. |

The device, OS, repository version and downloaded files come from the observations below, and the download source from the repository's setting at that version; none of them comes from the owner's report.

## Owner's report

The owner reported the result in the development conversation on 2026-11-18. The owner described it as very good and chose the answer that both the app and the voice had been checked. That answer states:

- the model could be downloaded, and speech generated, played and saved, in the Mac app;
- Irodori can be used as before;
- the voice quality has no problem.

The owner decided to keep the configuration with all components in FP16 as the default.

The steps given to the owner before the check also included an example sentence with a voice description and **Stop**: generate, play, stop and save a WAV, then check that Irodori still works. The answer came after those steps but did not go through them item by item; it does not list the voice description or **Stop** separately.

## Observed on the Mac

These facts were read on the Mac after the report. They show that the app on this Mac downloaded the model; they do not show the owner's operations on the screen.

- The app's model library records the artifact `voxcpm2-fp16` of the model `voxcpm2` (contract `voxcpm2-coreai-v1`, upstream revision `7936080b41163874f38610f46f438e2c0757f9fd`, 4,677,413,664 bytes) at 2026-11-18 04:54 UTC (13:54 JST). Its folder holds 22 files of that total size. As the version, the record holds the SHA-256 of the manifest, not a commit of the download source.
- At that time the repository's working tree was `main` at `c90a4aa`, after #594. In that version the app acquires VoxCPM2 only by downloading, from the source set above; the screen has no import. The build that the owner ran and the source commit that the owner's download used were not observed.
- The Mac is an Apple M5 Max (Mac17,6) with macOS 27.2.1 (26B131).

## Not covered

- Scored listening (similarity, naturalness or clarity) and the individual results of comparing the configurations by listening.
- The sentences, seeds, number of generations and times of the check; none were recorded, so this is not a performance measurement.
- Other Macs, iPhone, long continuous use, and interrupted or resumed downloads and generations.
