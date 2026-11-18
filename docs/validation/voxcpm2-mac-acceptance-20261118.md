# VoxCPM2 Mac app acceptance (2026-11-18)

[Demo guide](../voxcpm2.md) · [Model report](../model-reports/voxcpm2.md) · [Download check](voxcpm2-download-20261118/README.md)

Issue #595 (parent #573). On 2026-11-18 the owner reported the result of a check of the VoxCPM2 demo in the normal Mac app. This is a manual check, not an automated test. The owner's report, the facts observed on the Mac afterwards and the facts about the repository version are recorded separately.

## Conditions

| Item | Value |
|---|---|
| Date | 2026-11-18, the day of the owner's report |
| Device | Apple M5 Max (Mac17,6): the Mac whose app model library holds the entry below. The owner did not name the machine; that the check ran on this Mac is inferred from that entry. |
| OS | macOS 27.2.1 (26B131), on the same Mac, read after the report |
| Repository version when the asset was recorded (observed) | `main` at `c90a4aad96c75d671c0f4ef65f575972f4d5df41` (#594), the commit of the repository's working tree when the model library recorded the asset. How the app was built and launched was not observed. |
| Configuration | `fp16`, all components in FP16, observed in the app's model library as artifact `voxcpm2-fp16`, contract `voxcpm2-coreai-v1`: 22 files, 4,677,413,664 bytes. It is the only VoxCPM2 entry there, and in the repository version above the app uses only this configuration; that the owner generated with it is inferred from these two facts. |
| Download source set in that repository version | [okayuji/voxcpm2-coreai](https://huggingface.co/okayuji/voxcpm2-coreai) at the pinned revision `afe11d61de9dd7c3609140aa040b777ef77b2555`, folder `fp16/`. This is the app's setting in the repository version above, not an observation of the commit that the owner's download used. |

The device, OS, repository version and recorded asset come from the observations below, and the download source from the repository's setting at that version; none of them comes from the owner's report.

## Owner's report

The owner reported the result in the development conversation on 2026-11-18. The owner described it as very good and chose the answer that both the app and the voice had been checked. That answer states:

- the model could be downloaded, and speech generated, played and saved, in the Mac app;
- Irodori can be used as before;
- the voice quality has no problem.

In the same answer, the owner decided to keep the configuration with all components in FP16 as the default.

The steps given to the owner in the development conversation before the check also included an example sentence with a voice description and **Stop**: generate, play, stop and save a WAV, then check that Irodori still works. The answer came after those steps but did not go through them item by item; it does not list the voice description or **Stop** separately.

## Observed on the Mac

These facts were read on the Mac after the report. They show that the app's model library on this Mac recorded the FP16 asset at the time below. They do not show how the asset was acquired, because the library keeps the same entry for a download and for a local import, and they do not show the owner's operations on the screen. That the download worked comes from the owner's report.

- The app's model library holds an entry for the artifact `voxcpm2-fp16` of the model `voxcpm2` (contract `voxcpm2-coreai-v1`, upstream revision `7936080b41163874f38610f46f438e2c0757f9fd`, 4,677,413,664 bytes), written at 2026-11-18 04:54 UTC (13:54 JST). It is the only VoxCPM2 entry. The time is when the library recorded the asset, not a recorded time of the owner's check. The asset's folder holds 22 files of that total size. As the version, the entry holds the SHA-256 of the manifest, not a commit of the download source.
- When that entry was written, the repository's working tree was `main` at `c90a4aa`, after #594 (from its reflog). In that version the app acquires VoxCPM2 only by downloading, from the source set above, and the screen has no import; a copy imported with an earlier version is shown as ready without a download. The build that the owner ran and the source commit that the owner's download used were not observed.
- The Mac that holds this model library is an Apple M5 Max (Mac17,6) with macOS 27.2.1 (26B131), read from the system.

## Not covered

The owner's report and the observations above contain nothing on the following, so this record does not cover them:

- Scored listening (similarity, naturalness or clarity) and the individual results of comparing the configurations by listening.
- The sentences, seeds, number of generations and times of the check; none were recorded, so this is not a performance measurement.
- Other Macs, iPhone, long continuous use, and interrupted or resumed downloads and generations.
