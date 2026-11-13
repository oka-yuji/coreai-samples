# Mobile Image Lab release verification

This record covers [#562](https://github.com/oka-yuji/coreai-samples/issues/562). The [public model](https://huggingface.co/okayuji/mobile-image-lab-coreai) is distributed for noncommercial research/evaluation under the Qwen Research License. [Exact identities and checks](results.json).

The 54-file release and its full reachable Hugging Face history passed local, authenticated, gate and anonymous information audits. The app obtains only the 51 files under `bundle/`. Its original 48 computational/support files remain byte-identical; two neutral headings and their manifest entries establish a separate public identity. The old local-import artifact remains registered independently.

The new catalog, explicit calibration binding, permission ownership, cancellation, deletion and legacy selection passed CPU tests. The complete package suite passed after updating its expected artifact count from four to five. Mac Release and signed iOS Release passed. Both actual dependency sets match the pinned lockfiles. Actual app signatures, iOS file-sharing keys, increased-memory entitlement and 20 compiled Japanese/English entries were inspected. The Mac app does not contain the iOS-only ImageGenerationUI bundle.

Fresh production acquisition passed from an empty host Models store with no migration candidates. All 51 anonymous requests returned HTTP 200, and all 13,783,495,057 bytes passed canonical inspection plus an independent streaming hash comparison against the frozen release inventory. The acquired resource bound to ManagedImageGeneration and the real LiveImageGenerationService with the explicit known-phone calibration, then released; the storage lease count returned to zero. This does not exercise iOS adapter rendering or image generation. [Downloaded file hashes](download-files.json).

The entire verification wrapper took 1,426.31 seconds, including the download, inspections and binding checks. The diagnostic retains its existing 1,800-second acquisition-settle limit; slow networks can fail this bounded check. The limit belongs to the explicit diagnostic, not ordinary app downloads. This duration is not an inference benchmark or a transfer-speed guarantee.

The iOS runtime still accepts only `iPhone18,1` / `24A446` with the exact calibrated identity. Carrying forward the five unchanged component budgets is not a new device measurement. The historical ten-request measurements and unmet performance goals are unchanged. QD-07 remains an owner check on the target iPhone.

Raw execution logs, audit inventories and failed TSF runs remain private. Public records contain hashes, fixed revisions and bounded conclusions; they contain no local machine paths or credentials.
