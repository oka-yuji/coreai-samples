# Mobile Image Lab release verification

This record covers [#562](https://github.com/oka-yuji/coreai-samples/issues/562). The [public model](https://huggingface.co/okayuji/mobile-image-lab-coreai) is distributed for noncommercial research/evaluation under the Qwen Research License. [Exact identities and checks](results.json).

The 54-file release and its full reachable Hugging Face history passed local, authenticated, gate and anonymous information audits. The app obtains only the 51 files under `bundle/`. Its original 48 computational/support files remain byte-identical; two neutral headings and their manifest entries establish a separate public identity. The old local-import artifact remains registered independently.

The new catalog, explicit calibration binding, permission ownership, cancellation, deletion and legacy selection passed CPU tests. The complete package suite passed after updating its expected artifact count from four to five. Mac Release and signed iOS Release passed. Both actual dependency sets match the pinned lockfiles. Actual app signatures, iOS file-sharing keys, increased-memory entitlement and 20 compiled Japanese/English entries were inspected. The Mac app does not contain the iOS-only ImageGenerationUI bundle.

The fresh production download is running. Its result will be recorded separately from the cache-backed anonymous history audit. It must start with an empty managed store and no migration candidates, request the pinned files without authentication, fully inspect their hashes, bind the acquired resource to the real image service, release it and leave zero leases. No image generation is requested.

The iOS runtime still accepts only `iPhone18,1` / `24A446` with the exact calibrated identity. Carrying forward the five unchanged component budgets is not a new device measurement. The historical ten-request measurements and unmet performance goals are unchanged. QD-07 remains an owner check on the target iPhone.

Raw execution logs, audit inventories and failed TSF runs remain private. Public records contain hashes, fixed revisions and bounded conclusions; they contain no local machine paths or credentials.
