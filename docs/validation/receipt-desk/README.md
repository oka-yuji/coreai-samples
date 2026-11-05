# Receipt Desk functional validation — 2026-11-06

[Functional plan](plan-20261106.json) · [Results](results-20261106.json) · [User guide](../../receipt-desk.md)

These are functional checks of a Mac demo, not a receipt-accuracy or latency benchmark. The plan fixes the public
JEV bundle and four synthetic images before the native app model run. Its SHA-256 is
`3ed8dce38c207e8a7946a6145f012599f1aaffa320c3d12d38647dab9f4b63da`.

Host: Mac17,6, Apple M5 Max, 128 GiB RAM; macOS 27.2 (26B123); Xcode 27.0 (27A266a), Swift 6.4
(swiftlang-6.4.0.34.1). Model operations used the shared heavy-job lock, one model process at a time. AC reported
140 W before the app checks; `pmset -g therm` reported no recorded warning. Power/thermal state was not sampled
per call. Specialization/file caches were not cleared. No deliberate warmups or repeated timing protocol were used.
Other desktop applications and compilation were not treated as a controlled performance environment.

The bundle is `okayuji/jev-27b-coreai@acc4cc835f64a75ab9ecee9d5d2e119a3bd29212`, with `decision.json` SHA-256
`2f34243a3d28500075e09d02d9f9f565a7c4b02f94a7d47b2a3caba7262d357d`. The app inventory contains 13 files,
16,064,821,387 bytes, under the public `bundle/` root. The runtime uses the published all-token asset, not the
unpublished final-token candidate. No model weights, head, tokenizer or calibration were changed.

<a id="native-app"></a>

## Native Mac app

The Release app's normal Models controls downloaded and verified JEV, then opened an empty Receipt Desk without
running OCR or inference. The first UI run used code `9aecf6b5c6f37faa3802808c282340263bb31afd`:

- The sample action read the four images with Vision and classified actual OCR text with Core AI JEV. Results were
  supplies / meals / travel / review, with amounts 1320 / 1100 / 900 / 880 and date 2026-11-06.
- OCR confidence warnings remained visible. The first three receipts were reviewed and confirmed individually;
  the total was JPY 3,320. The mixed purchase remained unconfirmed.
- Escape in the folder chooser preserved the draft and re-enabled editing/export. A subsequent export contained
  three receipt copies plus CSV and JSON. The CSV sum, JSON total and original-byte hashes matched.
- Switching Japanese to English preserved the Japanese merchant/purpose, amounts, dates, scores and timings.
- Changing the supplies receipt's purpose to `Personal gift for a friend` invalidated confirmation. A new model
  call showed supplies at 63.5% and review at 36.0% (display-rounded values). The existing maximum-score threshold
  of 0.75 routed it to review; this was not a claim that the model's top choice was review. Amount/date stayed unchanged.
- A second window was refused while the first owned the model. After closing the sessions, a new window opened
  an empty desk. This check is distinct from the owner's multiwindow acceptance.

The final app check used code `60189c5c536a0e40cf8335077c46dbc10dbfd68b`. A standard file picker selected the
Japanese sample PNG without its preset purpose. Stop during model preparation returned to idle; Process remaining
then read and classified the same input successfully as meals, date 2026-11-06, JPY 1,100. Confirming, changing the
amount to 1200, moving focus with Tab, restoring 1100 and confirming again preserved the meals category while
correctly invalidating confirmation on value changes. The final chooser's Cancel button retained the reviewed draft.
The following export contained one receipt, total JPY 1,100; CSV/JSON and original bytes were independently read
back and checked. The file-picker input, stop/retry and edit/Tab paths are therefore actual app checks, not previews.

Wide layouts, original images, field controls, category scores and the export path were visually inspected. The
preview size and segmented-control label were corrected after that inspection. The compact branch is implemented
but was not visually verified with the available window controls; narrow-layout acceptance remains open. VoiceOver
labels were inspected through accessibility state, but VoiceOver speech and the system Reduce Motion toggle were not
exercised. The code disables its custom animations when Reduce Motion is set.

<a id="models-integration"></a>

## Models and production session integration

`ReceiptModelsIntegrationTests.nativeWorkflow` passed in Release at code
`06100e4e04487a03a78e9e997dc3b24e816942f9`. It used the pinned downloaded bytes through the existing
`ManagedModelStore.migrate` path with source removal disabled, then the actual catalog, inspection, lease,
`ReceiptDemoSessions`, Store, Runner, LiveReceiptService, Vision and JEV. Only the rendering adapter returned
an empty view to expose actions to the test; the data/model path was the production implementation.

The four expected labels and amounts matched. Three confirmable samples totaled JPY 3,320. CSV bytes and each
copied original's SHA-256 matched the confirmed ledger. Closing removed the session and left zero active leases;
the temporary installed copy could then be deleted. The source manifest remained unchanged. This test does not
exercise network acquisition; the normal app run above does.

```sh
RECEIPT_MODELS_BUNDLE=/path/to/public-jev-bundle RECEIPT_MODELS_REPORT=/new/report.json swift test --package-path CoreAISamplesKit -c release --filter ReceiptModelsIntegrationTests.nativeWorkflow
```

Use the shared heavy-job lock. The test is skipped in the default suite.

## CPU boundaries and review corrections

Core and runtime checks are recorded by PRs #510 and #511. The Mac workflow adds pure state, Runner/Store and owner
checks for request/session/revision mismatches, terminal stores, edits, exact totals, overflow, picker cancellation,
operation cancellation, publication acknowledgement and replacement owners. Independent reviews additionally ran
real-source CPU reproductions outside the repository. The useful four cancellation boundaries were retained as
`ReceiptOperationBoundaryTests`.

The Mac review and UI check found three behavior issues, all corrected:

1. Cancelling `fileImporter` does not call `onCompletion`; the export ticket needed the explicit `onCancellation`
   callback. Both Escape and the Cancel button were checked in the app.
2. A classification committed just before Stop was retained but did not advance the progress count. The same
   pure-state/Store reproductions failed before and passed after the counter correction.
3. An unchanged input binding notification invalidated a category/confirmation. The reducer now ignores identical
   field values, and the inspector uses the receipt's identity and current Store values. The previous edit/Tab failure
   was repeated successfully in the final app, with a permanent CPU regression.

## Limits and owner acceptance

The synthetic examples are not held-out receipt quality data. No general Japanese receipt accuracy, tax eligibility,
minimum RAM, throughput, energy, long-running memory bound, iPhone support or real-world correctness probability
is established here. Per-receipt timings include the work of that call, including an internal JEV runtime reload;
they exclude the initial download/prepare action and are not comparable to the older controlled M4 benchmarks.

The owner's personal-receipt workflow and stop/navigation/multiwindow acceptance remain unchecked in
[#508](https://github.com/oka-yuji/coreai-samples/issues/508). The native-app support mark covers the explicitly
recorded Mac/code/bundle/input paths, not those remaining acceptance checks.
