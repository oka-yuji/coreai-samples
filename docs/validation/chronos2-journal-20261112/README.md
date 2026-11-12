# Chronos-2 saved forecast verification

This record covers Issue [#547](https://github.com/oka-yuji/coreai-samples/issues/547). Successful on-device predictions now retain their complete input, source evidence, model identity and raw output. Later actual checks append separate files. A preview creates no prediction record.

Initial native and normal GUI evidence at `10a262c` and the locale correction at `9b98f01` are retained. Independent Claude review then found the inactive-capture boundary and hardware-identifier gaps described below. Required eight checks passed after those corrections at `cffd664`; a new native J4 at `48adfc2` and independent arithmetic checks passed. `815830f` only wraps the final check-error UI string. Exact final build, review and merge identities are in the PR; this file does not claim the user has accepted the app.

## Contract and CPU evidence

| Condition | Evidence |
| --- | --- |
| J1: immutable records and validated reload | `ChronosJournalFilesTests.immutable/forged`, `ChronosJournalBoundaryTests.apiEvidence`, Date and bounds suites; original input bytes, all 21×16 outputs per zone, actual model identity and earlier checks are retained |
| J2: failure, cancellation and competing owners | `failedPublication/stagedOwners/owners`, `initialLockRace`, JournalRunner/Feature tests, real Models/session publication tests; delayed stop, owner withdrawal before Task cancellation, same-ID retry and competing windows |
| J3: comparison states and arithmetic | `ChronosJournalStatusTests.status/correspondence`, missing/zero/unverified/ambiguous cases; nonnegative unrounded median, comparable pairs only, no WAPE for zero actual sum |
| J4: production native route | Models → production Store/service/session → live verified API → native prediction → journal → actual check → reload → stop, with a drawing-only test adapter |
| J5: repository checks | Eight required commands passed, signed iOS artifact verified, ten catalogs/1,459 keys, 113 Chronos catalog entries and 69 compiler-extracted static keys checked; ModelReports checks are recorded with the final documentation commit |
| J6: workflow evidence | [TukuruSF run ledger and improvement analysis](../chronos2-tsf-20261112/README.md), plus private exact review/acceptance receipts |
| J7: user operation | Pending on the user's Mac and target physical iPhone |

The shared `Store` template is unchanged. Files are prepared on the Files actor; a short MainActor attempt observes the actual owner/session/journal permission sources. Synchronous invalidation revokes publication before a later cancellation effect can run. The final permission check and rename share a mutex. File IO does not run on MainActor and no await occurs under the file lock. An unrelated observation change retries the same prepared UUID, timestamps and bytes without refetching the API.

## Native result and independent arithmetic

The origin is January 24, 2025 at 17:00 in New York. The production client validated 20,160 historical counts. All 6,720 native outputs satisfied `abs(actual - expected) <= 1e-4 + 1e-4 * abs(expected)` against the pinned official reference. Maximum absolute error was `0.000152587890625`; a value above absolute tolerance alone can pass the combined gate.

The saved record's eighty comparable actuals sum to 9,940. A separate Python calculation from the saved native Float32 outputs produced MAE `13.965053582191468` and WAPE `11.239479744218485%`, agreeing with the saved check within `1e-10`. Envelope/payload identities, input hash, original prediction hash and zero leases after close also passed. Recomputing from the official reference gives slightly different metrics because its outputs differ within the frozen native tolerance; those metrics are not substituted for this saved prediction.

These are twenty zones × four half-hours from one fixed demonstration origin. They establish the comparison implementation, not held-out forecasting accuracy, calibrated quantile coverage or general service performance. [Machine-readable results](results.json) retain the identities and limits. Raw HTTP bodies, full saved records and device-specific logs remain private.

## Normal Mac workflow

The normal Models/sidebar route saved fixed-scene and API predictions, appended checks without changing prior bytes, quit after dismissing the sheet, restored both records after relaunch and checked actuals without another model run. The detail showed 80/80, MAE 13.97 and WAPE 11.24%. Original and managed model copies remained unchanged. The final `815830f` application also saved another native record, reopened the existing records and appended an 80/80 actual check in English. All five pre-existing JSON files stayed unchanged, both model copies matched their nine pinned hashes, and Done followed by Quit exited normally. [GUI evidence](gui.json) separates that workflow from the later English-sheet correction.

The earlier report incorrectly described all 113 catalog-derived generated-symbol keys as the number of keys extracted from handwritten source. Including GeneratedStringSymbols output reproduces that total; the original collection command was not recovered, so its exact cause is not asserted. The final JSON stringsdata contains 69 unique static keys across seven source files, all present in the catalog. Dynamic failure/status keys were inspected through their callers; the remaining catalog entries also include older entries, so they are not all described as dynamic call sites. Including generated keys does not itself invalidate a missing-key check: `(source ∪ catalog) − catalog = source − catalog`. The numerical correction is retained separately from the successful catalog/GUI checks.

The first locale check failed: an English parent opened a Japanese sheet. Explicitly passing the parent locale fixed the reproduced case. Both English visual layout and accessibility labels passed; stored files remained unchanged. Catalog completeness alone had not caught this presentation boundary.

During the first workflow, Quit while the history sheet was visible did not terminate through CUA. Dismissing it with Done and then quitting worked. Source review found no journal-specific Quit handler, but the shared termination delegate and its pending work were not observed, so the cause remains unresolved. This observation is retained for user acceptance, not described as an intended macOS rule.

## Retained failures and corrections

- Initial native J4 at `00b6233` passed all raw-output comparisons, then failed exact snapshot equality after save. Millisecond-since-1970 JSON lost one reference-clock Date unit for production-created fractional dates. Journal envelope `/2` uses Foundation reference-clock seconds; the API canonical-byte contract is unchanged. Red/green fractional-date tests and both later native trials passed.
- Finite, aligned extreme dates and decoded extreme counts could reach calendar loops or unsafe integer conversion. The supported date/count domains are now validated first. Four safe constructor cases failed before the correction; rehashed corrupt records and `Int.min/max` reject afterward. No original full-record trap/hang was executed.
- An owner could withdraw permission before the Store's cancellation effect ran. A real-file regression reproduced a late publication. The repair binds invalidation to actual Models/session/journal state and retains unrelated-update retry controls.
- The publication SF run failed its protected verification when five existing capacity-dependent tests saw zero important-usage capacity. The unchanged protected verifier passed all 370 tests on the host for the exact candidate/specification. The SF failure is preserved separately; no test was disabled to obtain that host receipt.
- A separate initial `.lock` creation race reproduced with relative `openat(O_CREAT)` in Swift and independent C/Python probes. Existing-only open, exclusive create and at most one existing-only reopen resolved it while preserving no-follow/type/link/lock checks. A 32-directory Swift regression, ten isolated owner-suite repeats and 100 independently synchronized two-owner directories passed. The C probe's 2,000 opens target one new directory, not 2,000 independent initialization races. No universal OS defect or production failure rate is inferred.

## Independent review follow-up

A native result can remain valid while the journal is temporarily inactive: Models stops its session on background, whereas the journal suspends for inactive as well. The initial active guard discarded such a result. Pure Feature and production Runner/Store regressions failed before the fix. The journal now retains native captures until resuming, starts no publication while inactive and still rejects ended sessions. The same regressions and the broader publication suite passed. This is CPU lifecycle evidence, not an executed iPhone Control Center test.

The initial hardware whitelist dropped MacBookPro, MacBookAir, Macmini and iMac identifiers to unknown and rejected those stored values. [Apple's model list](https://support.apple.com/en-us/108052) confirms MacBookPro18,3 and related identifiers. Acquisition and reload now share a bounded whole-string family check; four failing Mac cases changed to pass, with eight accepted and five rejected inputs. Identifier preservation does not establish inference compatibility on those devices.

Save-failure text now asks the user to check storage access as well as capacity. Both prediction and actual-check messages use that wording. An error still retains the original pending save and requires explicit retry; it is not silently dropped. The staging test's release controller now runs on MainActor, while file IO remains on the Files actor. An attempted strict cooperative-pool microprobe terminated before its body on this SDK; it is retained as an unsuccessful probe, not proof of either version's scheduling behavior.

The remaining review observations define limits, rather than measured failures in normal use: an injected custom service could violate the production service's request/batch correspondence; unexpected non-journal entries inside a checks folder are rejected; listing validates every saved record and has no large-history latency benchmark; an unresolved save error retains the FIFO and can accumulate subsequent captures. There is no automatic pruning, external-folder repair or guarantee against arbitrary injected services. The normal production path and explicit retry tests are the accepted scope. The sheet-open Quit observation above remains unresolved.

## Reproduction and limits

Use the [journal command and workflow](../../chronos2.md#saved-forecasts-and-actual-checks). The explicit native test requires the shared model lock and a new output directory. Default tests skip models and live API access. API acceptance remains the independently verified January 2025 revision/range; other revisions or dates are not converted to zero.

Mac Release, signed iOS Release, actual signature/file-sharing settings and embedded reference bytes passed. Physical iPhone inference, memory/latency, actual GPU delegate selection and the user's acceptance remain unmeasured. The logged MPS import warning is retained. No model weights are published by this change.
