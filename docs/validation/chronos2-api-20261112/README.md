# Chronos-2 official NYC API verification

Issue [#546](https://github.com/oka-yuji/coreai-samples/issues/546), 2026-11-12 (JST). [Results and acquisition fingerprints](results.json), [nine annual metadata captures](annual-metadata.json).

The live official API produced all 20,160 Friday-history counts expected from the independent January 2025 Parquet aggregation. Production Models, Store, sessions and service then produced all 6,720 native quantile values within `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`. The session closed with zero leases. The report retains the original API bodies privately; the public summary includes their URLs, acquisition times and SHA-256 values. No API key or model download is used.

The final native run used commit `05f8363b5e3de8f9531ebb2fb332b71bb9b795fb`, the same nine-file manifest and graph as the previous conversion. Its input is the 1,008 half-hour intervals ending at 2025-01-24 17:00 America/New_York, for 20 zones. These inputs overlap the fixed Friday test; they are not 20 additional independent forecasting-quality samples. The reported maximum error reconstructs Float32 values from the native JSON before comparing them with the pinned official reference.

## Normal Mac route

The usual Models sidebar opened the prepared Chronos model. Selecting the API source alone did not fetch data or predict. Explicit coverage lookup displayed 2017–2025; explicit history acquisition displayed zero missing intervals. For Times Sq/Theatre District at 17:00, native prediction displayed median 120 and range 102–146. Actuals remained unrequested until the separate button was used, then displayed 130 for the same interval.

Changing the requested time to 17:30 retained the previous forecast and disabled prediction until history was fetched again. The new history ended with observed 130; another explicit native prediction displayed 128 and range 109–153, matching the previously verified fixed-scene calculation for that origin. Returning to a fixed scene cleared the forecast and displayed observed 95. Leaving the demo completed normally. The source and managed model copies still matched all nine pinned hashes after app exit.

The visible date control showed January 24 in New York, while the computer-use tool serialized its absolute date value as January 25 in the device's Tokyo zone. The visual date, the explicit EST time and resulting source intervals were checked together; that tool serialization alone was not treated as a date-picker defect. This is assistant-operated Mac verification, not the user's Mac/iPhone acceptance.

## Contract corrections and failure evidence

- SDK27.2's ISO8601 parser accepted a trailing suffix/newline and normalized impossible dates and hour 24. Four catalog-entry regression cases failed before the fix. Supported UTC seconds/milliseconds must now round-trip to the complete input string; valid revisions and original-response replay still pass.
- The original annual guard rejected the captured 2019, 2020 and 2021 metadata because publication dates can precede year end, and 2021 uses `As needed`. Validation now requires a matching catalog revision and data update after the annual period closed. The special frequency also requires the official historical-publication/corrections declaration. Identity, owner, provenance, publication state and column-type checks remain. Nine pinned captures pass; zero/future publication dates, an unproven frequency and rows last updated before year end are rejected. This trusts the provider's declared annual scope; it is not an independent numerical completeness audit of every year.
- Injected abandoned UUID staging files survived the former cache pruning. Recovery, write, rename and pruning now hold the same exclusive lock, preserving other active writers that use this implementation. The regression keeps unrelated files. The 32-window/64-MiB limit applies to committed cache files after a completed save; an active staging file adds temporary space. Actual process termination or power loss was not exercised.
- A preparation-only actuals request received HTTP503 once, then succeeded on a new attempt and matched all 80 pinned two-hour counts. The failure was retained separately from the successful retry. It was not labelled unpublished or replaced with a cached forecast.

## Verification scope

CPU checks cover metadata trust and malformed rows, 0 versus missing/error, both autumn folds, spring slots, year/chunk boundaries, bounded receiving, redirects, cancellation, generations, cache corruption, source-byte reconstruction, old tickets and unchanged prior display. Original-response replay independently passes all 20,160 history counts and 80 supplied actual counts. A separate replay checks the nine annual metadata captures. These explicit fixture checks are skipped when their environment variables are absent.

All ten String Catalogs contain 1,426 keys with no missing Japanese translations or placeholder mismatch. The 80 compiler-extracted static ChronosUI keys are present in its catalog. Mac Release and signed iOS Release builds passed; the actual iOS artifact passes signature verification and contains both file-sharing keys. Physical iPhone inference and UI acceptance remain open.

Host: Apple M5 Max / 137,438,953,472 B RAM, macOS27.2.1 (26B131), Xcode27.2 (27B52), Swift6.4.2, SDK27.2. Native tests use Debug; GUI uses Release. App builds ran concurrently with numerical validation. Power, thermal conditions, memory, controlled latency and actual delegate selection were not profiled. The existing GPU preference and a successful numerical comparison do not establish a performance benchmark.

Acceptance mapping: A1 has the schema failures and nine metadata captures; A2 has time/missingness and zero tests; A3 has stale/cancel/retain-display tests and the stated Mac time-change check; A4 has exact response/snapshot/hash replay and explicit native-only API mode; A5 has the live production path; A6 has the host builds, localization and recorded conditions. Persistent forecast records and whole-parent final integration are the responsibility of #547.
