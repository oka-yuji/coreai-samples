# Chronos-2 public download verification

Issue [#561](https://github.com/oka-yuji/coreai-samples/issues/561) publishes the existing single-series FP32 model and connects it to Models download. The model, all nine bundle files, original manifest and computation are unchanged. This record uses the Japan date, 2026-11-13; the machine-readable timestamps retain UTC.

The sanitized [results](results.json) join the release audit with a separate fresh application download and native comparison. Product code is `cf53f2932ccf8f80ffd4b067367f8e66b075fb2e`; the added opt-in test and its execution use `9c7da3d0de00ef86f8ca1235385a254bd82f7982`. The latter commit changes only the test and its execution instructions.

## Published distribution

[Chronos-2 Core AI FP32](https://huggingface.co/okayuji/chronos-2-coreai-fp32) is public. Release tag `v1.0.1` points to `f422aa3a7c1173566c99fbfceb97e874b0121dab`. Models pins `d3ead488046498faa8b3b5103b1923e52a313d0e` (`v1.0.0`) and selects `bundle/`. The later commit only clarifies the model card; all nine downloaded files are identical. No tag was moved.

| Scope | Files | Bytes |
|---|---:|---:|
| Complete published payload | 16 | 478,899,086 |
| Runtime bundle selected by Models | 9 | 478,864,956 |
| Distribution documents, attribution and CLI example outside `bundle/` | 7 | 34,130 |

The [pinned official model card](https://huggingface.co/amazon/chronos-2/blob/29ec3766d36d6f73f0696f85560a422f50e8498c/README.md) declares Apache-2.0. The packaged license matches the [official implementation license](https://github.com/amazon-science/chronos-forecasting/blob/v2.3.2/LICENSE), and the conversion NOTICE preserves the upstream copyright sentence and identifies the changes. The upstream HF snapshot has no standalone LICENSE file; that lookup's 404 remains distinct from the successful license-text comparison. The unchanged `SOURCE_README.md` describes upstream capabilities, while this conversion exposes only its documented 16-step, single-series ABI.

## Information audit and anonymous acquisition

Local payload inspection, authenticated full-history inspection and the final gate passed before publication. The anonymous full-history inspection passed after publication: three commits, sixteen unique objects and three refs, with pull-request-ref discovery included. Findings, exceptions and unread/error objects were all zero.

The history audit fully rescanned existing cached objects against anonymously observed public object identities and lengths. It did not perform a fresh model GET. Fresh acquisition was established separately:

| Check | Revision and route | Result |
|---|---|---|
| Distribution files | Seven files outside `bundle/`, at `f422aa3a…`; direct anonymous GET | Every size and SHA-256 matched; no model bodies read |
| Application bundle | Nine files inside `bundle/`, at `d3ead488…`; production `ArtifactTransfer.load` | All nine pinned requests returned HTTP 200; every acquired size and SHA-256 matched |
| Existing local resources | Same manifest identity with a new optional locator | CPU storage/lease regressions retain the existing registration and avoid unnecessary download |

The anonymous history and distribution-file checks removed owner credentials and disabled netrc/proxy authentication and cookies. The application check used the production ephemeral URLSession, with cookies, credential storage and URL cache disabled; it supplied no explicit request headers or authenticated HF client. Raw authentication traces and local environment paths are not published here.

## Identity and acquisition

The 40-character HF commit identifies the transfer source. The manifest SHA-256 `786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572` continues to identify the trusted `localManifest` resource, directory, ledger entry and leases. HTTP requests use the HF commit, not the 64-character manifest digest. Both contracts are validated when both are present. Adding a valid locator does not change the resource identity or force an existing import to download again.

Models offers **Download model** and **Import model**. Progress and cancellation labels follow the operation actually running. A hybrid resource is labelled **Model ready** without inventing how it was acquired. Information shows the repository revision and manifest identity. Self-test acquisition downloads when no import folder is selected and honors an explicitly selected folder.

CPU tests cover all source combinations, malformed locators and inventories, unchanged imported identities and leases, transfer URL revision, visibility restrictions, operation labels and self-test routing. Import failures retain their staging cleanup in the original import effect. Download validation retains reusable checked staging. Existing local-only image assets retain their license information; this change does not publish or replace those assets.

## Fresh download through native prediction

`ChronosDownloadNativeTests.freshModelsDownload` started with a new output directory, an absent Models directory and zero migration candidates. The production transfer requested all nine fixed URLs and validated their bytes before opening a model session. Acquisition itself left both the model session and demo Store absent, so it did not run forecasts automatically.

The drawing-only adapter then exercised production Models, Store, sessions and service on the Mac. The complete result was:

| Condition | Result |
|---|---|
| Reference | SHA-256 `8f894aaac769294a3e6843225e6f86519f85090d747b16d94a288fb987661470` |
| Workload | Three fixed January 2025 scenes × twenty zones; origin step 0, 1,008 input observations per series |
| Output comparison | Sixty cases × twenty-one quantiles × sixteen steps = 20,160 raw values |
| Frozen criterion | `abs(actual-reference) <= 1e-4 + 1e-4 * abs(reference)` |
| Maximum absolute error | `0.0002288818359375`; every scalar passed its absolute-plus-relative threshold |
| Saved evidence | All 20,160 persisted raw values were independently compared again |
| Resource cleanup | Zero leases after close; original nine-file bundle hashes unchanged |

These are the same fixed scenes used by the earlier demo check. They do not add independent held-out forecasting data. The raw values are compared before the UI's nonnegative count presentation.

The single test took 50.765 seconds; its wrapper took 64.47399308299646 seconds. The test interval includes hashing, downloading, validation, native setup, all forecasts, cleanup and evidence writing. The wrapper also includes SwiftPM build and orchestration. Neither duration is a model latency or throughput measurement. No controlled performance, memory, power or thermal benchmark was performed. The log retains `Failed to import MPS module`; the subsequent native comparison passed, and the actual delegate was not profiled.

Host identification, captured after this run on the same machine, was Apple M5 Max with 137,438,953,472 bytes RAM, macOS 27.2.1 (26B131), Xcode 27.2 (27B52), Swift 6.4.2 and SDK 27.2. The native test used a Debug package build and the shared heavy-job lock. The wrapper waited for its child process to finish and archived only its own lock after checking owner and inode.

Before publication, a separate authenticated CLI consumer at code `3b3cfe2f70fa6f2b4fa7d9a8aeed51211f890dd3` passed eighteen reference cases and 6,048 values plus reload for `v1.0.0`. The `v1.0.1` CLI example compared 336 values with maximum absolute difference 0.0. Those checks retain their private/authenticated scope; the new application GET provides the fresh anonymous bundle evidence.

## Builds and remaining acceptance

All eight required repository checks passed for the product commit. The later test-only commit passed the full default package suite, lint, format and privacy checks; the network/model test is skipped by default. Mac Release and the corrected signed iOS Release build passed. The actual iOS signature, file-sharing keys and embedded download locator/inventory were inspected. All ten String Catalogs, containing 1,465 keys, had no missing translations or placeholder mismatches. The seven changed keys matched the English and Japanese compiled resources on both platforms, for twenty-eight checks.

The [existing Mac GUI evidence](../chronos2-demo-20261112/README.md) remains attached to its original code and import route. This new drawing-adapter check does not establish normal GUI download operation. The 16-step ABI and verified January 2025 NYC API range are unchanged.

- [x] CD-04: release audit, anonymous history inspection and all sixteen files covered by fresh anonymous acquisition.
- [x] CD-05: fresh production Models download, native comparison, source preservation and zero leases.
- [ ] CD-07: the user confirms Models download and taxi prediction in the normal Mac UI and on the target iPhone.

Physical iPhone download/inference, user acceptance, held-out forecasting quality and controlled performance/memory measurements remain open. Repository checks and app builds do not close those items. Opposite-AI review and PR integration are tracked in the implementation PR rather than inferred from these execution results.

## Retained preparation failures

The first independent harness failed to compile because it supplied an extra argument label. After that harness correction, five tests ran and three failed semantically because the old catalog rejected combined manifest/download definitions. These results remain distinct. A later worker test compile failed on nested `#require` expansion and was corrected by checking the URL in a separate local value.

The first protected fixture omitted demo links, and the SF worker broadened acquisition to asset-only models to satisfy it. Parent review rejected that extra behavior, restored the existing visibility contract and corrected the fixture, with a no-demo regression. The original SF task, result and failure logs were preserved. TukuruSF run `8832b70171db4f8491aee3df4955a2d9` produced a verified candidate; operator correction and subsequent host verification were recorded separately.

The first fresh-worktree iOS build succeeded but produced an unsigned artifact because the ignored local signing configuration was absent. The existing configuration was copied without changing its source, then the signed artifact was checked separately. The native test's first borrowed AsyncStream iterator call failed isolation checking; `next(isolation: #isolation)` corrected it after checking SE-0421 and the SDK. Its successful compilation was recorded before the later actual network/native run.

## Reproduce the opt-in check

Use the shared heavy-job lock described in `AGENTS.md`. Supply the original known bundle only for before/after hash checks; acquisition must still use the network. Use a new output path with an existing parent directory and the fixed sixty-case reference.

```sh
COREAI_CHRONOS2_DOWNLOAD_OUTPUT=/new/output \
COREAI_CHRONOS2_DOWNLOAD_SOURCE=/path/to/original-bundle \
COREAI_CHRONOS2_DOWNLOAD_REFERENCE=/path/to/fixed-scenes-reference.json \
swift test --package-path CoreAISamplesKit --filter ChronosDownloadNativeTests.freshModelsDownload
```

The output retains `checkpoint.json`, `acquisition.json`, the managed files and `native.json`. Existing output paths are rejected. A default skip or a run without the final native evidence is not a pass.
