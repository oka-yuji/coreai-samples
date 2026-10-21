# Qwen Image local bundle import

This implements the acquisition contract from [#263](https://github.com/oka-yuji/coreai-samples/issues/263), QMI practical v2. The production allowlist is currently empty: a reviewed bundle from #216 must supply its real manifest hash and complete file inventory before Qwen appears in the ordinary Models catalog. CPU fixtures exercise the contract without loading model weights. They do not complete the real-device acceptance checks.

## Trust and storage

`ModelArtifact.localBundle` identifies a known profile by the SHA-256 of the saved `bundle.json` bytes. The manifest is listed in the catalog inventory, but not recursively in its own `assets` inventory. This source is distinct from a Hugging Face repository and pinned revision. Existing format-one ledger entries without `sourceKind` retain their Hugging Face identity and directory layout.

Models copies a selected folder into a unique staging directory while retaining its security-scoped access. Each source path component is opened without following symbolic links; the final descriptor must be a regular file. Copying checks size, SHA-256 and cancellation in bounded chunks. The selected folder must not overlap managed storage.

Before promotion, the store verifies the exact file set, pinned manifest, source revision, numeric and quality acceptance, and the contracts of text, vision, denoiser, VAE encoder and VAE decoder. A bundle's own `verified` flag does not establish trust. Extra files, missing files, links and mismatched profiles are rejected. LICENSE and NOTICE remain in the copied inventory.

Copying and promotion belong to one resource operation. The copied notification only changes the visible phase to validation; it does not create a second task. Cancellation or Store destruction between those phases therefore still reaches staging cleanup. A stop acknowledgement waits for that operation and re-inspects the installed resource. Failed replacement also re-inspects before restoring ready, preserving the failure for the UI and diagnostic report. Active leases prevent replacement and deletion.

## Catalog and diagnostics

`IOSDemoCatalog.make(localImageArtifacts:)` supports injecting trusted definitions for CPU verification. Ordinary app construction uses its built-in allowlist. Registered adapter IDs must be unique and known; missing adapters require an explicit unconnected-demo declaration. An unconnected image-generation demo can explain and acquire its resource without invoking a model adapter.

The Models self-test accepts `--import-local-bundle` with a Documents-relative folder and requires exactly one local artifact selected with `--acquire`. Its report distinguishes source kind, manifest identity, acquisition failure and the inspection of any preserved installation. Actual device commands and bundle identity will be recorded when #216 supplies the reviewed artifact.

```sh
swift test --package-path CoreAISamplesKit \
  --filter 'ModelLocalSourceTests|ModelLocalImportFeatureTests|LocalImageImportTests|LocalImportRunnerTests|LocalImageCatalogTests|ManagedModelStoreTests.legacySourceLedger'
```

## Acceptance boundaries

| Condition | Evidence and remaining work |
|---|---|
| QMI-01 | CPU tests cover source identity, a legacy ledger without rewriting, local promotion and restart inspection. |
| QMI-02 | CPU tests reject unknown hashes, self-reported acceptance and mismatched component/profile contracts. The actual #216 inventory is pending. |
| QMI-03 | CPU tests cover altered/missing/extra files, links, overlapping paths, capacity, cancellation, Store destruction and preservation of the previous resource. |
| QMI-04 | CPU tests cover active leases and ledger-write failure during promotion. |
| QMI-05 | Pending: import the real bundle through ordinary Models on the target iPhone, restart, inspect and delete. |
| QMI-06 | The ordinary catalog factory and acquisition harness pass with synthetic files and zero adapter callbacks. Real-bundle normal launch and UI verification are pending. |

This change does not establish image generation, output quality, device memory use or performance. Those remain in #216 and #218–#221.
