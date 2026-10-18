# Shared model library contracts

`ModelLibraryFeature` provides the catalog and state machine for model-first demo navigation. It does not download files, create inference services, or change the existing app entry points. The iOS integration consumes these contracts separately.

## Catalog

`ModelDemoDefinition` requires an ID, title, symbol name, summary, explanation, platform set, and visibility. `DemoDescriptor` is the value implementation. A `LibraryModel` declares one or more converted artifacts (variants), its default artifact, and its explicit `ModelDemoLink` entries. Artifact IDs are unique across the catalog, so several models can link the same demo ID without sharing installations.

A `ModelArtifact` has a display title, contract, platform set, visibility, optional distribution, optional upstream source, and the complete set of files a device needs. `ModelDownload` identifies an owner/repository, a lowercase 40-character revision, and a bundle-relative root. `ModelArtifactFile` lists a relative path, byte count, and lowercase SHA-256. The catalog rejects empty, absolute, `.` or `..` path components, malformed checksums, duplicate paths, and a configured distribution without files. An unconfigured artifact may omit files. A missing distribution or an unpinned revision cannot become executable through inspection of pre-existing files.

The distribution revision pins the converted repository. The upstream revision (`ModelUpstream`) identifies the source model. They are separate values and are never compared with each other.

Visibility intersects the model, artifact, demo, and link publication with their platform sets. A model is listed while at least one variant is visible; `ModelSection.artifacts` holds those variants. Search returns model sections, including when only a demo title matches.

## State and identity

`ModelLibraryState` receives a `LibraryEnvironment`: platform, audience, the demo IDs whose adapters are connected, and capability states (`available`, `undetermined`, `denied`, `unsupported`). A capability absent from the environment is `undetermined`. Pure logic never reads the device, OS, or permission state. The application also injects the state `owner`.

Requests, sessions, and stop operations use `LibraryTicket` (owner and number). A ticket from another owner, such as another scene or a previous launch, never matches, even with the same number.

Installations are keyed by artifact ID, so variants of one model have independent states. `selectedArtifacts` starts with each model's default. `selectedArtifact(for:)` returns nil when the selection is not visible. It never substitutes another variant, and `access(to:)` then returns `.unavailable` until the user selects a visible one. `selectArtifact` changes the selection immediately when that model has no demo session. Otherwise it stops the session through the same barrier as navigation and applies the new variant after `demoStopped`.

`ValidatedModelResource` carries the model and artifact IDs, contract, distribution revision, upstream revision, directory, and size. `matches(model:artifact:)` requires all identity fields to equal the catalog entry. `ModelArtifactIdentity` (model, artifact, contract, distribution revision) is available from the resource and from `ModelDemoSession.identity` so that results can record which variant produced them.

## Execution

Opening an unavailable model/demo combination is rejected. A supported demo whose artifact is missing, or whose adapter is not connected, only selects its explanation (`.preview`). `isConnected(_:)` lets the UI show that execution is not connected.

`permits(_:selection:)` is the execution gate. The active session ticket, connected adapter, selected artifact, validated resource, catalog eligibility, pinned distribution, and required capabilities must all hold. A `denied` or `unsupported` required capability blocks execution. `available` and `undetermined` allow the demo's explicit Start to request permission before any capture, recording, or inference. Optional capabilities never block execution. `capabilityChanged` only updates the environment and never removes a model or demo from the list.

Completing an inspection, transfer, or variant change does not issue an inference effect. `ValidatedModelResource` is the boundary value returned by the validator. Construct it only after checking the actual bundle and its identity. Matching this value in pure logic is not a substitute for file and checksum validation in the I/O implementation.

## Effect ownership

The existing generic `Store` owns tasks. The application supplies one runner that owns the actual transfer and demo resources. Resource actions carry the artifact ID and request ticket, and effects carry the model and artifact. The runner must preserve these values when sending progress, completion, or failure.

- `inspect` examines existing resources without loading an inference model. A refresh skips the artifact used by the active session.
- `download` writes an incomplete transfer to a staging location. `downloaded` requests a distinct validation operation; only `validated` can make the resource ready.
- Cancelling a resource emits both a cancellation command and `stopTransfer`. Cancellation is cooperative: `transferStopped` must arrive only after the previous I/O has terminated and its staging resources are no longer being modified. Cancellation acknowledgement is an infallible join contract, not a success report for the download. The state rejects late completion and prevents another transfer while stopping.
- `delete` is rejected while a demo of the same model owns a session or while a stop is pending. A deletion failure invalidates the cached availability until a subsequent inspection verifies the remaining files.
- `stopDemo` carries both the old session and a unique stop ticket. The runner must finish inference, camera/audio cleanup, and resource release before sending the matching `demoStopped`. `demoStopFailed` keeps navigation, execution, deletion, and variant changes blocked; `retryDemoStop` uses a new ticket. An old stop result cannot complete a later attempt.

Demo data remains owned by its feature. The library retains definitions, the selected destination, and the selected variants across transfer and stop failures. It never deletes input, history, or generated results. Keeping those values when a demo runner releases runtime resources is the integration's responsibility.

## Storage

`ModelLibraryStorage` provides `ManagedModelStore`, an actor that owns the managed model area. The application passes `Application Support/<bundle ID>/ModelLibrary` as the root; it is connected to the app in #192. Hashing, copying, and enumeration run on the actor, never on the main actor, and check cancellation between files and between 1 MiB reads.

| Path under the root | Contents |
|---|---|
| `artifacts/<artifact ID>/<distribution revision>/` | Verified bundles, laid out by the definition's relative file paths. Revisions are separate directories. |
| `staging/<artifact ID>/<distribution revision>/` | Incomplete files of that revision. `stagingDirectory(for:artifact:)` always returns this directory, creating it when needed and keeping the files already in it. One transfer per artifact runs at a time. |
| `trash/<UUID>` | The store's own holding area for a replaced or deleted bundle. The next promotion or deletion empties it. |
| `ledger.json` | Format 1. One entry per artifact and revision: model ID, artifact ID, contract, distribution revision, upstream revision, size, installation time, and generation. It is written to a temporary file and renamed atomically. |

The root itself may be a symbolic link. Below it, an existing `artifacts` or `staging` folder, artifact folder, or revision folder that is a symbolic link is rejected with `symbolicLink`, and an existing one must resolve inside the root (`pathEscape` otherwise). Every operation checks this before reading, moving, or deleting anything.

`inspect(model:artifact:)` returns `.missing` without reading files when the distribution is absent or unpinned. Otherwise it compares the artifact directory with the pinned definition, whether or not the ledger lists it. Every listed file must be a regular file with the listed size and SHA-256. Any other file or directory, a symbolic link, a folder that cannot be listed (`unreadableDirectory`), and an empty, absolute, `.` or `..` path component are rejected. Folders are listed with `readdir`, so hidden files, including AppleDouble `._` names that `FileManager` enumeration omits, count as unlisted files. The contract check follows:

- `gliner-gl4-*`: the definition includes `tokenizer/tokenizer.json` or `tokenizer.json`, `GLiNERAsset(directory:)` succeeds, and the `model` and `revision` in `contract.json` equal the upstream.
- `embeddinggemma2-fp16-safe-s128-v70` and `embeddinggemma2-int4-int8-ptq-s128-v70`: `EmbeddingBundle(directory:)` succeeds with the matching precision profile, and the manifest revision equals the upstream revision.
- Any other contract is rejected.

A passing bundle updates its ledger entry and returns `.ready`. The resource's `revision` is the distribution revision, and `upstreamRevision` is the upstream revision. A failing bundle returns `.invalid` with the original error (`ManagedModelStoreError`, `GLiNERAssetError`, `EmbeddingError`, or a file system error), including when only the ledger lists it. A ledger entry alone never makes a resource ready. Inspection and promotion ignore a ledger that cannot be decoded and replace it on their next write; deletion refuses it with `ledgerCorrupted`. Inspection does not connect to the distribution or load Core AI. Cancellation is thrown, not reported as an invalid bundle.

`promote(staging:model:artifact:)` accepts only the revision's staging directory or a directory below it. It runs the inspection checks, so an unlisted file such as a leftover partial download is rejected. It then moves an existing bundle of the same revision to the trash, moves the staging directory into place, records a new generation, and removes the trash and anything left in the staging directory. If a move or the ledger write fails, the directory returns to staging, the previous bundle returns from the trash, and the error is thrown. Other revisions are never touched. A process exit between the two moves leaves that revision missing rather than ready; its previous bundle remains in the trash until the next promotion or deletion removes it.

`migrate(candidates:model:artifact:)` requires every listed file of a candidate folder to be a regular file with the listed size and SHA-256, and rejects a symbolic link anywhere in the folder. Unlisted files, such as `SHA256SUMS`, verification records, another `.aimodel`, or `.DS_Store`, are ignored and never copied. A candidate that resolves to the root or a location inside it is `undetermined` (`pathEscape`) and is not read or changed. A candidate with a missing or changed file, a symbolic link, or an unsafe path is `undetermined` and is not registered. For a matching candidate, a capacity check runs first; the staging directory is then emptied, only the listed files are copied (never moved), and the copy is promoted, so the contract checks run on exactly the listed files. Failures are reported as `failed` and remove the copy; a capacity failure leaves the staging directory unchanged. Once the artifact is ready, the remaining candidates and later runs are `skipped`, so repeated runs create no duplicates. Originals are never modified or deleted, and `promote` refuses a folder outside the staging area.

`lease(identity:owner:)` returns a `ResourceLease`. `release()` is idempotent, and `activeLeases(identity:)` counts the remaining leases. Deleting or promoting a leased identity fails with `inUse`. `delete(identity:)` also requires a ledger entry with the same model, artifact, contract, and distribution revision (`notInstalled` otherwise). It removes the ledger entry, moves the bundle to the trash and deletes it, deletes that revision's staging area, and returns the bytes of those files, which `deletionSize(identity:)` reports beforehand. A failure after the ledger entry is removed is thrown. The next inspection verifies what remains: a moved or partially removed bundle is missing or invalid, and only a complete bundle that still passes every check becomes ready again. Promotion and deletion only change the store's own folders (that revision's bundle, its staging directory, and the trash). They never change the other artifacts recorded in the ledger, originals outside the staging directory, user data such as inputs and saved results, or the Core AI specialization cache.

Every ledger write that registers a bundle receives a larger generation, which never decreases while the process runs. The ledger stores the generation of each entry; leases exist only in memory. `ValidatedModelResource` has no generation field, so `generation(of:)` exposes the recorded value for the execution check in #192.

## Verification boundary

Issue #171 covers the original pure-logic contract. Issue #189 adds variants, file sets, revision identity, tickets, the environment, capabilities, and adapter connection. The tests cover catalog intersections, search ownership, inspection versus execution, identity and revision mismatches, pinned distributions, progress, stale and foreign completions, cancellation, deletion, variant changes, and stop/retry ordering. A shared-demo fixture and a ten-demo fixture use the same registration path. GPU inference and real model distribution are outside these tests.

Issue #190 adds `ModelLibraryStorageTests`, which build small GLiNER and EmbeddingGemma 2 fixtures (both precision profiles) in temporary directories, with definitions pinned to their actual SHA-256 values. They cover acceptance and rejection, interrupted and cancelled operations, capacity and ledger failures, migration, leases, deletion, and external replacement without downloading files or loading models.

The test fixtures use `owner/fixture` with fixed revisions and checksums. They are test inputs, not real distributions. `IOSDemoCatalog` remains unconfigured, without files or revisions, until the distribution definitions are adopted.

The initial independent review of #171 found an inspection path that bypassed the missing/invalid distribution rule. Validation completion and the execution gate now reject that path, with regression cases for both missing metadata and an unpinned revision.
