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

## Verification boundary

Issue #171 covers the original pure-logic contract. Issue #189 adds variants, file sets, revision identity, tickets, the environment, capabilities, and adapter connection. The tests cover catalog intersections, search ownership, inspection versus execution, identity and revision mismatches, pinned distributions, progress, stale and foreign completions, cancellation, deletion, variant changes, and stop/retry ordering. A shared-demo fixture and a ten-demo fixture use the same registration path. GPU inference and real model distribution are outside these tests.

The test fixtures use `owner/fixture` with fixed revisions and checksums. They are test inputs, not real distributions. `IOSDemoCatalog` remains unconfigured, without files or revisions, until the distribution definitions are adopted.

The initial independent review of #171 found an inspection path that bypassed the missing/invalid distribution rule. Validation completion and the execution gate now reject that path, with regression cases for both missing metadata and an unpinned revision.
