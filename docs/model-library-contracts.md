# Shared model library contracts

`ModelLibraryFeature` provides the catalog and state machine for model-first demo navigation. It does not download files, create inference services, or change the existing app entry points. The iOS integration consumes these contracts separately.

## Catalog and execution

`ModelDemoDefinition` requires an ID, title, symbol name, summary, explanation, platform set, and visibility. `DemoDescriptor` is the value implementation. A `LibraryModel` declares its converted artifact and its explicit `ModelDemoLink` entries. Model, artifact, demo, and link visibility must all allow the current audience; platform support is intersected across model, demo, and link.

Search returns model sections, including when only a demo title matches. Registered artifacts have stable identities and contracts. A distribution must identify an owner/repository and a lowercase 40-character revision. Missing distribution metadata or an invalid revision cannot become executable through inspection of pre-existing files.

Opening an unavailable model/demo combination is rejected. Opening a supported but missing model only selects its explanation. `permits(_:selection:)` is the execution gate: the active session, selection, validated resource, catalog eligibility, and distribution metadata must match. Completing a transfer or inspection does not issue an inference effect.

`ValidatedModelResource` is the boundary value returned by the validator. Construct it only after checking the actual bundle and its identity. Matching this value in pure logic is not a substitute for file and checksum validation in the I/O implementation.

## Effect ownership

The existing generic `Store` owns tasks. The application supplies one runner that owns the actual transfer and demo resources. Resource actions carry the model ID and request ID. The runner must preserve these values when sending progress, completion, or failure.

- `inspect` examines existing resources without loading an inference model.
- `download` writes an incomplete transfer to a staging location. `downloaded` requests a distinct validation operation; only `validated` can make the resource ready.
- Cancelling a resource emits both a cancellation command and `stopTransfer`. Cancellation is cooperative: `transferStopped` must arrive only after the previous I/O has terminated and its staging resources are no longer being modified. Cancellation acknowledgement is an infallible join contract, not a success report for the download. The state rejects late completion and prevents another transfer while stopping.
- `delete` is rejected while a demo owns the model. A deletion failure invalidates the cached availability until a subsequent inspection verifies the remaining files.
- `stopDemo` carries both the old session and a unique stop request ID. The runner must finish inference, camera/audio cleanup, and resource release before sending the matching `demoStopped`. `demoStopFailed` keeps navigation and execution blocked; `retryDemoStop` uses a new request ID. An old stop result cannot complete a later attempt.

Demo data remains owned by its feature. The library retains definitions and the selected destination across transfer failures. It never deletes input, history, or generated results. Keeping those values when a demo runner releases runtime resources is the integration's responsibility.

## Verification boundary

Issue #171 covers this pure-logic contract. The tests cover catalog intersections, search ownership, inspection versus execution, identity mismatches, pinned distributions, progress, stale responses, cancellation, deletion, and stop/retry ordering. GPU inference and real model distribution are outside these tests.

The initial independent review found an inspection path that bypassed the missing/invalid distribution rule. Validation completion and the execution gate now reject that path, with regression cases for both missing metadata and an unpinned revision.
