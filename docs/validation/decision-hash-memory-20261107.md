# Decision bundle hash memory

Issue #281. Swift 6.4, Xcode 27.0 (27A266a), macOS 27.2 (26B123). The old HubDownloader was removed by #466; the remaining production path is DecisionBundle.digest.

Each 4 MiB read and SHA-256 update now runs inside an autorelease pool. The digest, cancellation check, thrown errors and deferred file closure keep their existing behavior. No shared hash API or model contract is introduced.

## HMS-01

`COREAI_DECISION_HASH_MEMORY=1 swift test --package-path CoreAISamplesKit --force-resolved-versions --filter DecisionContractTests.hashMemory` runs alone in a new process. A sparse zero-filled 268,435,456-byte file is added to a synthetic bundle's SHA256SUMS. DecisionBundle.init verifies the actual inventory, including this file, against its known SHA-256. No trained model or GPU is used.

| Implementation | Kernel lifetime physical footprint peak | Below 134,217,728 bytes |
|---|---:|---|
| Original digest | 278,938,512 B | Fail |
| Per-chunk autorelease pool | 13,550,288 B | Pass |

The value comes from TASK_VM_INFO.ledger_phys_footprint_peak after successful bundle verification. It is not RSS, a sampled peak or a before/after subtraction. The regression fails on the original implementation and passes after the fix. Normal package runs skip this measurement.

## HMS-02 and HMS-03

- DecisionContractTests.bundleValidation retains the corrupt-hash and invalid-inventory rejection checks.
- DecisionContractTests.hashFailures confirms that a cancelled task receives CancellationError and a directory in place of a hashed file receives DecisionError. The cancellation case is pre-cancelled; it does not measure cancellation latency midway through a large file.
- The ordinary DecisionContractTests run reports nine tests with no failures and the memory check explicitly skipped. The isolated memory run passes separately.
- The default package suite reports 1,279 tests across 43 runs with no failures. Opt-in checks retain their skip status.
- Strict SwiftLint, changed-file swift-format, privacy checks, Rust Mac/iPhone input builds, Mac Release, signed CoreAISamples-iOS Release and package-pin checks pass.
- Independent review confirms that the synthetic file reaches the production digest, the pool encloses both allocation and consumption, and exceptions still unwind through file closure. No confirmed findings remain.

Production delta is six added/deleted lines. The tests and opt-in instructions are separate. The existing local fix remains one unit; no new state owner or dependency is added. The SHA and failure contracts are unchanged, so no model-report compatibility status is upgraded.
