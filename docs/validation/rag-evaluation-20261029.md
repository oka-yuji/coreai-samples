# Fixed RAG evaluation validation

Issue #373; parent #378 plan v1.1; PR #383. Base: `1b1822bf64f08d01b50d47e47ee5b760146e08e3`.

The user authorized implementation after clarifying Qwen3.8 27B 4bit Work as the target. This record covers the fixed-input infrastructure and baseline A. B/C and real-model adoption remain later units.

## Evidence

| Condition | Evidence | State |
|---|---|---|
| RG-E01 | `flatCompatibility` plus `rag-eval/flat-baseline`; original offsets, candidate/packing order and budgets | CPU pass |
| RG-E02 | `rag-eval/unknown-search`, `rag-eval/unknown-page`; replay misses have no live acquisition callback | CPU pass |
| RG-E03 | `invalidInputs`, `unsupported`, `unicodeEvidence`, `rejectsNormalizedEvidence`, `manifestBytes`; native validation and a UTF-16 missing-model case | CPU/native preflight pass |
| RG-E04 | `completeDenominator` and driver tests for failure, interruption, native cancellation, unconfirmed cleanup and missing control | CPU pass |
| RG-E05 | `frozenCorpus`; 8 synthetic documents, development 12, evaluation 24; source/gold/setting versions | Dataset checks pass; model artifact pin unresolved |
| RG-E06 | `rag-eval/native-work`, `work-metrics`, `answer-timing`, `failed-archive`, both no-answer checks; Release CLI connection | Native fake pass; Qwen end-to-end pending |
| RG-E07 | Required build/format/package/offline checks below | Pass with the lint path condition below |

Commands completed:

- `./Tools/Irodori/build-inputs.sh`: success, no learned model.
- `swift test --package-path CoreAISamplesKit`: success; opt-in model tests were not enabled.
- `swift test --package-path CoreAISamplesKit --filter RAG`: latest related changes, 28 test functions in 4 suites, success.
- `python3 -m unittest discover -s Tools/RAGEvaluation/tests -v`: 8 tests, success.
- `python3 Tools/Format/check_format.py --base origin/main`: 13 Swift files, success.
- Mac Release `xcodebuild`, followed by `CoreAISamples --agent-selftest --offline`: success, including 9 evaluation checks.
- Native `--validate-only`: the corpus is accepted and unimplemented strategies are rejected. This does not prove model execution.
- Native UTF-16 input with a deliberately missing model: exit 1 / failed, original manifest bytes retained and result hash equal to Python SHA-256, no inference.

Swift 6.4 / Xcode 27 / macOS SDK 27, Swift language mode 6. The Mac app does not set default MainActor isolation; the changed stateful types use explicit MainActor isolation.

## Lint path condition

SwiftLint 0.63.2 reports `/tmp/...` file names for this `/private/tmp/...` worktree, so its existing relative baseline does not match. The unchanged primary checkout passes the standard command. No repository baseline entries were added or relaxed.

The same 540 Swift/config inputs were copied to a normal path under the primary workspace's ignored `.local/validation/rag373-lint`, with every file verified byte-identical. Running the exact required strict lint command there found zero violations in 521 Swift files. The hash inventory and logs are retained in task scratch. This is a path workaround, not suppression of new violations.

## Review and corrections

Independent readers examined the core/data/driver and app/connection scopes, then rechecked corrections and remaining coverage. The coordinating reviewer checked the relevant code and CPU evidence. Nine distinct confirmed findings were corrected:

1. Work metrics read from a Chat-only property: use native session usage and retrieval metrics.
2. First transcript entry mislabeled as answer timing: observe the first nonempty answer segment, with an explicit interval.
3. Archive failure counted as completed: require saved persistence and retain the failure reason.
4. Unicode canonical equivalence accepted as original bytes: compare UTF-8 sequences.
5. No-answer UI text, or an earlier preamble after filtering it, recorded as the final answer: select the last answer before interpreting the placeholder.
6. Native cancellation overwritten by a driver exception: retain the native terminal state.
7. Unconfirmed child termination missing from case state: write a separate parent control record before launch.
8. Missing control record promoted a native result to completed: keep termination unconfirmed.
9. Accepted UTF-16 manifest hashed after lossy decoding: hash the original Data.

The parent-control tests also cover native results both present and absent. Independent CPU probes exercised HUP immediately after spawn, child cleanup, lock retention during unconfirmed termination, recording failures, immutable manifest copies and separate process IDs. These probes do not establish GPU cancellation behavior.

Snapshot/parameter identities, supplied-source ordering, permissions, injected dependencies across Host resets, fixed note paths, output-directory exclusivity and normal A behavior were included in review. No confirmed findings remain in the reviewed code. Model execution and real observed timing precision remain unverified.

## Remaining work and reflection

The registered legacy Qwen paths were unavailable on this machine. The user was asked for the current approved 4bit bundle location. `expectedModelHash` remains unresolved and actual inference is deliberately refused until the verified artifact is pinned. Do not substitute another model, claim measured A/B/C quality, or close the real-model acceptance conditions from fake execution.

The implementation grew beyond its initial line estimate because complete evidence handling spans native records, process control and failure aggregation. The original provisional 5 SP is retained; it is not recalculated from code size. These responsibilities share one fixed-input/result contract, so this remains one implementation issue. No new retrieval algorithm or normal UI was added.

Measured change size: 933 implementation lines (added plus removed); separately 473 test lines, 969 fixed experiment-data lines, 170 documentation lines and 16 generated project lines. The data file is a test corpus and its experimental conditions, not production application configuration.

The specification and tool documentation reflect the actual boundaries. Subsequent structure extraction can use the validated snapshot contract, while #373's real Qwen execution and #377's model/manual comparison remain pending. Merge and release are separate user decisions.
