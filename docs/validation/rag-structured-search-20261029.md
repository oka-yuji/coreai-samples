# Structured BM25 validation — 2026-10-29

Issue #375 / plan v1.1. Base `feat/374-rag-structure` at `e297a03`. macOS 27 / Xcode 27 / Swift 6.4. These are CPU and native Foundation Models fake-executor checks; no Qwen weights were loaded.

| Acceptance ID | Evidence | Result |
|---|---|---|
| RG-B01 | `RAGStructuredSelectionTests.headingMatch`, `rag-selection/parent-match` | Parent heading selects the corresponding original body |
| RG-B02 | `groupBudget`, `longBodies`, `duplicateBodies`, `SourceGroupTests` | Section boundaries, deduplication, page and final prompt group budgets |
| RG-B03 | `fallbackIdentity`, `headingOnlyLeaf`, native `no-match-fallback`, `unsupported-fallback`, `page-budget-reason` | Requested/effective strategy and fallback reasons remain distinct |
| RG-B04 | `rag-selection/docs-permission`, existing `conversation/access-*` suite | Denied Web acquisition is not called; Chat/Work permission boundaries preserved |
| RG-B05 | `rag-eval/native-work/structure-bm25`, `note-citation`, `SourceGroupTests.sharedHeadingAndArchive`, existing continuation/cancellation/archive suites | Work answer and saved note cite S1; original document/version/ranges survive ledger serialization |
| RG-B06 | `--rag-eval Tools/RAGEvaluation/corpus.json --validate-only --rag-strategy flat/structure-bm25 --retrieval-output NEWFILE` | Both modes produce 36 records with candidates, supplied bodies, reasons and elapsed time |
| RG-B07 | Required checks below and `--agent-selftest --offline` | Pass; flat native fixture and default strategy regressions preserved |

## Commands

All commands run from the isolated worktree. Build and native tests used `/private/tmp/rag374-derived` for compiler outputs, with no change to the user's original checkout.

```sh
swiftlint lint --config .swiftlint.yml --baseline .swiftlint-baseline.json --strict --no-cache
python3 Tools/Format/check_format.py --base origin/main
./Tools/Irodori/build-inputs.sh
swift test --package-path CoreAISamplesKit
xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples -configuration Release \
  -destination 'platform=macOS' -derivedDataPath /private/tmp/rag374-derived build
/private/tmp/rag374-derived/Build/Products/Release/CoreAISamples.app/Contents/MacOS/CoreAISamples \
  --agent-selftest --offline
```

Logs: `/private/tmp/rag375-{lint,format,irodori,full-tests,build,offline-final}.log`. Targeted selection/group tests: `/private/tmp/rag375-group-tests.log` (7 + 2 functions passed). SwiftLint reports zero violations. The complete package command passed; tests requiring absent model assets remain their existing skips, not actual-model evidence. Offline mode deliberately skips live search and WebView.

Model-free A/B inspection output: `/private/tmp/rag375-inspect-deky2dpd/{flat,structure-bm25}.json`. All 36 cases are present. A is always flat; B records both successful structured selection and explicit flat fallback. Its budget unit is characters; no answer quality or token-performance claim is made from this inspection.

## Review and follow-through

Independent core/app review identified duplicate body ranks, heading-only leaf evidence loss, page-budget rejection misclassified as no match, and final prompt trimming splitting heading/body groups. Each has a targeted regression test and a correction. Heading-only leaves conservatively fall back to flat; unranked candidates continue to be eligible as required by the approved A/B comparison.

Body-only Hit@k and all-supplied-range recall are intentionally separate measures. Their distinction and group packing are recorded in `docs/specs/rag-structured-search.md`. Verified packing/ranking findings are also appended to the Core AI guide INBOX. The Qwen bundle path remains unresolved, so the final real-model comparison and adoption decision belong to #377 and are not marked successful here.
