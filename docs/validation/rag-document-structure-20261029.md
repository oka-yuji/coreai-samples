# Document structure validation

Issue #374; PR #384, stacked on #383 / `ef12fd112c6e1a7ccf77ba279179e5d935a1be13`.

## Acceptance evidence

| Condition | Evidence | Result |
|---|---|---|
| RG-D01 | `hierarchy` for HTML/Markdown, `identities` | Pass |
| RG-D02 | `unicodeRanges`, `projection`, `snapshotVersion`; original ranges and byte-identical text | Pass |
| RG-D03 | `markdownBoundaries`, `htmlBoundaries`, `excludedAncestors`; independent 8 ancestor and 10 Markdown conditions | Pass |
| RG-D04 | `fallbacks`, CLI text-document results | Pass |
| RG-D05 | `rag-structure/cancelled-acquisition`, existing Conversation access/source regression | Pass |
| RG-D06 | Actual CLI, 36 questions / 8 documents; all text bytes and explicit versions match the manifest | Pass without learned weights |
| RG-D07 | Required commands and offline self-tests | See command record |

The CLI returned structure for the five HTML/Markdown manuals and `unsupportedFormat` for the three plain-text documents. This is an acquisition/parser check, not measured answer quality.

## Review

Independent reviewers covered the core/format/parser/fixture scope and the app/acquisition/project scope. The following boundaries were corrected before final acceptance:

- The pinned Markdown parser's Setext range included the next paragraph. Its AST still identifies the heading; the implementation verifies the original delimiter line to establish the actual end.
- Checking only the selected HTML root and descendants admitted articles inside non-body/hidden ancestors. Candidate eligibility now checks ancestors before selecting the root; the independent eight-condition counterexample is rejected.
- A computed version was not encoded in snapshots. The version is now stored with immutable source fields and checked during decoding.
- Content and preamble ranges are explicit, so subsequent retrieval can distinguish an introduction from surrounding boilerplate.

The original normalizer's function bodies and entity table were compared with the base implementation. A separate CPU probe compiled the actual `RAGPageRead.swift` and confirmed byte equivalence for long/short HTML and text, final URL and format, render fallback, and refusal before render. The permanent offline checks cover the same boundaries.

No confirmed findings remain in the reviewed scope. Actual network/WebKit/PDF activity and Qwen inference were not run for this unit. No new model is required for structure extraction.

## Commands

- `./Tools/Irodori/build-inputs.sh`: success.
- `swift test --package-path CoreAISamplesKit --filter RAGStructureTests`: 10 test functions, success.
- Related RAG/evaluation tests: success.
- Mac Release build: success.
- `CoreAISamples --agent-selftest --offline`: success, including seven structure checks and all previous evaluation checks.
- Model-free CLI structure inspection: success; 8 original texts and hashes match.

Full `swift test --package-path CoreAISamplesKit`, strict SwiftLint with the unchanged baseline, and the format check against the preceding branch succeeded. The combined format check against origin/main also succeeded. No skipped model test is counted as model acceptance.

## Scope and next work

The snapshot and parser interfaces preserve plan v1.1. The provisional 5 SP remains unchanged. Mature parsers are pinned instead of creating a second Markdown grammar; their locations are verified against original bytes before use. Structure failures fall back explicitly. Issue #375 adds ranking and packing, and #376 adds optional LLM section selection. The Qwen bundle location and #373/#377 real-model conditions remain pending; this unit does not resolve them.

The complete branch changes 705 implementation lines, 274 test lines, 82 documentation lines and 120 generated project/lockfile lines (added plus removed). The behavior-preserving normalizer relocation is a separate commit. The implementation remains within the 500–800 line estimate and retains the planned unit boundary.
