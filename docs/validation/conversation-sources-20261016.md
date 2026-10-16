# RAG source continuity — 2026-10-16

Issue #82, Draft PR #142, based on #141 at `e5b5ad6`. Merge order is
#139 → #140 → #141 → #142. Environment: macOS 27.0 / Xcode 27 SDK /
Swift 6.4, Swift 6 mode. These PRs have not been merged.

## Implemented behavior

RAG now retrieves sources and sends an expanded user prompt through the existing
ConversationSession. Its answer, source IDs and native transcript remain in the
same saved conversation. Normal follow-up questions use that history. Source
text stays in the user prompt; common instructions only state the conditional
citation requirement. Old conversations retain their original instructions and
tool definitions. Source replies, including Continue, refuse tool execution.

IDs are unique within a conversation and identify a document URI and the SHA-256
of its complete extracted text. The same document version reuses its ID and
retains selected passages; changed text gets another ID. Each passage records
its exact text and character range in that extracted version. For PDFs this is
PDFKit's extracted text, not the original PDF bytes. Metadata dates remain in
the user prompt. JSON stores the ledger and turn associations, and Markdown
includes a source appendix.

The actual expanded prompt, including the finalization message, must fit 8,000
tokens. Lower-ranked trailing passages are omitted when necessary, before IDs
are committed. If no source fits, generation does not run. Retrieval runs off
MainActor, and Stop waits for the cancelled retrieval group to finish. Model
identity is bound before retrieval, so an empty or cancelled first question can
be followed by another question. Statistics from an earlier answer are cleared.

Citation checks verify available IDs, not whether the claim is supported.
Collected IDs become available only after their source question is recorded in
the native transcript. On restore, availability is reconstructed from each
source turn's own transcript range; a later ordinary question cannot stand in
for a cancelled source turn. IDs remain reserved even when the original source
question was cancelled. Unknown or unavailable IDs keep affected notes as Drafts
with either Web-guard setting.

The Web URL ledger continues to record successful retrieval and selected text.
That fact can remain true when later generation is cancelled. Its hash, character
count and seen text all describe the same selected passages, grouped by document
version. The separate source-ID check additionally requires native prompt
delivery. Neither check establishes semantic support.

## Acceptance evidence

| Issue condition | Evidence | Result |
|---|---|---|
| RAG answer and follow-up retain the same IDs | State/transcript, app wiring and independent fake-session probes | CPU checks passed; real-model/manual acceptance open |
| IDs uniquely map to document version and passage | `ConversationSourcesTests.stableIDs`, `rejectsConflicts`, `sharesConversation`; `DocumentSourcesTests.exactPassages`; `ConversationArchiveTests.sourceArchive` | Passed |
| Revised GUI RAG steps plus Chat and Agent | Updated `rag-answer`, added `rag-followup`, strengthened current-turn result checks | Not executed; required GUI/model access unavailable |
| Four repository commands | Full default package tests, strict SwiftLint, changed-file swift-format, Mac Release build | Passed |

Additional regressions cover prompt budget, metadata-only dates, inline/fenced
code citation handling, cancelled source delivery, source archive validation and
late retrieval results. The offline app self-test checks real local-file
retrieval off the main thread, selected Web-text hashes, local citation checks,
Stop joining retrieval, and rejection of a previous answer by the GUI predicate.
These tests use isolated fixtures and do not load a model or operate the GUI.

The default suite passed across 24 test targets. The final affected runs passed
18 RAG tests, 4 native-transcript tests, 28 conversation-runtime tests and
54 conversation-feature tests. Release and the offline app self-test passed.
Runtime localization checked 91 keys with no missing Japanese translations;
all 659 catalog keys have their configured translations.

## Review and remaining verification

Independent review and the integration audit corrected: whole-document versus
excerpt hash mismatch; an indented false code fence hiding a citation; first
retrieval failure leaving the model identity unbound; stale generation results;
omitted metadata dates; a GUI follow-up accepting the previous answer; and
collected IDs being accepted before native prompt delivery. Each has a CPU
regression or isolated before/after probe. No confirmed finding remains open.

Implementation is 1,001 changed lines (588 additions / 413 deletions), including
removal of the separate RAG session and display path. Separately counted are
tests and validation drivers 410 (375/35), localization 70 and generated project
4. User documentation also reconciles the dependent archive/sidebar/note changes.
The shared transcript, source ledger, lifecycle and citation checks form one
acceptance unit; the revised 700–1,100 line plan and original SP 5–8 are retained.

Computer-use approval rejected access to Core AI Demos. Closing the user's
running model app has not been approved. GUI automation has not been invoked
through another route, and no competing model job was started. The required
GUI command, real-model source continuity and manual acceptance remain open.
PR #142 stays Draft. CPU results do not establish model answer quality, actual
KV reuse, physical-device performance or the GUI result.

Local evidence: `/private/tmp/coreai-conversation-remaining-20261016-2EjWXh/`,
including `rag-all-tests-final2.log`, `rag-regression-final.log`,
`rag-build-final2.log`, `rag-selftest-final2.log`,
`rag-localization-final2.log`, `rag-lint-final2.log`,
`rag-format-final2.log`, `review82-*` and `app82-probe`.
