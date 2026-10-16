# Conversation continuation and split notes — 2026-10-16

Issue #81, Draft PR #141, based on #140 at `2501e10`. Merge order is
#139 → #140 → #141. Environment: macOS 27.0 / Xcode 27 SDK / Swift 6.4,
Swift 6 mode. These changes have not been merged.

## Implemented behavior

Continue is available only after a capped answer with no unfinished control
syntax or context exhaustion. It sends the specified continuation prompt as a
new user turn. The UI joins the answers, preserving whitespace at the boundary;
JSON and Markdown retain the real turns. Citation validation checks the joined
answer and retains the original task's web usage and notes folder. Restarted
conversations preserve the previous instruction and tool definitions.

`write_note` stores part 1 and returns an app-issued identifier.
`append_note(note_id, part, content)` accepts parts in order, up to four parts
of 2,000 characters each. Repeating identical content succeeds without adding
it again. Gaps, changed retries, excess size and conflicting numeric citations
are rejected without truncation. The prompt targets 1,500 characters per part.
An incomplete write or append call has a distinct unsaved-note message.

An actor owns durable JSON records and Markdown exports under the notes folder.
New parts stay in `.draft/<note_id>/` until the task completes and the enabled
citation checks pass. Stop, cap and failures keep the draft. A canonical save
with an export failure returns the saved parts and a warning, so validation
never silently uses an older receipt. Publication compares the checked parts
with the current record. A concurrent append therefore cannot bypass checking.
Manual edits, moved files, redirected metadata paths and file-name collisions
are protected.

Publication repairs the draft export before exposing the completed file.
If only the subsequent metadata update fails, the result explicitly reports
that the file was published with a warning; restart can recognize the exact
published bytes and resume. This is not reported as an unpublished draft.

`move_note` operates on completed notes. New drafts cannot be moved during
the unfinished task, with either citation-guard setting. The prompt describes
the follow-up action. This makes the existing move feature consistent with
the requirement that interrupted tasks retain their drafts.

## Acceptance evidence

| Issue condition | Evidence | Result |
|---|---|---|
| Ordered and idempotent append; refuse missing/changed parts without truncation | `ConversationNotesTests.durableParts`, `rejectsInvalidParts`, `NotePartsTests` | Passed |
| Cut tool call is distinguished from cut answer | `ConversationContinuationTests.cutNote`, `refusesOpenSyntax` | Passed |
| Original 2026-10-12 task at 1536 versus default 4096 with a real model | No real-model run was performed | Open |
| Continue visibility and only the new turn is read | Pure-state/prompt/render tests passed; manual UI and KV observation not performed | Manual open |
| Four repository checks | Default package suite, strict SwiftLint, changed-file swift-format, Mac Release build | Passed |

Additional CPU regression checks cover stale validation receipts, failed Markdown
export, publication with interrupted metadata, external edits, managed moves,
case-insensitive file reservations, `/private/tmp` aliases and symbolic links.
The app's offline self-test covers Stop with both guard settings, restored task
folders and rejection of an unobserved append. Independent native Transcript
probes verify that old tool definitions remain unchanged.

The full default package suite passed across 24 test targets. The final affected
regression run passed 12 runtime and 16 feature/status tests. App offline self-tests and
Release build passed. The catalog has 652 keys and no missing configured
translations; runtime checking passed 85 UI keys. Model-dependent tests were
disabled, as required for an unlocked CPU run.

## Review, plan and remaining checks

Independent review covered pure continuation/rendering, tool schemas, archive
compatibility, note storage, cancellation and citation validation. Findings
included old-tool expansion, inconsistent Stop handling, restored folder drift,
premature move publication, continuation citation scope, inline-code citation
skipping, checked/published version mismatch, path aliases, case collisions and
publication-metadata recovery. Each was corrected and checked with a regression
test or an isolated probe. No confirmed finding remains open.

Implementation is 1,083 changed lines (940 additions / 143 deletions); separately
counted are tests and self-tests 552 (496/56), localization 77, generated project
4, and this record. The increase over the revised 500–900 line implementation
estimate covers storage failure recovery and compatibility boundaries found in
review. These share the same acceptance contract, so the single Issue/PR boundary
and original SP 5 are retained.

Computer-use approval rejected access to Core AI Demos. Closing the user's
running model app has not been approved. GUI automation is not invoked through
another route, and real-model jobs are not started alongside that app. The
required model comparison and manual continuation check remain open; PR #141
therefore remains Draft. CPU tests do not establish model quality, KV reuse or
the 1536/4096 comparison.

Evidence is in `/private/tmp/coreai-conversation-remaining-20261016-2EjWXh/`:
`notes-all-tests-final.log`, `notes-regression-final2.log`,
`notes-app-build-final6.log`, `notes-selftest-final4.log`,
`notes-lint-final2.log`, `notes-format-final3.log`,
`notes-localization-final.log`, and `review81-*` / `app81-probe`.
The accepted next implementation is #82, keeping RAG answers and source IDs
inside the same conversation.
