# Conversation sidebar validation — 2026-10-16

Issue #80, Draft PR #140. The base is #139 at `5854f7f`; merge order is #139 then
#140. Environment: macOS 27.0 / Xcode 27.0 SDK / Apple Swift 6.4, Swift 6 mode.

## Implemented behavior

The sidebar groups saved conversations, ordered by update time, separately from
the six existing Tools. New is a separate action. The active conversation remains
visible before its first save so a user visiting Tools can return to stop it.
Other conversation selections and mutations are disabled while generating,
saving, restoring, loading a model or changing archive files.

The selected conversation is restored on the next launch. Explicit New records
an empty selection. Initial save and selection are one archive-actor operation,
so a superseded save cannot later undo New. Opening another model's history does
not permit sending until that model is loaded. Failed opening preserves the
previous visible history and persistence state.

Rename updates JSON, Markdown, list metadata and the selected conversation's
saved revision. If only the Markdown write fails, the committed JSON version is
adopted and a retry rewrites only the export. Delete requires confirmation and
moves both files together to a visible folder in the macOS Trash. A failed Trash
operation rolls the staged files back. Read-only files with a malformed body can
also be deleted after their revision is checked; an unknown-version request
cannot delete a file that has since become a valid snapshot.

The same Store owns archive mutations, restore and termination. Quit waits for
restore to settle before choosing what to save or release. Restoring another
conversation clears the previous session's displayed generation statistics.

## Acceptance evidence

| Issue condition | Evidence | Result |
|---|---|---|
| Sidebar, New, selection and continuation | App offline self-test `conversation/sidebarRename`, `sidebarNewAndSwitch`, `missingHistoryPreservesSelection`; model-match checks from #79 | CPU application wiring passed; visual and real-model acceptance open |
| Switching disabled during a running turn | `ConversationLibraryFeatureTests.blocksConflictingWork`, `unsavedCurrentRow` and independent source/probe review | Automated state checks passed; manual UI acceptance open |
| Rename updates list/file; delete moves files to Trash | `ConversationLibraryTests.renameAndSelect`, `movesToTrash`, `failedDelete`, `renameExportFailure`, `deletesCorruptFile` | Passed, including real macOS Trash with isolated fixtures |
| New GUI E2E cases and existing GUI cases pass | Added `new-conversation` and `switch-conversation` to the conversation driver | Not executed; required GUI/model access is unavailable |
| Repository lint, format, package tests and Mac Release | Strict SwiftLint, changed-file swift-format, full default package suite (24 targets), Release build | Passed |

The app's offline self-test passed. Runtime localization checked 85 keys with no
missing Japanese translations. All 645 catalog keys have their configured
translations. Model-dependent package tests remained disabled. Temporary Trash
fixtures were removed using only the returned fixture paths.

## Review and remaining verification

Independent reviews found five confirmed defects and verified their corrections:
missing navigation to an unsaved running conversation, termination racing with
restoration, previous-conversation metrics, partial rename version mismatch and
deletion of corrupt read-only history. The superseded first-save selection
ordering was additionally hardened and checked. No confirmed review finding
remains unresolved.

GUI and physical-model checks remain open. Computer-use approval rejected access
to Core AI Demos, and closing the user's running model app has not been approved.
The GUI driver is not invoked through another mechanism to bypass that rejection.
The automated substitutes above are not counted as passing the GUI command.
This PR remains Draft until that required command can be validated.

Local logs, manifests, review ledgers and CPU probes:
`/private/tmp/coreai-conversation-remaining-20261016-2EjWXh/` (`sidebar-*`,
`review80-*`, `app80-probe`).

## Plan reconciliation

Implementation: 750 changed lines (698 additions / 52 deletions). Separately
counted: tests and validation drivers 334 (333/1), localization 121, generated
project 4, and this record. Recovery for partial rename, restore/quit ordering and
the unsaved active row account for the increase over 450–700 lines. Storage and UI
are kept together because selection, archive revision and termination share one
acceptance contract. The original SP 5 estimate is retained.

The accepted next work is #81 (continuation and split notes) and #82 (RAG source
continuity). Their implementation can proceed while GUI/model verification is
waiting. None of these new PRs has been merged.
