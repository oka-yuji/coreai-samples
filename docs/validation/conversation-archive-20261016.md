# Conversation archive validation — 2026-10-16

Issue: #79. PR: #139. Base: `c182d7d895cf3fed5c135b44d9d138dd2be7e377`.
Environment: macOS 27.0, Xcode 27.0 SDK, Apple Swift 6.4; Swift 6 language mode.

## Behavior and ownership

`conversations/<id>.json` is canonical. Each confirmed step and turn end saves a
versioned `TranscriptRecord`, turn settings, model location, metrics and citation
evidence. API credentials remain outside the archive. Markdown is a derived
export; both files are individually replaced atomically. A failed Markdown write
reports an error while retaining valid JSON, and retry repairs the export.

The archive actor serializes I/O. Compare-and-swap against the loaded/saved
revision prevents another window from overwriting newer history. A writer can
finish its own queued checkpoints even when a replaced Store task no longer
delivers its acknowledgement. Conflicts remain conflicts on retry; Save a copy
uses a new identifier and preserves the original conversation.

New and model switching wait for saving. Window closure retains the owner until
stop and save finish. Application termination waits for the same acknowledgement.
Failure keeps the app alive and offers a separate-copy recovery path, including
for a window that has already closed.

Startup restores the visible history before loading a model. Choosing the saved
model retains that history. The first request creates the Foundation Models
session from public Transcript entries and tool definitions. Historical tool
calls are not executed again. Missing results receive an interruption result;
turn indexes are adjusted to preserve their association after this insertion.
Multiple tool-call entries with intervening response text remain valid.

The runtime publishes actual completed/total prefill token counts. Stop retains
the stopping state until the generation task group finishes. The final evidence
snapshot is taken after the session has stopped, including late tool completion.

Unsupported or corrupt files fall back to read-only Markdown and are preserved.
Counters that could overflow during rendering or export are rejected before the
snapshot enters the screen state.

## Acceptance evidence

| Issue condition | Evidence | Result |
|---|---|---|
| Snapshot round trip and newer-version handling | `ConversationArchiveTests.roundTrip`, `unsupportedFile`, `invalidCounters` | Passed |
| Markdown rewritten with the same turns | `roundTrip`, `failedExport`, existing `ConversationSnapshotTests` | Passed |
| Relaunch, show history, prepare and continue | `ConversationArchiveSelfTest`: startup without a model, settings/evidence restore, matching-model retention; `ConversationTranscriptTests.continueWithoutReplay` uses the real FM session with a CPU model | Automated portions passed; real-model and visual acceptance remain open |
| Stop during preparation, then no further generation | `PreparationProgressTests.cancelsReading`, `ConversationPersistenceTests.stopsPreparation`; runtime fake blocks at a real prefill boundary | Automated portions passed; physical-model acceptance remains open |
| Repository commands | Native Mac input build, full default package suite, strict SwiftLint, changed-file swift-format, Mac Release build | Passed |

The full default package suite passed across 24 test targets. Model-dependent
tests were not enabled. After the final stop/repair changes, the affected
conversation and preparation suites passed again. The app's offline agent
self-test passed, including the archive integration checks. Runtime localization
verification found 79 keys and no missing Japanese translations; the catalog has
634 keys with all configured translations present.

## Review and limits

Independent app and persistence reviews identified and verified fixes for
concurrent-writer overwrite, valid multi-call transcript rejection, malformed
counter overflow, closed-window conflict recovery, termination/repair ordering,
citation-evidence restoration and validation-driver save timing. CPU probes
confirmed the concurrency and lifetime failure sequences and their corrections.
The final review has no unresolved confirmed findings.

The two Issue #79 manual conditions remain unchecked. Another CoreAISamples
process is open, and permission to close it for exclusive model execution is
pending. An isolated bundle with a synthetic saved conversation was launched,
but computer-use approval rejected access to Core AI Demos; no visual result is
claimed. Real AppKit quit/window events and real-model response quality are not
proven by the fake-based lifetime and model tests.

Local logs, review ledgers, before/after CPU probes and code hashes are in
`/private/tmp/coreai-conversation-remaining-20261016-2EjWXh/`. This document records
the durable findings without including user conversations, credentials or raw
machine logs.

## Plan reconciliation

The measured implementation is 956 changed lines (879 added, 77 removed).
Separately counted: tests and validation drivers 513 lines (501/12), localization
110, generated Xcode project 15, package/project configuration 10, and this record.
The increase over the 600–900 line estimate comes from conflict recovery,
termination ownership and citation-evidence continuity. Splitting at storage/UI
boundaries would separate the save/restore/stop acceptance contract, so the
previously selected single implementation Issue is retained. The original 5–8
SP estimate is unchanged.

The next implementation is #80, which consumes this archive to list, rename,
trash and select conversations. #81 and #82 remain separate accepted work. The
new PR is not merged as part of this validation.
