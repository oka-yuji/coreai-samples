# Conversation budget validation

Issue #145, acceptance version v1; PR #146 on top of PR #144 (`507aaf5`) and
PR #132 (`dda3e1d`). The user approved the UI and explicitly requested fixing
early termination and merging the current stack. This record covers the
[budget contract](../specs/conversation-step-budget.md), not model quality.

Environment: Apple silicon Mac, macOS 27.0 (26A428), Xcode 27.0 (27A266a),
Swift 6, macOS 27 SDK. The app uses its existing MainActor isolation settings.
No Core AI model was loaded for these checks. Native Foundation Models checks
use a scripted CPU `LanguageModel`, temporary notes and archives, and no network.

## Acceptance evidence

The acceptance wording remains in Issue #145. Each ID maps to these results:

| ID | Evidence | Result |
|---|---|---|
| STEP-01 | `ConversationBudgetTests.searchFetchAnswer`; native `conversation/budget-research` runs two search/fetch/answer exchanges with independent budgets | pass |
| STEP-02 | `budgets`, `normalizedArchive`, headless preference-default check | pass: Chat 16, independent Work setting, 1–64 normalization |
| STEP-03 | `finalAnswer`, `ConversationTranscriptTests.finalAnswerWithoutReplay`, native `budget-answer` and `budget-toolInFinal` | pass: same transcript, pending-call closure, disallowed tools, one final request, no tool replay |
| STEP-04 | `cancelsHandoff`, `boundedAndCancellable`, native `budget-stopBudget`, `budget-stopFinal`, `budget-workRepairStop` | pass: explicit Stop and termination prevent handoff; stale events ignored; repair and busy state settle |
| STEP-05 | `legacyBudget`, `ConversationArchiveTests.finalAnswerArchive`, native `budget-restoreMarker` | pass: old limit two, formats 1–4, marker remap, JSON round trip and internal-prompt-free Markdown |
| STEP-06 | `finalFailure`, `continuationPermissionsAndPrefix`, native `budget-failFinal` and `budget-emptyFinal` | pass: retained research, no automatic repeat; manual continuation remains tool-free and citation prefix is preserved |
| STEP-07 | Required commands below | pass |
| STEP-08 | Multiple live Qwen exchanges, changed limit, Stop, bounded final answer and reopening history | not performed for this fix; user acceptance remains open |

## Commands

- `swiftlint lint --config .swiftlint.yml --baseline .swiftlint-baseline.json --strict --no-cache`:
  289 files, zero violations.
- `python3 Tools/Format/check_format.py --base origin/main`: 26 changed Swift
  files, pass, including the pending UI stack.
- `MACOSX_DEPLOYMENT_TARGET=27.0 ./Tools/Irodori/build-inputs.sh`: pass.
- `swift test --package-path CoreAISamplesKit`: all 24 test executables passed;
  the runner reports 475 tests and 35 intentionally skipped model/optional-fixture
  tests. Conversation Feature reports 63 tests; Conversation Runtime reports 30.
- `xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples -configuration Release -destination 'platform=macOS' build`:
  pass. An earlier build also passed with localization extraction disabled;
  the final extraction build used the normal project settings.
- Release app `--agent-selftest --offline`: 163 checks passed, including nine
  native budget scenarios and the restored-marker check. A Work fixture writes
  a temporary draft and verifies publication after the final answer. Another
  stops during citation repair and verifies held drafts, a warning, cleared
  repair state and released busy state.
- Release app `--check-localization`: 91 dynamic runtime keys, zero missing
  Japanese translations. Submitted catalogs contain 668 Localizable keys and
  102 Ingredients keys. Full language/placeholder validation passed; compiled
  bundle validation made 1,512 lookups with zero failures.
- `git diff --cached --check`: pass.

The catalog was checked using a staged-content copy so unrelated pre-existing
working-tree extraction changes were not silently included. The six new entries
are the only catalog changes in this PR.

An additional extracted-source audit found 18 pre-existing keys absent from the
submitted catalog (17 speech labels and the conversation rename field's `Title`).
All 18 were already absent before this fix; none is a new budget label. They are
tracked separately in [Issue #152](https://github.com/oka-yuji/coreai-samples/issues/152).
The initial source audit had no `.stringsdata` because extraction was disabled;
rerunning the normal build produced 404 keys from 75 files and exposed the
18 existing omissions. Neither audit is reported as a full-source localization
pass. Existing working-tree entries and metadata are preserved.

Local command evidence is retained under the task scratch directory, with
`red.log`, `suite-final.log`, `lint-first.log`, `format-staged.log`,
`rust-final.log`, `xcode-localization.log`, `selftest-final.log`,
`catalog-final.log` and `source-keys-extracted.log`. The original regression
failed under the two-step policy before the implementation. Raw user
conversations, paths and private cited URLs are excluded from public records.

## Review and corrections

Two independent review passes covered Feature policy/transcript/archive behavior
and app session/runner/tool/citation cleanup. Each reviewer also checked the
other pass's candidate fixes. Four confirmed findings were corrected:

| Finding | Correction and verification |
|---|---|
| R1: budget cancellation during citation repair skipped cleanup and could leave the UI busy | `settleBudgetStop` preserves running state through cleanup; native `budget-workRepairStop` verifies notes, guard and busy state |
| R2: finalization of a continuation lost its earlier answer prefix for citation checking | rebuild the same concatenated prefix displayed by the UI; `continuationPermissionsAndPrefix` |
| C1: Continue after a truncated final answer re-enabled tools | persist `toolsDisabled`; independent CPU probe and continuation test verify no tools, one step and reasoning off |
| C2: an out-of-range setting was clamped at runtime but saved unnormalized | normalize new-turn settings and preferences; independent archive-validation probe and `normalizedArchive` |

Final review has no unresolved confirmed or plausible candidates, failed
review tasks or unparsed candidates. Scope includes all implementation and test
changes, catalog deltas, project registration and documentation. Real-model/GUI
behavior remains excluded from automated review evidence. The staged review
scope and its content-sensitive token were checked before commit.

## Acceptance, integration and learning

Implementation and automated checks are complete for STEP-01–07. STEP-08 remains
unchecked. Automatic approval review previously rejected access to Xcode and
Core AI Demos, so GUI interaction was not retried through another control path.
The user's approval of the preceding UI is retained; it is not treated as
live-model validation of the new budget behavior. The explicit merge request
authorizes integration while that manual result remains separate.

The source delta is 287 lines (251 added, 36 removed), below the provisional
350–600 estimate. Tests are separately 593 lines (573 added, 20 removed), above
the 250–500 estimate because native cancellation, draft cleanup and archive
restoration need integration fixtures. Catalog registration is 66 added lines;
XcodeGen project registration is four added lines. Documentation is counted
separately in the PR. The original provisional 5 SP remains unchanged. Splitting
the budget from finalization would leave the no-answer failure in an intermediate
state, so the single implementation PR remains appropriate.

The specification, demo guide, both READMEs and historical diagnosis follow-up
now agree with the implementation. The reusable lesson is to reserve a bounded
answer opportunity and test native cancellation plus post-generation cleanup,
not only increase the tool limit. The Core AI guide records the native handoff
evidence separately. The next acceptance action is the manual Qwen check;
translation omissions are deferred to #152 rather than added to this change.

Integration order is #132 → #144 → #146. Each merge must match its reviewed head.
Any merge-only history adjustment must preserve the tested source tree. The
local checkout is then advanced to remote main without discarding the existing
catalog edits. Merged state does not imply release qualification or completion
of the guide's ten shipping checks.
