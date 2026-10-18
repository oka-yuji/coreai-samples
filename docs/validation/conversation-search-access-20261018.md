# Search access validation

Issue #175 / PR #176, acceptance v1, based on main `19ea6bd`. The user's
October 18 request authorizes this fix and merging it into main. The
[specification](../specs/conversation-search-access.md) records the approved
OFF/ON behavior and independent Chat/Work selection.

Environment: Apple silicon Mac, macOS 27.0 (26A428), Xcode 27.0 (27A266a),
Swift 6 and the macOS 27 SDK. Native Foundation Models checks use a scripted
CPU LanguageModel and injected source/tool I/O. Note writes, archive saves,
the Store/Runner/Session, tool decisions and citation guard use the app code.
The new access fixtures load no Core AI model and perform no network I/O.

## Acceptance evidence

| Condition | Evidence | Result |
|---|---|---|
| ACCESS-01 | `ConversationAccessTests.offDeniesWeb`, native `access-chatOff` / `access-workOff` | pass: no web tool execution; Work note writes remain available |
| ACCESS-02 | `workSources`, native `access-chatDocs`, `access-workDocs`, `access-workWeb` | pass: selected flags reach retrieval; Docs-only web requests are refused before injected I/O |
| ACCESS-03 | `sourceModes`, native Work scenarios with real temporary note storage | pass: Work remains Work and publishes its note; Chat permits no tools |
| ACCESS-04 | `invalidSources`, `access-settings`, native empty/failed/stopped/retry retrieval cases | pass: no fallback generation, busy state clears, retry succeeds |
| ACCESS-05 | `continuationAccess`, `archivePermission`, `sourceBudget`, existing continuation/archive suites, native `access-workLimit` | pass: permission cannot expand; formats 1–5; final answer retains sources without another retrieval |
| ACCESS-06 | Commands below | pass for the required scoped checks |
| ACCESS-07 | Updated GUI driver plus real-model OFF/ON, Web/Docs and Chat/Work checks | unperformed; manual acceptance remains open |

The pre-change Feature sources were copied into a minimal CPU package, with
the same FeatureStore dependency and the new regression test. The original
code failed both tests, across four parameter cases, with 14 expectation
failures: OFF allowed web execution, and a Work source request became Chat.
This was an executed reproduction, not only source inspection.

## Commands and boundaries

- `MACOSX_DEPLOYMENT_TARGET=27.0 ./Tools/Irodori/build-inputs.sh`: pass.
- `swift test --package-path CoreAISamplesKit`: all 25 test executables passed.
  The runner reports 531 tests and 30 intentionally skipped model/optional
  fixture tests. The skipped tests are not treated as executed evidence.
- Strict SwiftLint with the checked-in configuration/baseline: 320 files,
  zero violations. `Tools/Format/check_format.py --base origin/main`: 21 changed
  Swift files passed. `git diff --check`: pass.
- Mac Release `xcodebuild`, using a separate DerivedData directory: pass,
  including the final GUI diagnostic correction.
- Release app `--agent-selftest --offline`: 177 checks passed. Fourteen are the
  new access/settings/citation cases, including the Work source-budget handoff.
- `Tools/Localization/check-catalogs.py`: all catalog entries and placeholders
  passed for their declared English/Japanese languages. Localizable has 702
  keys, Ingredients 102 and the separate InfoPlist catalog one. This PR adds
  seven translated keys only.
- Compiled Mac Localizable/Ingredients catalogs: 1,574 lookups, zero failures.
  App `--check-localization`: 92 dynamic keys, zero missing Japanese translations.

The optional full source-extraction audit reports the 18 pre-existing omitted
keys tracked in [#152](https://github.com/oka-yuji/coreai-samples/issues/152),
plus the iOS-only ObjectFinderView file absent from the Mac build's stringsdata.
All newly added access labels are present. A broader bundle check also tried
the iOS camera InfoPlist key against the Mac app and reported two missing
lookups; that catalog is outside this target's conversation resources. These
broader audits are not recorded as passing checks. No unrelated translations,
camera configuration or primary-checkout changes were included.

XcodeGen initially also rewrote existing Object Finder project entries. Those
unrelated generated changes were excluded; the submitted project change is
only the four registration lines for ConversationAccessSelfTest. The final
build uses that submitted project configuration.

Local evidence: task scratch `coreai-search-toggle-20261018-rl7pnb3h`, including
`red.log`, `suite-final.log`, `xcode-accepted.log`, `selftest-final.log`,
`lint-final.log`, `format-accepted.log`, `catalog.log`, `bundle-conversation.log`
and `source-keys.log`. Private user conversations and local model paths are not
part of the public test fixtures or records.

## Review and fixes

Independent passes covered Feature/archives and app/runtime/UI integration,
including the other pass's correction evidence. Four review findings and one
integration-test finding were corrected:

| Finding | Correction |
|---|---|
| A1: Work source finalization retained its retrieval request | Clear the effective request while retaining source IDs and the historical request; `sourceBudget` and native `access-workLimit` verify one retrieval |
| T1: the new test fixture retained its Store through a reference cycle | Make the test reference weak |
| G1: the GUI diagnostic required KV reuse across tool-definition changes | Mark those boundaries not applicable and retain a separate same-configuration Chat follow-up check |
| G2: the first of two GUI follow-ups could fail unnoticed | Capture its successful Chat outcome before sending the second question |
| I1: valid source-ID citations triggered the URL-only auto requirement | Accept valid supplied IDs for auto mode; native checks retain explicit URL requirements and rejection of unknown URLs; note-only citations still fail |

The native Work+Web test failed before I1: it attempted duplicate repair tools
and kept the note unpublished. The same test passes after the correction.
Final reviewed scope has no unresolved confirmed/plausible findings, failed
review tasks or unparsed candidates. GUI execution, live network behavior and
Qwen answer quality remain unverified by these passes.

## Completion and next action

Implementation, automated validation and specification alignment are complete
for ACCESS-01–06. ACCESS-07 remains unchecked. Previous automatic approval
review rejected Xcode/Core AI Demos access; no alternative GUI control path was
used to bypass it. Merging is explicitly requested, but does not establish
manual acceptance or the guide's full shipping qualification.

The source change is 180 lines (130 added, 50 removed), below the provisional
250–500 estimate because existing retrieval and task owners were reused. Tests
and diagnostics are 539 lines (488 added, 51 removed), within the 300–600
estimate. Catalog additions are 70 lines and generated registration four;
documentation is counted separately in the PR. The provisional 5 SP remains
unchanged. Keeping labels, permission gates and mode preservation in one PR
avoids an intermediate OFF setting that still performs retrieval.

Both READMEs, the demo guide and presentation specification now match the
access contract. The Core AI guide records the permission and finalization
pitfalls. Existing translation work remains with #152. The remaining user
acceptance is to try OFF/ON with the selected model after rebuilding; no new
feature work is implied by that check.
