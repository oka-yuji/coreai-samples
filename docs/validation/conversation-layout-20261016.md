# Conversation layout — 2026-10-16

Scope: [Issue 143](https://github.com/oka-yuji/coreai-samples/issues/143), based on
`dda3e1d` from the Rust build fix in PR 132. PR 144 is stacked on PR 132; merge
132 before 144. The presentation contract is in
[demo-presentation.md](../specs/demo-presentation.md#conversation-layout).

## Acceptance and evidence

| Condition | Evidence | Result |
| --- | --- | --- |
| UI-01: unloaded, loading, failed and new-conversation notices start at the top leading edge | View branches use leading stacks and the parent fills the detail area from its top edge | Implemented; visual acceptance pending |
| UI-02: short conversations start at the top, long ones remain scrollable, composer stays below | Separate `.alignment` scroll anchor; existing latest-position behavior retained; composer outside the scroll view | Implemented; real scrolling and window-resize checks pending |
| UI-03: Chat / Work is visible before loading, with operation guards | Picker moved from the ready-only composer to the header; explicit running/restoring/terminating/read-only/RAG guards | Implemented; manual operation check pending |
| UI-04: model, new, copy and settings have consistent upper placements; settings do not shrink the transcript | Single common new-conversation toolbar action; conversation copy/settings toolbar; scrollable settings popover | Implemented; visual acceptance pending |
| UI-05: English/Japanese and wide/narrow windows remain usable | Twelve changed catalog keys; model label can compress; mode explanation is a localized help/accessibility hint | Catalog verified; rendering and narrow-window acceptance pending |
| UI-06: required lint, format, Rust input build, default package tests and Mac Release build | Commands below | Passed |
| UI-07: changed catalog and offline app checks | Commands below | Passed |
| UI-08: modes on the left, model on the right of the same row, native glass controls and readable accessibility variants | Native segmented picker, glass model menu, safe-area bars, single regular-glass composer | Implemented; visual/accessibility-settings acceptance pending |
| UI-09: user messages on the right and assistant answers on the left, with readable roles in long/narrow/localized content | Distinct alignment, role labels and background styles; wrapped selectable text with opposite-side spacing | Implemented; visual acceptance pending |

## Automatic checks

- `swiftlint lint --config .swiftlint.yml --baseline .swiftlint-baseline.json --strict --no-cache`: 287 files, zero violations.
- `python3 Tools/Format/check_format.py --base origin/main`: five changed Swift files, passed.
- `./Tools/Irodori/build-inputs.sh`: release input adapter built successfully.
- `swift test --package-path CoreAISamplesKit`: all 24 default test targets passed. Opt-in model tests were not enabled.
- `xcodebuild -project CoreAISamples.xcodeproj -scheme CoreAISamples -configuration Release -destination 'platform=macOS' SWIFT_EMIT_LOC_STRINGS=NO build`: Release succeeded. The final reviewed source is rebuilt after the mode-guard correction.
- `Tools/Localization/check-catalogs.py` against an isolated copy of the submission catalogs: 662 Localizable keys and 102 Ingredients keys, zero translation or placeholder errors.
- Built app `--check-localization`: 91 runtime keys, no missing Japanese translations.
- `xcrun swift Tools/Localization/check-bundle.swift` against the built app and submission catalogs: 1,500 compiled English/Japanese lookups, zero failures.
- Built app `--agent-selftest --offline`: passed; live web-search and WebView checks skipped as requested by the offline mode. No model was loaded.

Environment: Xcode 27, Apple Swift 6.4, macOS 27, arm64; application language
mode Swift 6 with MainActor default isolation. Logs are retained in the local
temporary directory `coreai-conversation-layout-20261016-3sn_ui2v`.

The working checkout already contained Xcode extraction changes in
`Localizable.xcstrings` before this task: 18 added keys and 33 metadata changes,
with no existing translation values changed. They are preserved in the working
tree and excluded from the submitted catalog. The submitted catalog changes
only nine Work translations and three new labels. The initial whole-working-tree
catalog check reported those 18 pre-existing missing translations; it is not
represented as a successful full-working-tree audit. After the build, three
removed source keys had new stale markers; only that new metadata was removed.
A structured comparison confirmed that all pre-existing catalog data remains
intact.

## Review and corrections

Two independent reviewers checked the view/layout and the toolbar/model/history
paths. The parent verified their evidence and the final corrections.

- Removed the added Command-N binding because `WindowGroup` already assigns it
  to New Window.
- Removed the context-full notice's extra New button; its notice remains and
  the common toolbar provides the action once the operation can be left.
- Replaced `canLeave` in the mode picker with explicit operation guards.
  A failed save blocks leaving the conversation, but does not require blocking
  the next turn's Chat / Work choice. The mode action does not modify prior
  turns or the saved snapshot; two reviewers independently confirmed this.

The model/history guards, Markdown copy path, settings ownership and source
translations were inspected. The scroll-anchor API exists in SDK 27, and
Apple's documented separate alignment role supports top alignment for content
shorter than its container while retaining a different anchor for long content.
No new runtime, storage format, Swift concurrency contract or model behavior is
introduced, so no Core AI guide update is required for this presentation change.

## Remaining manual acceptance

GUI operation was unavailable because automatic approval review previously
rejected access to Xcode and Core AI Demos. No alternative GUI control or
screenshot route was used. Source review and compilation do not establish the
rendered appearance, exact popover size, toolbar overflow or actual scrolling.

On the built app, check UI-01 through UI-05 in both languages, with the sidebar
shown and hidden and at a narrow window width. Verify one short exchange, a
long conversation, scrolling back during generation, model readiness/failure,
Chat / Work switching, and opening/closing settings. The user owns these manual
checks; no manual checkbox is marked complete on their behalf.

## Reflection and next state

The original estimate of provisional 3 SP remains unchanged. One cohesive
presentation change was sufficient; splitting the header, empty states and
composer would have left inconsistent intermediate layouts. The existing
presentation specification is updated with the approved behavior. This work
does not add tests that merely inspect View syntax; existing behavior tests and
the build cover the unchanged operation paths, with visual acceptance kept
separate. Implementation and automatic verification are distinct from manual
acceptance, merging and release. No unrelated improvement is included.

## Follow-up after the user's first check

The user's further 2026-10-16 feedback specifies the same-row placement and a
native Liquid Glass appearance. Issue 143 v2 keeps UI-01 through UI-07 unchanged
and adds UI-08. The [official reference comparison](../research/chat-controls-and-liquid-glass-20261016.md)
records ChatGPT, Claude, Gemini and Apple's material guidance before applying the
new layout. The first submitted revision is `7c76bb3`; this follow-up changes
three presentation files and the associated documentation.

The mode control remains a native segmented picker. Its description becomes a
localized tooltip and accessibility hint. Model selection moves to the opposite
edge of the same row, uses the native glass button style and allows long names
to truncate. Top/bottom `safeAreaBar` regions replace fixed separators. The
composer uses one regular-glass surface, with no opaque input background or
nested glass buttons. Operation guards and all persistence/runtime code are
unchanged. The original provisional 3 SP is retained.

Review identified an observation gap for composer height changes: safe-area bars
can change content insets without changing the scroll container size. The
follow-up accounts for top/bottom insets in the available height and for the
bottom inset in the end-of-content comparison. Scrolling to the end still
requires that the previous position was at the end, so reading earlier content
does not opt into following the tail. This is an adjustment based on the geometry
contract, not a claim that a visual failure was reproduced.

Two reviewers independently probed the public `ScrollGeometry` initializer
without opening a window. At fixed container height, changing only the insets
did not change `visibleRect.height`; replacing `containerSize.height` with that
property alone would therefore not establish the intended inset observation.
Those value-type probes do not reproduce the geometry supplied by a real
ScrollView. Actual scrolling remains part of manual acceptance.

Final follow-up checks passed: strict SwiftLint (287 files, zero violations),
diff format check (five Swift files), the Rust input build, all 24 default
SwiftPM test targets, Mac Release build, offline app self-test, 91 runtime
localization keys and 1,500 compiled catalog lookups. Both independent reviewers
rechecked the inset-aware expressions and found the observation gap addressed;
no additional confirmed code finding remains. Logs for this follow-up are in
the local temporary directory `coreai-liquid-glass-20261016-uj7yn6mn`.

GUI appearance, narrow-window resizing, scroll behavior and reduced-transparency
or increased-contrast settings still need manual confirmation. The previous
automatic approval refusals remain in effect; no GUI operation or alternative
screenshot route was attempted in this follow-up.

## Message role follow-up

Issue 143 v3 adds UI-09 after the user found questions and answers difficult to
distinguish. User bubbles now align right, assistant bubbles left. The changes
only affect the two message row variants; text, identity, history, operation
guards and execution policy remain unchanged. No catalog keys are added.
The user's existing private step-limit logs were inspected separately; the
[sanitized diagnosis](../research/chat-step-limit-20261016.md) records the fixed
two-step Chat budget and proposed improvements without copying the transcript.

The message-row revision passes strict SwiftLint, diff formatting, the Rust
input build, all 24 default SwiftPM test targets, Mac Release build, offline app
self-test, 91 runtime localization keys and 1,500 compiled catalog lookups.
The independent UI review found no confirmed code defect. A second reviewer
checked the diagnosis against the policy, counter reset, tool rejection and
saved metadata; the report distinguishes the stopped follow-up from a completed
answer and the proposed continuation from the existing note-save action.
Logs are in the local temporary directory
`coreai-chat-feedback-20261016-865i4mm_`. GUI checks remain pending under the
previous approval refusal; no model was rerun for the diagnosis.
