# iPhone Duo Models navigation

Issue [#533](https://github.com/oka-yuji/coreai-samples/issues/533), following the toolbar implementation in PR #540.

## Implementation and ownership

One navigation split view owns the sidebar and selected detail. Its preferred compact column and column visibility are local presentation state. `ModelLibraryNavigation` is a pure policy; it does not create or stop sessions. All navigation still reaches the existing `openDemo` and `closeDemo` actions. The library Store and renderer stay outside the changing column layout.

The **Back to Models** button in both layouts sends the close request before any column transition. It is disabled while the stop is pending. The system Back is hidden from the start, and a MainActor UIKit bridge disables edge and content pop gestures while the detail owns them. Overlapping guards share ownership; the last release restores the original enabled state. A second return cannot leave the stopping or retry view. Matching acknowledgement clears the selection, then the preferred column returns to the sidebar. The same selection keeps the same detail identity. Accessibility escape invokes the same guarded request as the return button; while a stop is pending it sends no new action.

Both columns have flexible width preferences, and regular widths request both columns. This avoids the default portrait sidebar overlay observed on the Duo simulator without inspecting the device model or hardcoding a screen size. The grid's adaptive tile sizing is unchanged.

## Automated and simulator evidence

Xcode 27.2 (`27B52`), Swift 6.4.2, device SDK iOS 27.2 and deployment target iOS 27.0 were used. The separate preview was built for the simulator SDK and run on a dedicated iPhone Duo with iOS 27.1 (`24A94425`). It uses synthetic resources and performs no inference.

| Check | Result |
|---|---|
| NAV-01: selecting the current demo | The policy requests detail without replacing its session; simulator reselection preserved the sample state. |
| NAV-02: return while stopping | The policy keeps detail through a stale acknowledgement and returns only after the matching acknowledgement; the controlled simulator return kept its title and stopping view. |
| NAV-03: failed stop and retry | Failure exposed retry in detail. Retry remained in detail until the fixture completed the new stop. |
| NAV-04: resize with selection | Synchronization requested no navigation action and kept the selected detail. |
| Existing stop/session tests | The demo-switch, failed-stop/retry and background-suspension tests remained successful. |
| Layout | Closed showed one column. Open landscape, open portrait and Partially Open showed the sidebar and detail. |
| Retention | Fixture-set nondefault input and the sample result remained through opening, rotating, partial folding, reselection and closing. |
| Two-column return | The return button retained stopping detail until completion, then showed the empty detail and enabled model deletion. |
| Pop-gesture ownership | Both recognizers were disabled in Closed, Open detail, Closed again and while stopping. After returning to Models, both restored their original enabled state. |
| Two-column switching | Selecting another demo kept the old stopping detail until completion, then showed the new demo. |
| View lifetime | A temporary root/renderer-child probe kept one Store, owner and active session through Closed, Open, rotation, Partially Open and Closed. Neither view disappeared and scenePhase did not change during those actions. |
| Repository gates | Privacy, strict lint, changed-file formatting, all String Catalogs, the Rust input adapter, default package tests, macOS Release, dependency pins and unsigned iOS Release passed. |

The four new `NAV-*` tests live in `ModelLibraryTests.swift`. The targeted command was `swift test --package-path CoreAISamplesKit --force-resolved-versions --filter ModelLibraryTests`. The repository's required full test and build commands were also used; isolated build folders are selected explicitly when checking dependency pins.

## Findings retained from verification

- Clamping the preferred-column binding alone allowed a second native Back to leave the stopping view. Dynamically hiding Back after the first pop did not repair that navigation state. The final controlled return avoids the initial pop and keeps the disabled return and retry controls in detail.
- The default split layout hid or overlaid the sidebar in the inner portrait presentation. Requesting both columns and giving the detail a flexible width resolved the observed 669-point regular-width preview case.
- The older preview driver had a nonexhaustive effect switch. Its synthetic effects were updated, and explicit held-stop/failure/completion controls were added. Held stops use a cancellable stream rather than timed sleeps.
- An accessibility value assignment did not establish a committed text-field edit. Retention checks therefore use **Set sample input**, which changes the fixture's actual observable state. Native keyboard entry is not claimed by that check.
- The lifecycle probe's control case detected renderer disappearance and session removal on an ordinary return to Models, then inactive/background when Home was pressed. The probe code was removed; these are synthetic single-window observations, not proof of real image-generation continuity or multi-window behavior.
- Runtime version numbers alone did not identify Duo support: iOS 27.2 `24B8117` lacked the device, while iOS 27.1 `24A94425` supported it. The separately available iOS 27.2 `24B91016s` also listed Duo; it was not used for these UI checks.

- Hiding the system Back did not disable either pop recognizer on this runtime: both were enabled and their delegates allowed beginning. A temporary UIKit probe verified that the final guard disabled both and restored them after return. Coordinate-based swipe automation failed with `noWindowsAvailable`; actual finger gestures on hardware remain an owner check. The diagnostic code was removed.

## Acceptance boundary

NAV-01 through NAV-05 have code/test/build evidence. The simulator observations support the navigation investigation, but do not replace the owner's NAV-06 through NAV-10 acceptance on main. Physical Duo behavior, the ordinary iPhone, real-model continuity, VoiceOver escape and multi-window scene-phase effects remain unverified here. The automation bridge's secondary Cancel action did not produce a navigation change and is not treated as a VoiceOver gesture test. The existing background/stop cleanup contracts were not changed. Screenshots remain private.
