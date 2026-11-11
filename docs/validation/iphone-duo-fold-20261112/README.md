# iPhone Duo fold layout study

Issue [#535](https://github.com/oka-yuji/coreai-samples/issues/535), recorded on 2026-11-12. This is the result of two bounded prototype work blocks. The prototype remains on a separate disposable branch; this report does not change production layouts.

## Decision

Keep the current `ModelDemoHost` scrolling contract and the current video and camera layouts. Do not add `ArrangementView` inside the host's `ScrollView` or the split-view detail column. The tested split variants lose access to the secondary query and run controls in closed landscape. The other variants do not provide sufficient evidence for a production replacement.

Keep the camera's existing `.resizeAspect` full-field preview. The synthetic fit/fill comparison demonstrates the crop introduced by filling this viewport, but does not establish the best placement of a real camera preview on a physical inner display. No grid, sheet-placement, or vertical-toolbar override is adopted by this spike.

These are bounded decisions to retain the existing implementation, not a finding that every production screen is correct. No implementation Issue is proposed because no additional production change is adopted. Physical-device inspection and the incomplete coverage below remain necessary before concluding that no further changes are needed.

## Environment and main validation

The prerequisite PRs [#540](https://github.com/oka-yuji/coreai-samples/pull/540), [#549](https://github.com/oka-yuji/coreai-samples/pull/549), and [#552](https://github.com/oka-yuji/coreai-samples/pull/552) were merged. A detached checkout of main `4b9a9f85a123ed277a545576e4965d3bea0b5ed9` passed all eight commands required by `AGENTS.md`, the localization check, and an unsigned iOS Release build. The selected toolchain was Xcode 27.2 (27B52), Swift 6.4.2, with the iPhoneOS 27.2 SDK and the existing iOS 27.0 deployment target. Package pins were checked against the build's DerivedData. Model-report consistency and its Python tests also passed. No model inference was run.

UI observations used the iPhone Duo simulator on iOS 27.1 (24A94425). The installed iOS 27.2 runtime build 24B8117 did not list Duo among its supported device types; it was not used for these observations. The production app could not be run in this simulator because its SDK lacks `CoreAI.framework`. The separate preview application used the production library shell with simulated installation and synthetic demo bodies. No connected physical Duo was available.

The report was later integrated with main `3c75011`, which adds Chronos Models support and a ninth demo. The 504-row inventory and UI observations remain scoped to the eight demos at `4b9a9f8`; they do not cover Chronos. The later host changes preserve its outer scroll container while adding classification presentation and preview interaction policy. Integration builds/tests are separate from the earlier UI observations.

## Comparison protocol and evidence

[observations.csv](observations.csv) contains 96 unique comparison cases. [evidence.json](evidence.json) records the environment, selected rectangles, synthetic scaling results, and exclusions. Screenshots, raw event logs, and the disposable prototype are retained privately; they are not published here.

| Block | Container and content | Poses and layouts | Cases | Successful AX run activations |
|---|---|---|---:|---:|
| 1 | Video fixture inside the library host | Six poses, all six layouts | 36 | Not attempted |
| 2 | Camera-sheet fixture | Six poses, all six layouts | 36 | 18 |
| 2 | Video fixture, standalone and split-detail-root containers | Closed landscape with all six layouts; Book and Laptop with Split, Overlay, and Reserved regions | 24 | 20 |

The six poses are closed portrait, closed landscape, open portrait, open landscape, Book, and Laptop. The layouts are Current, Split, Vertical split, Overlay, Vertical overlay, and Reserved regions. The vertical variants restrict the arrangement axis. The split-detail-root comparison removes the host scroll view but still places the fixture in a `NavigationSplitView` detail column; it is an experimental placement, not an adopted contract.

Split uses media as primary content and controls as secondary content. Overlay reverses these roles: controls are primary and media is secondary. These are concrete configurations, so the different control availability does not isolate the layout style from content assignment. Media AX presence was not recorded; preserved query/run controls in Overlay do not prove that all media content remained available in closed landscape.

AX presence and AX activation are different measurements. Eight cases omitted both query and run controls from AX: closed-landscape Split and Vertical split in each of the four containers. The other listed cases exposed both controls. The 38 successful activation checks incremented a synthetic counter; they did not execute a search or camera model and do not establish finger reachability. The first block and the first 18 camera cases checked presence only. Geometry may remain for controls absent from AX, so a stored rectangle is not evidence of visibility or operability.

Block 1 used prototype `1095b96`; block 2's comparison used `96dcb60`. The corrected scaling and final fold observations used `c546dcc`. A diagnostic header, synthetic controls/results, and test media affect the available space. The video fixture is not the production pinned-player/results implementation, and the camera fixture is not `AVCaptureVideoPreviewLayer`.

Block 2 changed the input preset from `A synthetic blue object` to `Retained synthetic input` before the Laptop camera observations and retained it through the subsequent fixture observations. The comparison therefore does not keep input text constant throughout the block. No claim is made that the change has zero effect on layout.

## Findings and alternatives

| Alternative | Observation | Decision |
|---|---|---|
| Current scrolling layout | Preserved query/run AX presence in the listed cases. In the library-host Laptop fixture, the run control intersects the active division rectangle. | Keep the production contract; the fixture intersection identifies a check to repeat on the actual screen, not a demonstrated production defect. |
| Split and Vertical split | Closed landscape omitted the secondary controls in every tested container. Axis restriction did not restore them. | Reject these configurations as tested; any future proposal needs an explicit fallback preserving all operations. |
| Overlay and Vertical overlay | Preserved query/run AX presence in the listed cases. | Do not adopt inside the existing host/detail column. Presence alone does not establish better placement, reachability, or compatibility. |
| Reserved regions | The tested algorithm splits only when panes meet its own 160-point horizontal or 140-point vertical minimum; otherwise it falls back to a scroll view. The Laptop library-host media rectangle intersects the division after fallback. It contains no occlusion-avoidance algorithm. | Do not adopt this algorithm. Its thresholds and fallback do not establish a limitation of the `ReservedRegion` API itself. |

In Block 1's Book cases, the recorded media/query/run rectangles did not intersect the active division. The Current open-landscape round trip reproduced the same recorded geometry. Neither observation covers native model execution, all production content, or physical-device acceptance.

Apple's API documentation explicitly advises against putting `ArrangementView` inside scroll or split views and documents primary-only split content in closed landscape. Removing the host scroll view therefore does not, by itself, make the detail-root experiment a supported integration. [ArrangementView documentation](https://developer.apple.com/documentation/swiftui/arrangementview)

The Tech Talk lists some nesting restrictions without explicitly permitting placement inside a split-view detail column. Its omission is not evidence that the API documentation's restriction can be ignored. This study follows the explicit restriction; it does not claim to have resolved a contradiction in the framework implementation. [Tech Talk 111463, 16:09 onward](https://developer.apple.com/videos/play/tech-talks/111463/?time=969)

Region intersections use each node's local coordinate space and only active regions. The region frame already includes its margins; margins are not added a second time. Scroll containers do not automatically divide their content around a fold. [ReservedRegion documentation](https://developer.apple.com/documentation/swiftui/reservedregion)

## Camera fit and fill

The camera-sheet fixture also had active-division intersections: its media rectangle intersected the fold in Laptop/Current and in Book with Split, Vertical split, Overlay, Vertical overlay, and Reserved regions. These six cases all passed the synthetic Run activation check, which says nothing about the visibility of the whole preview. They are evidence to repeat on the native camera screen; the diagnostic header and surrogate media prevent treating them as production defects. Retaining the current layout does not establish that it avoids the fold.

The corrected Current camera-sheet fixture used a 621 by 385.55 point viewport in open portrait. The visible fraction is the geometric intersection of a synthetic field and that viewport, divided by the field's area.

| Synthetic field | Fit visible fraction | Fill visible fraction |
|---|---:|---:|
| 9:16 | 1.00000 | 0.34923 |
| 16:9 | 1.00000 | 0.90601 |

Screenshots confirmed that the corrected fill fixture clipped the oversized field inside the viewport. This comparison supports retaining full-field presentation until a real-camera evaluation justifies cropping. It does not verify camera orientation, detection-box alignment, usable touch targets, or whether an aligned versus full-screen preview is preferable on hardware. Those parts of FOLD-05 remain open.

## Multiple windows and invalid observations

The study enabled the production app's multiple-scene setting and created a second scene. AX listed both scenes while the screenshot showed the frontmost scene. Lifecycle notifications are retained in `evidence.json`; simultaneous tiling, other-app Split View, and background behavior caused specifically by folding were not established. A final foreground open/closed/open sequence emitted no additional scene-phase notifications in that process; it does not establish the lifecycle of an unseen secondary window.

The following observations are excluded from acceptance:

- Block 1 v1/v2 geometry allowed a late outgoing view to replace a newer node observation. Block 1 v3 uses an active appearance-owner ledger; only v3 comparison geometry is retained here.
- The earlier block 2 fill experiment allowed its synthetic content to expand before clipping. Its fill result is discarded and replaced by the four corrected `c546dcc` records.
- Device Hub's resize-mode experiment displayed a black screen while AX and geometry logs continued updating. It supplies no successful visual resizing result. Resize mode was turned off before the final observations.

## Coverage and acceptance boundary

[coverage.csv](coverage.csv) enumerates 24 production screens or presentation groups across seven poses and three window contexts, for 504 rows. It includes Models, all eight demos, and the normal-path sheets and system presentation groups identified from source. Source paths identify the owning screen, not an executed test. Every production acceptance field is `unverified`; fixture annotations identify available partial evidence. This inventory does not satisfy FOLD-02's requirement for a completed problem/no-problem assessment.

Tent posture, other-app Split View, simultaneous app-window layouts, expanded Live Activity, an active physical inner camera, all eight native demo bodies, and the full sheet/picker/export set remain unverified. Camera/status-bar/fold-region geometry in a synthetic fixture is not a pass for these production checks. No per-screen grid, presentation-placement, or toolbar change is inferred from an untested cell.

| Issue condition | Evidence and remaining work |
|---|---|
| FOLD-01 | Passed on main `4b9a9f8`: required PR checks and iOS Release build with Xcode 27.2. |
| FOLD-02 | Inventory supplied; production problem/no-problem assessment incomplete. |
| FOLD-03 | Bounded comparison and decisions supplied; owner review and physical validation pending. |
| FOLD-04 | Keep the existing host scrolling contract; owner review pending. |
| FOLD-05 | Retain current camera layout and `.resizeAspect`; native placement and per-screen overrides remain unverified. |
| FOLD-06 | No additional implementation adopted or Issue drafted; a global no-further-change conclusion is not established. |

FOLD-02 through FOLD-06 remain unchecked manual conditions. Issue #535 and the parent acceptance remain open. The two-block limit ends this prototype investigation; it does not convert unavailable device evidence into a pass or authorize more layout implementation.
