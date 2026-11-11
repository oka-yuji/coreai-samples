# Native model library preview

This iOS 27 sample hosts the shared SwiftUI `ModelLibraryView` without Core AI. It uses a synthetic catalog and in-memory transfer controls. It never downloads or reads model files and does not run inference. Keep it separate from the production app and its model directories.

From the repository root:

```sh
xcodegen generate --spec Examples/ModelLibraryPreview/project.yml
open Examples/ModelLibraryPreview/ModelLibraryPreview.xcodeproj
```

Select an iOS 27 simulator and run. The sample text model starts installed; the sample image model starts missing and has FP16 and INT8 variants, which you can switch in Model information. Use **Preview controls** to complete or fail a pending simulated download. Cancel, delete, reopen, search, and switch among ten sample demos. The controls inside a missing model's demo remain disabled. Each preview window owns its own Store, driver and renderer. Each model/demo pair retains its sample input and result while that window remains alive. The preview's fixture controls are intentionally English; the reusable UI has English and Japanese resources.

## Duo and stop-barrier checks

Use Xcode 27.1 or later and a runtime that actually lists iPhone Duo among its supported device types. `xcrun simctl list runtimes -j` exposes the runtime build and `supportedDeviceTypes`. In the November 11 check, iOS 27.1 build `24A94425` supported Duo, as did the separately installed iOS 27.2 build `24B91016s`; iOS 27.2 build `24B8117` did not. A matching version number or an installed device-type definition alone is insufficient.

1. In Closed, open an installed sample demo. Use **Set sample input** and **Show sample result** to establish nondefault state without relying on accessibility setters to commit a text edit.
2. In Device Hub, choose Open and Partially Open, then Closed. Check the two-column and single-column layouts, the selected demo, and the unchanged input/result. Also check both open orientations and reselect the current tile from the sidebar.
3. Enable **Hold demo stops** in Preview controls. In Closed and Open, use **Back to Models**. Detail must remain visible, with the return control disabled while stopping.
4. Choose **Fail pending stops**. The failure and **Retry stopping** must remain in detail. Retry, then choose **Complete pending stops**; only that completion returns to Models.
5. In a two-column layout, switch demos with stops held. The old detail remains until completion, then the new demo appears. Disable held stops when finished with these cases.

The delayed stop uses a cancellable stream and explicit fixture controls, not a timed sleep. No file import, acquisition lease or model execution is validated by this synthetic driver. These checks do not replace physical-device, real-model, VoiceOver or owner acceptance.

Check the following manually:

- Models uses the system search toolbar, native navigation, and scalable text.
- Search preserves model ownership. Returning from a demo retains the search query.
- A missing model permits explanations and model information, but not demo controls.
- Starting, cancelling, completing, or failing a simulated download affects the shared model state.
- Deletion confirms the model and preserves its demo definitions.
- The largest Dynamic Type sizes, VoiceOver, and portrait/landscape layout remain usable.

A simulator build of the main Core AI app is not a substitute for this preview: Core AI is not available in the installed simulator SDK. The main app must be built with the device SDK.
