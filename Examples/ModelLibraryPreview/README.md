# Native model library preview

This iOS 27 sample hosts the shared SwiftUI `ModelLibraryView` without Core AI. It uses a synthetic catalog and in-memory transfer controls. It never downloads or reads model files and does not run inference. Keep it separate from the production app and its model directories.

From the repository root:

```sh
xcodegen generate --spec Examples/ModelLibraryPreview/project.yml
open Examples/ModelLibraryPreview/ModelLibraryPreview.xcodeproj
```

Select an iOS 27 simulator and run. The sample text model starts installed; the sample image model starts missing. Use **Transfer controls** to complete or fail a pending simulated download. Cancel, delete, reopen, search, and switch among ten sample demos. The controls inside a missing model's demo remain disabled. Each model/demo pair retains its sample input and result while the preview app runs. The preview's fixture controls are intentionally English; the reusable UI has English and Japanese resources.

Check the following manually:

- Models uses the system search toolbar, native navigation, and scalable text.
- Search preserves model ownership. Returning from a demo retains the search query.
- A missing model permits explanations and model information, but not demo controls.
- Starting, cancelling, completing, or failing a simulated download affects the shared model state.
- Deletion confirms the model and preserves its demo definitions.
- The largest Dynamic Type sizes, VoiceOver, and portrait/landscape layout remain usable.

A simulator build of the main Core AI app is not a substitute for this preview: Core AI is not available in the installed simulator SDK. The main app must be built with the device SDK.
