#if os(macOS)
    import OCRUI
    import SwiftUI

    struct OCRHostView: View {
        let library: OCRMacLibrary

        var body: some View {
            Group {
                if let store = library.store, let rendering = library.rendering {
                    if store.state.pendingDestination != nil {
                        VStack(spacing: 12) {
                            ProgressView("Stopping OCR…")
                            if store.state.failure != nil {
                                Text("OCR could not finish stopping. Try again.")
                                Button("Try again") { store.send(.retryDemoStop) }
                            }
                        }
                    } else if let session = store.state.session {
                        rendering.content(for: session)
                    } else {
                        VStack(alignment: .leading, spacing: 16) {
                            OCRModelControls(store: store) { store.send(.openDemo($0)) }
                            Spacer()
                        }.padding(24)
                    }
                } else {
                    ContentUnavailableView {
                        Label("OCR is unavailable", systemImage: "exclamationmark.triangle")
                    } description: {
                        Text(LocalizedStringKey(library.problem ?? "The OCR catalog could not be loaded."))
                    }
                }
            }
            .demoNavigationTitle("TeleOCR")
            .onAppear { library.refresh() }
        }
    }
#endif
