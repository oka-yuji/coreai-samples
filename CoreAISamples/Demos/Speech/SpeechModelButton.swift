import FeatureStore
import Foundation
import SpeechFeature
import SwiftUI
import UniformTypeIdentifiers

struct SpeechModelButton: View {
    let store: Store<SpeechFeature>
    @Environment(DemoNavigator.self) private var navigator
    @State private var importing = false

    var body: some View {
        Button("Open speech model…", systemImage: "folder") { importing = true }
            .disabled(store.state.busy || store.state.exportVisible)
            .accessibilityIdentifier("speech.open")
            .fileImporter(isPresented: $importing, allowedContentTypes: [.folder]) { result in
                switch result {
                case .success(let url):
                    store.send(.appeared)
                    store.send(.openBundle(url.path))
                    navigator.selection = .speech
                case .failure(let error):
                    if (error as? CocoaError)?.code != .userCancelled {
                        store.send(.appeared)
                        store.send(.uiError("The selected speech folder could not be opened."))
                        navigator.selection = .speech
                    }
                }
            }
    }
}
