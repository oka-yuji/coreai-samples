import SpokenSearchRuntime
import SwiftUI

struct SpokenSearchPreview: View {
    @State private var session: SpokenSearchSession?
    @State private var owner: UUID?

    var body: some View {
        Group {
            if let session {
                SpokenSearchScreen(store: session.store)
            } else {
                Color.clear.frame(height: 0)
            }
        }
        .onAppear {
            guard session == nil else { return }
            let id = UUID()
            owner = id
            session = SpokenSearchSession { owner == id }
        }
        .onDisappear {
            owner = nil
            session?.store.send(.disappeared)
            session = nil
        }
    }
}
