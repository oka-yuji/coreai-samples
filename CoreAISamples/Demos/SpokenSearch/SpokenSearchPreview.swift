import SpokenSearchRuntime
import SwiftUI

struct SpokenSearchPreview: View {
    @State private var session: SpokenSearchSession?

    var body: some View {
        Group {
            if let session {
                SpokenSearchScreen(store: session.store)
            } else {
                Color.clear.frame(height: 0)
            }
        }
        .onAppear {
            if session == nil { session = SpokenSearchSession() }
        }
    }
}
