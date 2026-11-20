import SpokenSearchRuntime
import SwiftUI

struct SpokenSearchPreview: View {
    var source: SpokenTopicSource?
    @Environment(\.scenePhase) private var phase
    @State private var session: SpokenSearchSession?
    @State private var latestSource: SpokenTopicSource?
    @State private var owner: UUID?
    @State private var dismissal: AsyncStream<Never>.Continuation?

    var body: some View {
        Group {
            if let session {
                SpokenSearchScreen(store: session.store)
            } else {
                Color.clear.frame(height: 0)
            }
        }
        .task(id: phase == .background) {
            guard phase != .background, !Task.isCancelled else { return }
            let id = UUID()
            let (lifetime, signal) = AsyncStream<Never>.makeStream()
            owner = id
            dismissal = signal
            let active = SpokenSearchSession { owner == id }
            active.setTopicSource(latestSource)
            session = active
            var iterator = lifetime.makeAsyncIterator()
            _ = await iterator.next()
            if owner == id { dismissRecording() }
            await active.close()
        }
        .onChange(of: source?.identity, initial: true) { _, _ in
            latestSource = source
            session?.setTopicSource(source)
        }
        .onChange(of: phase) { _, next in
            if next == .background { dismissRecording() }
        }
        .onDisappear { dismissRecording() }
    }

    private func dismissRecording() {
        owner = nil
        session?.suspend()
        session = nil
        dismissal?.finish()
        dismissal = nil
    }
}
