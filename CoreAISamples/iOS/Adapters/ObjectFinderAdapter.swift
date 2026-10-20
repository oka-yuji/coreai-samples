#if os(iOS)
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import ObjectFinderFeature
    import ObjectFinderRuntime
    import SwiftUI

    @MainActor
    final class ObjectFinderAdapter: ModelDemoAdapter {
        let demoID = "object-finder"
        private let sessions = ObjectFinderDemoSessions<ObjectFinderPreview>()

        func preview(for selection: DemoSelection) -> AnyView {
            let state = sessions.runtime(for: selection)?.store.state ?? ObjectFinderState()
            return AnyView(ObjectFinderInputs(query: state.query, minimumSimilarity: state.minimumSimilarity))
        }

        func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
            let runtime = sessions.bind(session, permits: gate.isPermitted)
            return AnyView(
                ObjectFinderContent(store: runtime.store, preview: runtime.preview, model: session.resource.artifactID)
                    .id(ObjectIdentifier(runtime.store)))
        }

        func stop(_ session: ModelDemoSession) async throws {
            try await sessions.stop(session)
        }
    }

    private struct ObjectFinderInputs: View {
        let query: String
        let minimumSimilarity: Double

        var body: some View {
            VStack(alignment: .leading, spacing: 10) {
                Text("Object to find").font(.headline)
                TextField("Type any object name", text: .constant(query))
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Object to find")
                HStack {
                    Text("Match threshold")
                    Spacer()
                    Text(minimumSimilarity, format: .number.precision(.fractionLength(2)))
                }
                Button("Start searching", systemImage: "viewfinder") {}
                    .buttonStyle(.borderedProminent)
            }
            .disabled(true)
        }
    }
#endif
