#if os(iOS)
    import FeatureStore
    import Foundation
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import ObjectFinderFeature
    import ObjectFinderRuntime
    import SwiftUI

    extension ObjectFinderAssembly {
        struct ManagedSession {
            let store: Store<ObjectFinderFeature>
            let preview: ObjectFinderPreview
            let runner: ObjectFinderRunner
        }

        static func makeManagedSession(
            resource: ValidatedModelResource, gate: @escaping @MainActor () -> Bool,
            retaining previous: ObjectFinderState? = nil
        ) -> ManagedSession {
            let preview = ObjectFinderPreview()
            let runner = ObjectFinderRunner(service: LiveObjectFinderService(preview: preview), permits: gate)
            let store = Store<ObjectFinderFeature>(
                initialState: .managed(folder: resource.directory, retaining: previous), run: runner.run)
            return ManagedSession(store: store, preview: preview, runner: runner)
        }
    }

    @MainActor
    final class ObjectFinderAdapter: ModelDemoAdapter {
        private struct Entry {
            let resource: ValidatedModelResource
            let session: ObjectFinderAssembly.ManagedSession
            var gate: DemoExecutionGate
        }

        let demoID = "object-finder"
        private var entries: [DemoSelection: Entry] = [:]

        func preview(for selection: DemoSelection) -> AnyView {
            let state = entries[selection]?.session.store.state ?? ObjectFinderState()
            return AnyView(ObjectFinderInputs(query: state.query, minimumSimilarity: state.minimumSimilarity))
        }

        func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
            let managed = bind(session, gate: gate)
            return AnyView(
                ObjectFinderContent(store: managed.store, preview: managed.preview).id(ObjectIdentifier(managed.store)))
        }

        func stop(_ session: ModelDemoSession) async throws {
            guard let entry = entries[session.selection] else { return }
            let managed = entry.session
            guard entry.gate.session.id == session.id || managed.store.state.phase == .idle else { return }
            try await managed.runner.release(managed.store.state.session, send: managed.store.send)
        }

        private func bind(_ session: ModelDemoSession, gate: DemoExecutionGate) -> ObjectFinderAssembly.ManagedSession {
            let selection = session.selection
            let resource = session.resource
            if let entry = entries[selection], entry.resource == resource {
                entries[selection]?.gate = gate
                return entry.session
            }
            let managed = ObjectFinderAssembly.makeManagedSession(
                resource: resource,
                gate: { [weak self] in
                    guard let entry = self?.entries[selection], entry.resource == resource else { return false }
                    return entry.gate.isPermitted()
                }, retaining: entries[selection]?.session.store.state)
            entries[selection] = Entry(resource: resource, session: managed, gate: gate)
            return managed
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
