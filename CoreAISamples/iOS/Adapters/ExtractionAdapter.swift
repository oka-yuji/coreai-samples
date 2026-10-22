#if os(iOS)
    import ExtractFeature
    import ExtractRuntime
    import FeatureStore
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import SwiftUI

    @MainActor
    final class ExtractionAdapter: ModelDemoAdapter {
        private struct Demo {
            let session: ModelDemoSession
            let store: Store<ExtractFeature>
            let runner: ManagedExtractRunner
        }

        let demoID = "extraction"
        private var demos: [DemoSelection: Demo] = [:]

        func preview(for selection: DemoSelection) -> AnyView {
            let state = demos[selection]?.store.state ?? Self.initialState
            return AnyView(PhoneExtractContent(state: state, text: .constant(state.text)) { _ in })
        }

        func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
            guard gate.session == session, gate.isPermitted() else { return preview(for: session.selection) }
            return AnyView(ManagedExtractView(store: demo(for: session, gate: gate).store))
        }

        func stop(_ session: ModelDemoSession) async throws {
            guard let demo = demos[session.selection], demo.session.id == session.id else { return }
            demo.store.send(.closed)
            try await demo.runner.release()
        }

        private func demo(for session: ModelDemoSession, gate: DemoExecutionGate) -> Demo {
            if let demo = demos[session.selection], demo.session.id == session.id { return demo }
            let identity = session.identity
            let state = (demos[session.selection]?.store.state ?? Self.initialState).managed(
                path: session.resource.directory.path(percentEncoded: false),
                identity: "\(identity.modelID)/\(identity.artifactID)@\(identity.revision)")
            let runner = ManagedExtractRunner(
                service: GLiNERExtractService(correctionWords: ExtractResources.correctionWords),
                isPermitted: gate.isPermitted, describe: ExtractRunner.describe)
            let demo = Demo(session: session, store: Store(initialState: state, run: runner.run), runner: runner)
            demos[session.selection] = demo
            return demo
        }

        private static var initialState: ExtractState {
            var state = ExtractState(form: ExtractResources.form, labels: ExtractResources.examples.labels)
            state.text = ExtractResources.examples.notes.first ?? ""
            return state
        }
    }

    private struct ManagedExtractView: View {
        let store: Store<ExtractFeature>

        var body: some View {
            PhoneExtractContent(
                state: store.state,
                text: Binding(get: { store.state.text }, set: { store.send(.textChanged($0)) }),
                send: store.send)
        }
    }
#endif
