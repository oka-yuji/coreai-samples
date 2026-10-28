#if os(iOS)
    import FeatureStore
    import ModelLibraryFeature
    import ModelLibraryRuntime
    import SpeechFeature
    import SpeechRuntime
    import SwiftUI

    @MainActor
    final class SpeechAdapter: ModelDemoAdapter {
        let demoID = "speech"
        private let sessions = SpeechDemoSessions()

        func preview(for selection: DemoSelection) -> AnyView {
            var state = sessions.state(for: selection) ?? SpeechState(examples: Self.examples)
            state.presented = false
            state.phase = .idle
            state.audio = nil
            state.message = nil
            state.exportTicket = nil
            return AnyView(PhoneSpeechContent(state: state, send: { _ in }, isPermitted: { false }))
        }

        func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
            guard gate.session == session, gate.isPermitted() else { return preview(for: session.selection) }
            let store = sessions.bind(session, examples: Self.examples, permits: gate.isPermitted)
            return AnyView(PhoneSpeechStoreContent(store: store, isPermitted: gate.isPermitted).id(session.id))
        }

        func stop(_ session: ModelDemoSession) async throws {
            await sessions.stop(session)
        }

        static var examples: [SpeechExample] {
            guard let url = Bundle.main.url(forResource: "speech-examples", withExtension: "json"),
                let data = try? Data(contentsOf: url),
                let values = try? JSONDecoder().decode([SpeechExample].self, from: data)
            else { return [] }
            return values
        }
    }

    private struct PhoneSpeechStoreContent: View {
        let store: Store<SpeechFeature>
        let isPermitted: @MainActor @Sendable () -> Bool

        var body: some View {
            PhoneSpeechContent(state: store.state, send: store.send, isPermitted: isPermitted)
        }
    }
#endif
