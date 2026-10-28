#if os(iOS)
    import FeatureStore
    import Darwin
    import SpeechFeature
    import SpeechRuntime
    import SwiftUI

    struct PhoneSpeechCheckScreen: View {
        @State private var store: Store<SpeechCheckFeature>
        @Environment(\.scenePhase) private var scenePhase
        private let request: SpeechCheckRequest

        init(runID: String) {
            let service = PhoneSpeechCheck(documents: .documentsDirectory, examples: SpeechAdapter.examples)
            _store = State(
                initialValue: Store(initialState: SpeechCheckState(), run: SpeechCheckRunner(service: service).run))
            request = SpeechCheckRequest(
                folder: LaunchArguments.value("--speech-model-folder") ?? "Irodori-Reference",
                runID: runID,
                referenceAudio: LaunchArguments.value("--reference-audio"),
                playAudio: false)
        }

        var body: some View {
            VStack(spacing: 20) {
                Text("Irodori device check").font(.title2)
                Text(store.state.message).textSelection(.enabled)
                if store.state.phase == .running || store.state.phase == .stopping {
                    ProgressView()
                    Button("Stop") { store.send(.stop) }.disabled(store.state.phase == .stopping)
                }
            }
            .padding()
            .onAppear { store.send(.start(request)) }
            .onDisappear { store.send(.stop) }
            .onChange(of: scenePhase) { _, phase in
                if phase == .background { store.send(.stop) }
            }
            .onChange(of: store.state.phase) { _, phase in
                if phase == .finished, CommandLine.arguments.contains("--exit-after-speech-check") {
                    exit(store.state.succeeded == true ? 0 : 1)
                }
            }
        }
    }
#endif
