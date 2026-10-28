import Darwin
import FeatureStore
import SpeechFeature
import SwiftUI

struct SpeechCheckView: View {
    @Bindable var store: Store<SpeechCheckFeature>
    let request: SpeechCheckRequest
    @Environment(\.scenePhase) private var scenePhase
    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 20) {
                Label("Irodori device check", systemImage: "waveform.path.ecg").font(.title2)
                Text(store.state.message).textSelection(.enabled)
                if store.state.phase == .running || store.state.phase == .stopping { ProgressView() }
                if store.state.phase == .running {
                    Button("Stop", role: .destructive) { store.send(.stop) }
                }
                Text("Keep this screen open. Normal launches do not start this check.")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
            }
            .padding()
            .navigationTitle("Device check")
        }
        .onAppear { store.send(.start(request)) }
        .onDisappear { store.send(.stop) }
        .onChange(of: scenePhase) { _, phase in if phase == .background { store.send(.stop) } }
        .onChange(of: store.state.phase) { _, phase in
            if phase == .finished, CommandLine.arguments.contains("--exit-after-speech-check") {
                exit(store.state.succeeded == true ? 0 : 1)
            }
        }
    }
}
