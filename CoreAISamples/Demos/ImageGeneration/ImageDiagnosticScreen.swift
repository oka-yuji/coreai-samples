import FeatureStore
import Foundation
import ImageDiagnosticFeature
import ImageGenerationRuntime
import SwiftUI

struct ImageDiagnosticScreen: View {
    let configuration: URL
    @State private var store = Store<ImageDiagnosticFeature>(
        initialState: ImageDiagnosticState(), run: ImageDiagnosticRunner().run)
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Image generation diagnostics").font(.headline)
            Text(verbatim: store.state.message)
            if let outcome = store.state.outcome {
                Text(verbatim: outcome.directory).textSelection(.enabled)
            }
            if [.running, .stopping].contains(store.state.phase) { ProgressView() }
            Button("Stop") { store.send(.stop) }
                .disabled(![.running, .cleanupFailed].contains(store.state.phase))
        }
        .padding()
        .onAppear { store.send(.start(configuration)) }
        .onDisappear { store.send(.ended) }
        .onChange(of: scenePhase) { _, phase in if phase == .background { store.send(.stop) } }
        .onChange(of: store.state.phase) { _, phase in
            if phase == .finished { exit(ImageDiagnosticCommand.exitCode(store.state)) }
        }
    }
}

@MainActor
enum ImageDiagnosticCommand {
    static func run(configuration: URL) async -> Int32 {
        let store = Store<ImageDiagnosticFeature>(
            initialState: ImageDiagnosticState(), run: ImageDiagnosticRunner().run)
        store.send(.start(configuration))
        while store.state.phase != .finished && store.state.phase != .cleanupFailed {
            do { try await Task.sleep(for: .milliseconds(25)) } catch {
                store.send(.stop)
                await withTaskCancellationShield { try? await Task.sleep(for: .milliseconds(25)) }
            }
        }
        let text = (store.state.outcome?.directory ?? "") + "\n" + store.state.message + "\n"
        FileHandle.standardOutput.write(Data(text.utf8))
        return exitCode(store.state)
    }

    static func exitCode(_ state: ImageDiagnosticState) -> Int32 {
        switch state.outcome?.status {
        case .completed: 0
        case .cancelled: 2
        default: 1
        }
    }
}
