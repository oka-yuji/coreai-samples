import FeatureStore
import SpeechFeature
import SwiftUI

struct SpeechView: View {
    @Bindable var store: Store<SpeechFeature>
    let model: DemoModel
    let limit: LocalizedStringKey
    let missing: LocalizedStringKey
    var voiceHint: LocalizedStringKey?
    @Environment(MacModelLibrary.self) private var library
    @State private var attempted: String?

    var body: some View {
        SpeechForm(
            store: store, title: model.title, limit: limit, missing: library.installed(model) == nil ? missing : nil,
            voiceHint: voiceHint,
            canGenerate: store.state.canGenerate(usingModelAt: library.resource(inUse: model)?.directory.path)
        )
        .navigationTitle(model.title)
        .onAppear {
            store.send(.appeared)
            library.use(model)
        }
        .onDisappear {
            store.send(.ended)
            library.stopUsing(model)
            attempted = nil
        }
        .onChange(of: library.resource(inUse: model)?.directory) { _, new in
            if new == nil { attempted = nil }
        }
        .onChange(of: bundleToOpen, initial: true) { _, path in
            guard let path else { return }
            attempted = path
            store.send(.openBundle(path))
        }
        .background {
            if let ticket = store.state.exportTicket, let audio = store.state.audio {
                SpeechExportPresenter(
                    ticket: ticket, data: audio.wav, filename: model.libraryID + "-speech.wav", send: store.send
                )
                .id(ticket.id)
            }
        }
    }

    private var bundleToOpen: String? {
        guard let path = library.resource(inUse: model)?.directory.path, store.state.editable,
            store.state.model == nil || store.state.lastPath != path, attempted != path
        else { return nil }
        return path
    }
}
