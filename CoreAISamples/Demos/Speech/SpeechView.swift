import FeatureStore
import ModelLibraryFeature
import SpeechFeature
import SwiftUI

struct SpeechView: View {
    private struct OpenKey: Equatable {
        let path: String
        var identity: ModelArtifactIdentity?
        var location: String?
        var registration: Int?
    }

    @Bindable var store: Store<SpeechFeature>
    let model: DemoModel
    let limit: LocalizedStringKey
    let missing: LocalizedStringKey
    var voiceHint: LocalizedStringKey?
    @Environment(MacModelLibrary.self) private var library
    @State private var attempted: OpenKey?

    static func present(_ store: Store<SpeechFeature>, model: DemoModel, in library: MacModelLibrary) {
        guard !store.state.presented else { return }
        store.send(.appeared)
        library.use(model)
    }

    static func dismiss(_ store: Store<SpeechFeature>, model: DemoModel, in library: MacModelLibrary) {
        guard store.state.presented else { return }
        store.send(.ended)
        library.stopUsing(model)
    }

    var body: some View {
        SpeechForm(
            store: store, title: model.title, limit: limit, missing: library.installed(model) == nil ? missing : nil,
            voiceHint: voiceHint, canGenerate: canGenerate
        )
        .navigationTitle(model.title)
        .onAppear { Self.present(store, model: model, in: library) }
        .onDisappear {
            Self.dismiss(store, model: model, in: library)
            attempted = nil
        }
        .onChange(of: store.state.presented) { _, presented in
            if !presented { attempted = nil }
        }
        .onChange(of: library.resource(inUse: model)?.directory) { _, new in
            if new == nil { attempted = nil }
        }
        .onChange(of: bundleToOpen, initial: true) { _, key in
            guard let key else { return }
            attempted = key
            store.send(.openBundle(key.path))
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

    private var openKey: OpenKey? {
        guard let resource = library.resource(inUse: model) else { return nil }
        guard model == .voxcpm2 else { return OpenKey(path: resource.directory.path) }
        return OpenKey(
            path: resource.directory.path, identity: resource.identity,
            location: resource.directory.standardizedFileURL.path, registration: resource.registrationGeneration)
    }

    private var canGenerate: Bool {
        let key = openKey
        guard store.state.canGenerate(usingModelAt: key?.path) else { return false }
        return model != .voxcpm2 || attempted == key
    }

    private var bundleToOpen: OpenKey? {
        guard let key = openKey, store.state.editable else { return nil }
        if model == .voxcpm2 { return attempted == key ? nil : key }
        guard store.state.model == nil || store.state.lastPath != key.path, attempted?.path != key.path else {
            return nil
        }
        return key
    }
}
