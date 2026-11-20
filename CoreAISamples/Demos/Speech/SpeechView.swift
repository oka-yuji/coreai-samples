import FeatureStore
import ModelLibraryFeature
import SpeechFeature
import SwiftUI

struct SpeechView: View {
    private struct OpenKey: Hashable {
        let path: String
        var identity: ModelArtifactIdentity?
        var location: String?
        var registration: Int?
        var grant: LibraryTicket?
    }

    private struct Attempt: Hashable {
        let key: OpenKey
        var verification: LibraryTicket?
    }

    @Bindable var store: Store<SpeechFeature>
    let model: DemoModel
    let limit: LocalizedStringKey
    let missing: LocalizedStringKey
    var voiceHint: LocalizedStringKey?
    var canPresent: @MainActor () -> Bool = { true }
    @Environment(MacModelLibrary.self) private var library
    @State private var attempted: Attempt?

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
        .onAppear {
            if canPresent() { Self.present(store, model: model, in: library) }
        }
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
        .onChange(of: model == .voxcpm2 ? nil : bundleToOpen, initial: true) {
            guard model != .voxcpm2, let attempt = bundleToOpen else { return }
            attempted = attempt
            store.send(.openBundle(attempt.key.path))
        }
        .background {
            if model == .voxcpm2, let expected = bundleToOpen {
                Color.clear.frame(width: 0, height: 0)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
                    .onAppear {
                        guard expected == bundleToOpen else { return }
                        attempted = expected
                        store.send(.openBundle(expected.key.path))
                    }
                    .id(expected)
            }
            if let ticket = store.state.exportTicket, let audio = store.state.audio {
                SpeechExportPresenter(
                    ticket: ticket, data: audio.wav, filename: model.libraryID + "-speech.wav", send: store.send
                )
                .id(ticket.id)
            }
        }
    }

    private var current: Attempt? {
        guard model == .voxcpm2 else {
            return library.resource(inUse: model).map { Attempt(key: OpenKey(path: $0.directory.path)) }
        }
        guard canPresent(), let state = library.store?.state,
            let artifact = state.selectedArtifact(for: model.libraryID),
            let installation = state.installations[artifact.id], installation.phase == .ready,
            let installed = installation.resource, let usage = state.usages[model.libraryID],
            let grant = usage.latestGrantTicket, grant.owner == state.owner, let resource = usage.resource,
            resource == installed, resource.registrationGeneration == installed.registrationGeneration
        else { return nil }
        let key = OpenKey(
            path: resource.directory.path, identity: resource.identity,
            location: resource.directory.standardizedFileURL.path, registration: resource.registrationGeneration,
            grant: grant)
        return Attempt(key: key, verification: installation.lastSuccessfulVerification)
    }

    private var canGenerate: Bool {
        guard let current, store.state.canGenerate(usingModelAt: current.key.path) else { return false }
        return model != .voxcpm2 || attempted?.key == current.key
    }

    private var bundleToOpen: Attempt? {
        guard let current, store.state.editable else { return nil }
        let path = current.key.path
        guard model == .voxcpm2 else {
            guard store.state.model == nil || store.state.lastPath != path, attempted?.key.path != path else {
                return nil
            }
            return current
        }
        guard let attempted, attempted.key == current.key else { return current }
        let opened = store.state.model != nil && store.state.lastPath == path
        return opened || attempted.verification == current.verification ? nil : current
    }
}
