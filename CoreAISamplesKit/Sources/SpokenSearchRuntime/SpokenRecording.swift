import Foundation
import SpokenSearchFeature

@MainActor
final class SpokenRecording {
    let url: URL
    let generation: Int
    var transcript: SpokenTranscript?
    private let scoped: Bool

    init(_ url: URL, generation: Int) {
        self.url = url
        self.generation = generation
        scoped = url.startAccessingSecurityScopedResource()
    }

    deinit {
        if scoped { url.stopAccessingSecurityScopedResource() }
    }
}
