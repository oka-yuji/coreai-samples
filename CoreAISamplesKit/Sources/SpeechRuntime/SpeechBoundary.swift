#if os(macOS) || os(iOS)
    import Foundation
    import IrodoriInputLogic
    import IrodoriTokenizers
    import SpeechCore

    protocol SpeechContext: Sendable {
        var frames: Int { get }
    }

    protocol SpeechPipeline: Sendable {
        @concurrent func prepare(_ input: IrodoriTextInput) async throws -> any SpeechContext
        @concurrent func generate(_ context: any SpeechContext, seed: UInt64) async throws -> [Float]
    }

    protocol SpeechTokenizer {
        func prepare(_ text: String) throws -> IrodoriTextInput
    }

    extension IrodoriTokenizer: SpeechTokenizer {}

    struct SpeechDependencies: Sendable {
        var validate: @Sendable (URL) throws -> ValidatedSpeechBundle
        var tokenizer: @Sendable (URL) throws -> any SpeechTokenizer
        var pipeline: @Sendable ([URL]) async throws -> any SpeechPipeline
        var didRequestRelease: @Sendable () -> Void = {}

        static let live = SpeechDependencies(
            validate: { try ValidatedSpeechBundle.open($0) },
            tokenizer: { try IrodoriTokenizer(tokenizerJSON: $0) },
            pipeline: { try await Pipeline(assets: $0) })
    }

    public protocol SpeechGenerating: Actor {
        func open(_ directory: URL) async throws -> SpeechModel
        func generate(text: String, seed: UInt64) async throws -> SpeechAudio
        func release() async
    }
#endif
