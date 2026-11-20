import Foundation
import SpokenSearchFeature

@MainActor
protocol SpokenTranscribing: AnyObject {
    func transcribe(_ url: URL) async throws -> SpokenTranscript
    func cancel() async
}

extension SpokenFileTranscriber: SpokenTranscribing {}
