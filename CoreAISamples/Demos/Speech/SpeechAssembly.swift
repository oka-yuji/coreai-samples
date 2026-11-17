import FeatureStore
import Foundation
import SpeechFeature
import SpeechRuntime

@MainActor
enum SpeechAssembly {
    static func makeStore(_ engine: SpeechEngine = .irodori) -> Store<SpeechFeature> {
        LiveSpeechService.makeStore(engine: engine, examples: (try? examples(engine)) ?? [])
    }

    static func examples(_ engine: SpeechEngine) throws -> [SpeechExample] {
        let resource =
            switch engine {
            case .irodori: "speech-examples"
            case .voxcpm2: "voxcpm2-speech-examples"
            }
        guard let url = Bundle.main.url(forResource: resource, withExtension: "json") else {
            throw CocoaError(.fileNoSuchFile)
        }
        return try JSONDecoder().decode([SpeechExample].self, from: Data(contentsOf: url))
    }
}
