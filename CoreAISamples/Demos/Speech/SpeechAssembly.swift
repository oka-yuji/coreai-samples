import FeatureStore
import Foundation
import SpeechFeature
import SpeechRuntime

@MainActor
enum SpeechAssembly {
    static func makeStore(_ engine: SpeechEngine = .irodori) -> Store<SpeechFeature> {
        let resource =
            switch engine {
            case .irodori: "speech-examples"
            case .voxcpm2: "voxcpm2-speech-examples"
            }
        let examples: [SpeechExample]
        if let url = Bundle.main.url(forResource: resource, withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let loaded = try? JSONDecoder().decode([SpeechExample].self, from: data)
        {
            examples = loaded
        } else {
            examples = []
        }
        return LiveSpeechService.makeStore(engine: engine, examples: examples)
    }
}
