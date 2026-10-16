import FeatureStore
import Foundation
import SpeechFeature
import SpeechRuntime

@MainActor
enum SpeechAssembly {
    static func makeStore() -> Store<SpeechFeature> {
        let examples: [SpeechExample]
        if let url = Bundle.main.url(forResource: "speech-examples", withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let loaded = try? JSONDecoder().decode([SpeechExample].self, from: data)
        {
            examples = loaded
        } else {
            examples = []
        }
        let service = LiveSpeechService()
        return Store(
            initialState: SpeechState(
                examples: examples,
                lastPath: UserDefaults.standard.string(forKey: LiveSpeechService.modelPathKey)),
            run: SpeechRunner(service: service).run)
    }
}
