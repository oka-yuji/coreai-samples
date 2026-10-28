import FeatureStore
import SpeechFeature
import SpeechRuntime
import SwiftUI

@main
struct SpeechPhoneApp: App {
    @State private var speech = SpeechPhoneAssembly.speech()
    @State private var library = SpeechPhoneAssembly.library()
    @State private var checks = SpeechPhoneAssembly.checks()
    private let checkRequest = SpeechPhoneAssembly.checkRequest()
    var body: some Scene {
        WindowGroup {
            if let checkRequest {
                SpeechCheckView(store: checks, request: checkRequest)
            } else {
                SpeechPhoneView(speech: speech, library: library)
            }
        }
    }
}

@MainActor
enum SpeechPhoneAssembly {
    static func examples() -> [SpeechExample] {
        let examples: [SpeechExample]
        if let url = Bundle.main.url(forResource: "speech-examples", withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let decoded = try? JSONDecoder().decode([SpeechExample].self, from: data)
        {
            examples = decoded
        } else {
            examples = []
        }
        return examples
    }
    static func speech() -> Store<SpeechFeature> {
        Store(initialState: SpeechState(examples: examples()), run: SpeechRunner(service: LiveSpeechService()).run)
    }
    static func library() -> Store<SpeechLibraryFeature> {
        Store(
            initialState: SpeechLibraryState(),
            run: SpeechLibraryRunner(library: SpeechLibrary(directory: .documentsDirectory)).run)
    }
    static func checks() -> Store<SpeechCheckFeature> {
        Store(
            initialState: SpeechCheckState(),
            run: SpeechCheckRunner(service: PhoneSpeechCheck(documents: .documentsDirectory, examples: examples())).run)
    }
    static func checkRequest() -> SpeechCheckRequest? {
        let args = CommandLine.arguments
        func value(_ flag: String) -> String? {
            guard let index = args.firstIndex(of: flag), args.indices.contains(index + 1) else { return nil }
            return args[index + 1]
        }
        guard let runID = value("--speech-check-run") else { return nil }
        return SpeechCheckRequest(
            folder: value("--speech-model-folder") ?? "sampler-irodori-fp32", runID: runID,
            referenceAudio: value("--reference-audio"), playAudio: !args.contains("--no-playback"))
    }

}
