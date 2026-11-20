import AppKit
import FeatureStore
import Foundation
import SpeechFeature
import SpeechRuntime
import SwiftUI

@MainActor
enum SpeechAssembly {
    static func makeStore(_ engine: SpeechEngine = .irodori) -> Store<SpeechFeature> {
        LiveSpeechService.makeStore(engine: engine, examples: (try? examples(engine)) ?? [])
    }

    static func makeVoiceDesignStore(library: MacModelLibrary) -> Store<SpeechFeature> {
        LiveSpeechService.makeStore(
            engine: .voxcpm2, examples: (try? examples(.voxcpm2)) ?? [],
            acquireVoxCPM: library.executionLeaseAcquirer(for: .voxcpm2))
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

    static func checkMinimumHeight(_ check: (String, Bool, String) -> Void) {
        _ = NSApplication.shared
        let guidance = LocalizedStringKey(
            String(repeating: "This guidance is long enough to wrap across many lines in a narrow window. ", count: 3))
        let cases: [(name: String, voiceHint: LocalizedStringKey?)] = [
            ("withVoiceDescription", guidance), ("withoutVoiceDescription", nil),
        ]
        for (name, voiceHint) in cases {
            let form = SpeechForm(
                store: Store<SpeechFeature>(initialState: SpeechState()) { _, _ in }, title: "VoxCPM2", limit: guidance,
                missing: guidance, voiceHint: voiceHint, canGenerate: false)
            let height = NSHostingController(rootView: form).sizeThatFits(in: .zero).height
            check(
                "speech/minimumHeight/" + name, height <= DemoApp.defaultSize.height,
                "Minimum height \(height) pt; default window height \(DemoApp.defaultSize.height) pt.")
        }
    }

    static func checkPhaseText(_ check: (String, Bool, String) -> Void) {
        for phase in SpeechState.Phase.allCases {
            let english = SpeechForm.phaseText(phase, locale: Locale(identifier: "en"))
            let japanese = SpeechForm.phaseText(phase, locale: Locale(identifier: "ja"))
            check(
                "speech/phaseText/\(phase)", !japanese.isEmpty && japanese != english,
                "English: \(english); Japanese: \(japanese)")
        }
    }
}
