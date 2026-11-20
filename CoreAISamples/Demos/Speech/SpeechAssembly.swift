import AppKit
import FeatureStore
import Foundation
import SpeechCore
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
        checkReferenceHeight(check)
    }

    private struct ReferenceLayoutCase {
        let id: String
        let profile: String
        let supportsReference: Bool
        let referenceSamples: Int
        let phase: SpeechState.Phase
        let typedLimit: (count: Int, limit: Int)?
    }

    private static let referenceLayoutCases = [
        ReferenceLayoutCase(
            id: "legacy", profile: "fp16", supportsReference: false, referenceSamples: 0, phase: .idle, typedLimit: nil),
        ReferenceLayoutCase(
            id: "empty", profile: "reference_fp32", supportsReference: true, referenceSamples: 0, phase: .idle,
            typedLimit: nil),
        ReferenceLayoutCase(
            id: "selected", profile: "reference_fp32", supportsReference: true, referenceSamples: 80_000, phase: .idle,
            typedLimit: nil),
        ReferenceLayoutCase(
            id: "recording", profile: "reference_fp32", supportsReference: true, referenceSamples: 0,
            phase: .recording, typedLimit: nil),
        ReferenceLayoutCase(
            id: "limitError", profile: "reference_fp32", supportsReference: true, referenceSamples: 160_000,
            phase: .idle, typedLimit: (80, 79)),
    ]

    private static func referenceState(_ layout: ReferenceLayoutCase) throws -> SpeechState {
        var state = SpeechState(examples: (try? examples(.voxcpm2)) ?? [])
        state.presented = true
        let model = SpeechModel(
            name: "VoxCPM2", directory: URL(filePath: "/VoxCPM2/" + layout.profile),
            supportsReferenceAudio: layout.supportsReference)
        state.model = model
        state.lastPath = model.directory.path
        state.phase = layout.phase
        if layout.referenceSamples > 0 {
            state.reference = try SpeechReferenceAudio(
                samples: Array(repeating: 0.25, count: layout.referenceSamples), source: .file("reference-voice.wav"),
                format: .voxcpm2)
        }
        if let typed = layout.typedLimit { state.statusMessage = .tokenLimit(count: typed.count, limit: typed.limit) }
        return state
    }

    private static func checkReferenceHeight(_ check: (String, Bool, String) -> Void) {
        for layout in referenceLayoutCases {
            for language in ["en", "ja"] {
                let name = "speech/minimumHeight/voxcpm2/" + layout.id + "/" + language
                let store: Store<SpeechFeature>
                do {
                    store = Store<SpeechFeature>(initialState: try referenceState(layout)) { _, _ in }
                } catch {
                    check(name, false, error.localizedDescription)
                    continue
                }
                let form = SpeechForm(
                    store: store, title: "VoxCPM2",
                    limit: "Short passages work best. The text and the voice description share a limit of 144 tokens.",
                    missing: nil,
                    voiceHint:
                        "Optional. Describe the voice in a few words, for example: A young woman, gentle and calm voice. A description that contains hiragana or katakana is translated to English on this Mac before it reaches the model; if the translation changes, the same seed can give different audio. Without a description the model chooses the voice.",
                    canGenerate: store.state.canGenerate, showsVoxReference: layout.supportsReference)
                let height = NSHostingController(rootView: form.environment(\.locale, Locale(identifier: language)))
                    .sizeThatFits(in: .zero).height
                let state = store.state
                let typed =
                    if case .tokenLimit(let count, let limit)? = state.statusMessage {
                        "\(count)/\(limit)"
                    } else {
                        "none"
                    }
                let maximum = DemoApp.defaultSize.height
                check(
                    name, height.isFinite && height >= 0 && height <= maximum,
                    "Minimum height \(height) pt; default window height \(maximum) pt. profile=\(layout.profile)"
                        + " phase=\(state.phase) referenceSamples=\(state.reference?.samples.count ?? 0)"
                        + " supportsReference=\(state.model?.supportsReferenceAudio == true)"
                        + " typedLimit=\(typed) locale=\(language)")
            }
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
        for (count, limit) in [(145, 144), (111, 110), (80, 79)] {
            let english = SpeechTokenLimitPresentation.message(
                count: count, limit: limit, locale: Locale(identifier: "en"))
            let japanese = SpeechTokenLimitPresentation.message(
                count: count, limit: limit, locale: Locale(identifier: "ja"))
            let numbers = [String(count), String(limit)]
            check(
                "speech/tokenLimit/\(limit)/en",
                english == SpeechError.textExceedsLimit(count: count, limit: limit).errorDescription, english)
            check(
                "speech/tokenLimit/\(limit)/ja",
                japanese != english && numbers.allSatisfy { japanese.contains($0) }, japanese)
        }
    }
}
