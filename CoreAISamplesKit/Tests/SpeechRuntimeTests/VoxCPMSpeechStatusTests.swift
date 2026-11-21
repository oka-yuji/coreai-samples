import FeatureStore
import Foundation
import SpeechCore
import SpeechFeature
import SpeechRuntime
import Testing

@MainActor
struct VoxCPMSpeechStatusTests {
    private struct FailingService: SpeechService {
        let error: SpeechError
        func open(path: String) async throws -> SpeechModel { throw error }
        func generate(text: String, seed: UInt64) async throws -> SpeechAudio { throw error }
        func generate(text: String, seed: UInt64, reference: SpeechReferenceAudio?) async throws -> SpeechAudio {
            throw error
        }
        func translate(voice: String) async throws -> String { "translated" }
        func play(_ data: Data) async throws { throw error }
        func stop() async {}
    }

    @Test(
        "VCU-03 The speech runner preserves the measured Vox token count and limit for the active generation",
        arguments: [144, 110, 79])
    func limits(limit: Int) async throws {
        let error = SpeechError.textExceedsLimit(count: limit + 1, limit: limit)
        var state = SpeechState()
        _ = SpeechFeature.reduce(&state, .appeared)
        state.model = SpeechModel(name: "Synthetic", directory: URL(filePath: "/model"), supportsReferenceAudio: true)
        state.text = "forward the original text"
        state.seed = "1708"
        if limit != 144 {
            state.reference = try SpeechReferenceAudio(
                samples: Array(repeating: 0.25, count: limit == 110 ? 80_000 : 160_000), source: .file("fixture.wav"),
                format: .voxcpm2)
        }
        let retained = state.reference
        let commands = SpeechFeature.reduce(&state, .generate)
        guard case .run(let effect, _)? = commands.first else {
            Issue.record("The real reducer did not start a generation.")
            return
        }
        let session = state.session
        let action = SpeechAction.textLimitExceeded(session, count: limit + 1, limit: limit)
        var received: [SpeechAction] = []
        await SpeechRunner(service: FailingService(error: error)).run(effect) {
            received.append($0)
            _ = SpeechFeature.reduce(&state, $0)
        }
        #expect(received == [action])
        #expect(state.phase == .idle && state.statusMessage == .tokenLimit(count: limit + 1, limit: limit))
        #expect(state.message == error.errorDescription && state.audio == nil && state.reference == retained)
        #expect(state.text == "forward the original text" && state.seed == "1708")
        let settled = state
        _ = SpeechFeature.reduce(&state, .textLimitExceeded(session, count: 999, limit: 3))
        #expect(state == settled)
        _ = SpeechFeature.reduce(&state, .setText("new text"))
        #expect(state.message == nil && state.statusMessage == nil)
        _ = SpeechFeature.reduce(&state, .generate)
        let next = state
        _ = SpeechFeature.reduce(&state, action)
        #expect(state == next)
        _ = SpeechFeature.reduce(&state, .failed(state.session, "ordinary failure"))
        #expect(state.message == "ordinary failure" && state.statusMessage == .text("ordinary failure"))
        state.message = nil
        #expect(state.statusMessage == nil)
    }

    @Test(
        "VCU-03 A translated Vox generation uses the typed error while old and string-shaped failures keep their existing actions"
    )
    func compatibility() async {
        let limit = SpeechError.textExceedsLimit(count: 96, limit: 95)
        let failures: [(SpeechError, SpeechEffect, SpeechAction)] = [
            (limit, .generateTranslated(7, "voice", "body", 1), .textLimitExceeded(7, count: 96, limit: 95)),
            (.textTooLong(256), .generate(7, "body", 1), .failed(7, SpeechError.textTooLong(256).localizedDescription)),
            (
                .invalidAsset(limit.localizedDescription), .generate(7, "body", 1),
                .failed(7, limit.localizedDescription)
            ),
            (limit, .open(7, "/invalid"), .failed(7, limit.localizedDescription)),
            (.invalidSamples, .play(7, Data()), .failed(7, SpeechError.invalidSamples.localizedDescription)),
        ]
        for (error, effect, expected) in failures {
            var got: [SpeechAction] = []
            await SpeechRunner(service: FailingService(error: error)).run(effect) { got.append($0) }
            #expect(got == [expected])
        }
        var state = SpeechState()
        state.message = "previous"
        #expect(state.statusMessage == .text("previous"))
        state.statusMessage = .tokenLimit(count: 111, limit: 110)
        #expect(state.message == SpeechError.textExceedsLimit(count: 111, limit: 110).errorDescription)
        state.message = "replacement"
        #expect(state.statusMessage == .text("replacement"))
    }
}
