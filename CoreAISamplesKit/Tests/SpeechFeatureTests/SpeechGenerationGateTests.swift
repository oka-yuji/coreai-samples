import Foundation
import SpeechCore
import SpeechFeature
import Testing

struct SpeechGenerationGateTests {
    func opened(_ path: String) -> SpeechState {
        var state = SpeechState(examples: [SpeechExample(id: "a", title: "A", text: "Test")])
        _ = SpeechFeature.reduce(&state, .appeared)
        _ = SpeechFeature.reduce(&state, .openBundle(path))
        _ = SpeechFeature.reduce(
            &state, .opened(state.session, SpeechModel(name: "Model", directory: URL(filePath: path))))
        return state
    }

    @Test("VUI-02 Generation is possible only while the folder in use is the folder that was opened")
    func folderInUse() {
        let state = opened("/models/a")
        #expect(state.canGenerate && state.lastPath == "/models/a")
        #expect(state.canGenerate(usingModelAt: "/models/a"))
        #expect(!state.canGenerate(usingModelAt: "/models/b"))
        #expect(!state.canGenerate(usingModelAt: nil))
    }

    @Test("VUI-02 Without an acquired model or an opened model generation is unavailable")
    func notAcquired() {
        var saved = SpeechState(lastPath: "/models/a")
        _ = SpeechFeature.reduce(&saved, .appeared)
        _ = SpeechFeature.reduce(&saved, .setText("Test"))
        #expect(!saved.canGenerate(usingModelAt: "/models/a"))
        #expect(!saved.canGenerate(usingModelAt: nil))
        #expect(!SpeechState().canGenerate(usingModelAt: nil))
        var managed = SpeechState(
            managedModel: SpeechModel(name: "Model", directory: URL(filePath: "/models/a")),
            examples: [SpeechExample(id: "a", title: "A", text: "Test")])
        _ = SpeechFeature.reduce(&managed, .appeared)
        #expect(managed.canGenerate && managed.lastPath == nil)
        #expect(!managed.canGenerate(usingModelAt: nil))
        var removed = opened("/models/a")
        _ = SpeechFeature.reduce(&removed, .bundleUnavailable)
        #expect(!removed.canGenerate(usingModelAt: nil))
        #expect(!removed.canGenerate(usingModelAt: "/models/a"))
    }

    @Test("VUI-02 The folder check keeps the other conditions for generation")
    func otherConditions() {
        var state = opened("/models/a")
        _ = SpeechFeature.reduce(&state, .setText(" "))
        #expect(!state.canGenerate(usingModelAt: "/models/a"))
        _ = SpeechFeature.reduce(&state, .setText("Test"))
        _ = SpeechFeature.reduce(&state, .setSeed("x"))
        #expect(!state.canGenerate(usingModelAt: "/models/a"))
        _ = SpeechFeature.reduce(&state, .setSeed("1708"))
        _ = SpeechFeature.reduce(&state, .generate)
        #expect(!state.canGenerate(usingModelAt: "/models/a"))
    }
}
