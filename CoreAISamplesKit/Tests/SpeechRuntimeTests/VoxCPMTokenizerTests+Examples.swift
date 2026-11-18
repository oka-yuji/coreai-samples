import Foundation
import SpeechFeature
import Testing
import VoxCPMCore
import VoxCPMTokenizers

extension VoxCPMTokenizerTests {
    @Test(
        "Every VoxCPM2 example of the Mac app has at most 144 text tokens after the kanji split",
        .enabled(if: ProcessInfo.processInfo.environment["VOXCPM2_TOKENIZER"] != nil))
    func appExamples() throws {
        let folder = try #require(ProcessInfo.processInfo.environment["VOXCPM2_TOKENIZER"])
        let tokenizer = try VoxCPMTokenizer(tokenizerJSON: URL(filePath: folder).appending(path: "tokenizer.json"))
        let file = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "CoreAISamples/Demos/Speech/Resources/voxcpm2-speech-examples.json")
        let examples = try JSONDecoder().decode([SpeechExample].self, from: Data(contentsOf: file))
        #expect(examples.map(\.id) == ["plain", "calm", "steady"])
        for example in examples {
            let ids = try VoxCPMPrompt.inputIDs(
                textTokenIDs: tokenizer.tokenIDs(for: VoxCPMText.normalized(example.prompt)),
                vocabulary: tokenizer.vocabulary)
            print("VoxCPM2 example \(example.id): \(ids.count - 1) text tokens")
            #expect(ids.count > 1 && ids.count - 1 <= 144, "\(example.id)")
        }
    }
}
