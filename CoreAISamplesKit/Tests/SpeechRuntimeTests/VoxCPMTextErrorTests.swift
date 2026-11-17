import Foundation
import SpeechCore
import Testing
import VoxCPMTokenizers

@testable import SpeechRuntime

@Suite(.timeLimit(.minutes(1)))
struct VoxCPMTextErrorTests {
    static let directory = URL(filePath: "/unused-model")

    @Test(
        "VUI-05 Text the tokenizer cannot read fails with a sentence that says what to change and the runtime stays usable"
    )
    func unreadableText() async throws {
        let driver = FakeVoxCPMDriver()
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies())
        _ = try await runtime.open(Self.directory)
        await #expect(throws: SpeechError.unreadableText) { try await runtime.generate(text: "a\u{E9}b", seed: 1) }
        #expect(await driver.factoryCount == 0)
        let audio = try await runtime.generate(text: "ab", seed: 1)
        #expect(audio.normalizedText == "ab")
        #expect(await driver.factoryCount == 1)
        #expect(
            SpeechError.unreadableText.errorDescription
                == "The model cannot read some characters in this text. Remove or replace them, then try again.")
        await runtime.release()
    }

    @Test("VUI-05 A tokenizer that cannot be created fails the open with a sentence instead of an internal error")
    func tokenizerCreation() async throws {
        var dependencies = FakeVoxCPMDriver().dependencies()
        dependencies.tokenizer = { _ in throw VoxCPMTokenizerError.invalidVocabulary }
        let runtime = VoxCPMRuntime(dependencies: dependencies)
        await #expect(throws: SpeechError.invalidAsset("The VoxCPM2 tokenizer could not be loaded.")) {
            try await runtime.open(Self.directory)
        }
        await #expect(throws: SpeechError.modelNotOpened("VoxCPM2")) { try await runtime.generate(text: "ab", seed: 1) }
    }
}
