import Testing
import VoxCPMCore

@Suite("VoxCPM2 prompt token IDs")
struct VoxCPMPromptTests {
    @Test(
        "For each fixture sentence the split token IDs with the audio start token match the original implementation",
        arguments: ["c1_plain", "c2_design", "c3_reference", "space_prefixed_kanji"])
    func splitIDs(name: String) throws {
        let reference = try VoxCPMReference.load()
        let prompt = try #require(reference.prompts.first { $0.name == name })
        let vocabulary = Dictionary(uniqueKeysWithValues: reference.vocabulary.map { ($0.id, $0.token) })
        #expect(try VoxCPMPrompt.inputIDs(textTokenIDs: prompt.tokenIDs, vocabulary: vocabulary) == prompt.expectedIDs)
    }
}
