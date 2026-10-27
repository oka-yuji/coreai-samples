import CryptoKit
import Foundation
import IrodoriInputLogic
import IrodoriTokenizers
import SpeechCore
import Testing

@testable import SpeechAssets
@testable import SpeechRuntime

@Test("Pinned normalization matches the Unicode reference corpus")
func normalizationCorpus() throws {
    struct Entry: Decodable {
        let input: String
        let output: String
    }
    let url = try #require(
        Bundle.module.url(forResource: "normalization", withExtension: "json", subdirectory: "Resources"))
    let entries = try JSONDecoder().decode([Entry].self, from: Data(contentsOf: url))
    #expect(entries.count == 283)
    for item in entries {
        #expect(Array(try IrodoriText.normalize(item.input).utf8) == Array(item.output.utf8))
    }
}

@Test("Noise and token layout retain the verified input contract")
func pinnedInputs() throws {
    let values = try IrodoriNoise.generate(frames: 13, seed: 1708)
    let digest = values.withUnsafeBytes { SHA256.hash(data: Data($0)) }.map { String(format: "%02x", $0) }.joined()
    #expect(digest == "1eb73ea9a5ae6976cd07cf901b2e7fb6419c9783bc654913a8e6b67a7603636f")
    let input = try IrodoriTextInput(normalizedText: "<pad>", bodyTokenIDs: [3], originalTokenCount: 300)
    #expect(input.originalTokenCount == 300)
    #expect(throws: IrodoriInputError.self) {
        try IrodoriTextInput(normalizedText: "x", bodyTokenIDs: [3], originalTokenCount: 0)
    }
    #expect(input.tokenIDs.prefix(3) == [1, 3, 3])
    #expect(input.attentionMask.prefix(3) == [true, true, false])
    #expect(throws: IrodoriInputError.self) { try IrodoriNoise.generate(frames: 751, seed: 1) }
}

@Test("Unsupported bundle metadata and changed files are rejected")
func bundleValidation() throws {
    for manifest in [
        SpeechManifest(
            format: "other", modelID: "Aratako/Irodori-TTS-v4.1-Small-MF", precision: "fp32", sampleRate: 48000),
        SpeechManifest(format: "irodori-coreai-v1", modelID: "other", precision: "fp32", sampleRate: 48000),
        SpeechManifest(
            format: "irodori-coreai-v1", modelID: "Aratako/Irodori-TTS-v4.1-Small-MF", precision: "fp16",
            sampleRate: 48000),
    ] {
        #expect(throws: SpeechError.self) { try manifest.validate() }
    }
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let manifest = SpeechManifest(
        format: "irodori-coreai-v1", modelID: "Aratako/Irodori-TTS-v4.1-Small-MF", precision: "fp32", sampleRate: 48000)
    try JSONEncoder().encode(manifest).write(to: directory.appendingPathComponent("irodori.json"))
    #expect(throws: (any Error).self) { try ValidatedSpeechBundle.open(directory) }
    let file = directory.appendingPathComponent("bytes")
    try Data("abc".utf8).write(to: file)
    #expect(try hashFile(file) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    #expect(throws: (any Error).self) { try IrodoriTokenizer(tokenizerJSON: file) }
}

@Test("All pinned noise shapes and seeds match the CPU reference")
func noiseCorpus() throws {
    struct Entry: Decodable {
        let frames: Int
        let seed: String
        let sha256: String
    }
    let url = try #require(Bundle.module.url(forResource: "noise", withExtension: "json", subdirectory: "Resources"))
    let entries = try JSONDecoder().decode([Entry].self, from: Data(contentsOf: url))
    #expect(entries.count == 49)
    for entry in entries {
        let seed = try #require(UInt64(entry.seed))
        let values = try IrodoriNoise.generate(frames: entry.frames, seed: seed)
        let digest = values.withUnsafeBytes { SHA256.hash(data: Data($0)) }.map { String(format: "%02x", $0) }.joined()
        #expect(digest == entry.sha256)
    }
}
