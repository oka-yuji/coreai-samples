import EmbeddingCore
import Foundation
import Testing

@testable import EmbeddingRuntime

@Suite("Embedding runtime boundaries")
struct EmbeddingRuntimeTests {
    @Test("Validate enumerated bundle files independently of filesystem path aliases")
    func bundleEnumeration() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let names = [
            "tokenizer.json", "tokenizer_config.json", "token_embeddings.bf16", "MODEL_LICENSE.txt",
            "text.aimodel/metadata.json", "vision.aimodel/metadata.json",
        ]
        var hashes: [String: String] = [:]
        for name in names {
            let file = root.appending(path: name)
            try FileManager.default.createDirectory(
                at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("fixture".utf8).write(to: file)
            hashes[name] = try EmbeddingBundle.digest(file)
        }
        let manifest: [String: Any] = [
            "format": 1, "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
            "precision": "fp16-safe-s128-v70", "files": hashes,
        ]
        try JSONSerialization.data(withJSONObject: manifest).write(to: root.appending(path: "embedding-manifest.json"))
        _ = try EmbeddingBundle(directory: root)
        _ = try EmbeddingBundle(directory: root.resolvingSymlinksInPath())
        try Data("changed".utf8).write(to: root.appending(path: "tokenizer.json"))
        #expect(throws: (any Error).self) { try EmbeddingBundle(directory: root) }
    }

    @Test("Decode grayscale PNG, grayscale JPEG, and CMYK JPEG as RGB")
    func colorModes() throws {
        for name in ["L.png", "L.jpg", "CMYK.jpg"] {
            let url = try #require(Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures"))
            let image = try EmbeddingImageReader.read(url)
            #expect(image.width == 3 && image.height == 2)
            #expect(image.pixels.count == 3 * 2 * 3)
            if name.hasPrefix("L.") {
                #expect(image.pixels[0] == image.pixels[1] && image.pixels[1] == image.pixels[2])
            }
        }
    }

    @Test("Reject missing manifests and paths outside the model directory")
    func bundleBoundaries() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: (any Error).self) { try EmbeddingBundle(directory: root) }
        #expect(throws: (any Error).self) { try EmbeddingBundle.containedFile("../outside", root: root) }
        #expect(throws: (any Error).self) { try EmbeddingBundle.containedFile("/outside", root: root) }
        try FileManager.default.createSymbolicLink(
            at: root.appending(path: "escape"), withDestinationURL: root.deletingLastPathComponent())
        #expect(throws: (any Error).self) { try EmbeddingBundle.containedFile("escape/file", root: root) }
        #expect(try EmbeddingBundle.containedFile("inside/file", root: root) == root.appending(path: "inside/file"))
    }

    @Test("Detect modified files using SHA256")
    func checksum() throws {
        let file = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        try Data("abc".utf8).write(to: file)
        #expect(try EmbeddingBundle.digest(file) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        try Data("abd".utf8).write(to: file)
        #expect(try EmbeddingBundle.digest(file) != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    @Test("Release busy state after failed operations")
    func failureRecovery() async throws {
        let model = EmbeddingModel()
        await #expect(throws: EmbeddingError.notLoaded) { try await model.encode("test") }
        await #expect(throws: EmbeddingError.notLoaded) { try await model.encode(imageURL: URL(filePath: "/missing")) }
        try await model.unload()
        await #expect(throws: (any Error).self) { try await model.load(directory: URL(filePath: "/missing")) }
        await #expect(throws: EmbeddingError.notLoaded) { try await model.encode("test") }
    }

    @Test("Reject overlapping work and release ownership after cancellation", .timeLimit(.minutes(1)))
    func cancellation() async throws {
        let entered = AsyncStream<Void>.makeStream()
        let resume = AsyncStream<Void>.makeStream()
        let model = EmbeddingModel {
            entered.continuation.yield(())
            var iterator = resume.stream.makeAsyncIterator()
            _ = await iterator.next()
            try Task.checkCancellation()
        }
        let loading = Task { try await model.load(directory: URL(filePath: "/missing")) }
        var iterator = entered.stream.makeAsyncIterator()
        _ = await iterator.next()
        await #expect(throws: EmbeddingError.busy) { try await model.encode("test") }
        await #expect(throws: EmbeddingError.busy) { try await model.unload() }
        loading.cancel()
        await #expect(throws: CancellationError.self) { try await loading.value }
        try await model.unload()
        await #expect(throws: EmbeddingError.notLoaded) { try await model.encode("test") }
    }

    @Test(
        "Match reference token IDs and reject overlong text",
        .enabled(if: ProcessInfo.processInfo.environment["EG2_TOKENIZER"] != nil))
    func tokenization() throws {
        struct Sample: Decodable {
            let text: String
            let role: EmbeddingTextRole
            let tokens: [Int]
        }
        let path = try #require(ProcessInfo.processInfo.environment["EG2_TOKENIZER"])
        let tokenizer = try EmbeddingTokenizer(directory: URL(filePath: path))
        let file = try #require(
            Bundle.module.url(forResource: "tokens", withExtension: "json", subdirectory: "Fixtures"))
        let samples = try JSONDecoder().decode([Sample].self, from: Data(contentsOf: file))
        for sample in samples {
            #expect(try tokenizer.tokens(for: sample.text, role: sample.role) == sample.tokens)
        }
        #expect(throws: (any Error).self) {
            try tokenizer.tokens(for: String(repeating: "word ", count: 200), role: .query)
        }
    }
}
