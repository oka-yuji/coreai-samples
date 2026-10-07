import DecideFeature
import DecideRuntime
import Foundation
import Testing

struct DecideFilesTests {
    @Test("UTF-8 text and Markdown import preserves content and unsupported files are refused")
    func importing() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let files = DecideFiles()
        for name in ["document.txt", "document.md"] {
            let url = root.appendingPathComponent(name)
            try Data("Document\r\nSecond line".utf8).write(to: url)
            #expect(try await files.read(url) == "Document\r\nSecond line")
        }
        await #expect(throws: DecideServiceError.self) {
            try await files.read(root.appendingPathComponent("document.pdf"))
        }
        let oversized = root.appendingPathComponent("large.txt")
        try Data(repeating: 65, count: 4 * 1_024 * 1_024 + 1).write(to: oversized)
        await #expect(throws: DecideServiceError.self) { try await files.read(oversized) }
        let largeLink = root.appendingPathComponent("large-link.txt")
        try FileManager.default.createSymbolicLink(at: largeLink, withDestinationURL: oversized)
        await #expect(throws: DecideServiceError.self) { try await files.read(largeLink) }
        let smallLink = root.appendingPathComponent("small-link.txt")
        try FileManager.default.createSymbolicLink(
            at: smallLink, withDestinationURL: root.appendingPathComponent("document.txt"))
        #expect(try await files.read(smallLink) == "Document\r\nSecond line")
        let invalid = root.appendingPathComponent("invalid.txt")
        try Data([0xFF]).write(to: invalid)
        await #expect(throws: (any Error).self) { try await files.read(invalid) }
    }

    @Test("Export creates an atomic JSON file at the selected local path")
    func writing() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("review.json")
        let data = Data("{\"review\":true}".utf8)
        try await DecideFiles().write(data, to: url)
        #expect(try Data(contentsOf: url) == data)
    }
}
