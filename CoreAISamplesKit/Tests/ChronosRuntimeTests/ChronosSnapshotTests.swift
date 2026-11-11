import ChronosCore
import Foundation
import Testing

@testable import ChronosRuntime

struct ChronosSnapshotTests {
    @Test(
        "A verified snapshot survives original asset replacement and is removed with its owner",
        arguments: [false, true])
    func independent(useStream: Bool) async throws {
        let source = try ChronosTestFolder()
        let destination = try ChronosTestFolder()
        defer {
            source.clean()
            destination.clean()
        }
        try source.writeBundle()
        let expected = try await ChronosBundle.open(source.url)
        var snapshot: ChronosBundleSnapshot? = try await ChronosBundleSnapshot.open(
            source.url, parent: destination.url,
            copy: useStream ? ChronosBundleSnapshot.streamCopy : ChronosBundleSnapshot.copyFile)
        let location = try #require(snapshot?.bundle.url)
        #expect(snapshot?.bundle.identity == expected.identity)
        #expect(location != source.url)
        let original = source.url.appendingPathComponent("Model.aimodel/main.mlirb")
        try FileManager.default.removeItem(at: original)
        try Data("replacement".utf8).write(to: original)
        #expect(try await ChronosBundle.open(location).identity == expected.identity)
        #expect(
            try Data(contentsOf: location.appendingPathComponent("Model.aimodel/main.mlirb"))
                == Data("Model.aimodel/main.mlirb".utf8))
        snapshot = nil
        #expect(!FileManager.default.fileExists(atPath: location.path))
    }

    @Test("Copy corruption and copy failure reject the snapshot and remove incomplete files", arguments: [false, true])
    func rejected(corrupt: Bool) async throws {
        let source = try ChronosTestFolder()
        let destination = try ChronosTestFolder()
        defer {
            source.clean()
            destination.clean()
        }
        try source.writeBundle()
        await #expect(throws: (any Error).self) {
            try await ChronosBundleSnapshot.open(source.url, parent: destination.url) { descriptor, url in
                if corrupt {
                    try ChronosBundleSnapshot.copyFile(descriptor, to: url)
                    if url.lastPathComponent == "main.mlirb" { try Data("changed".utf8).write(to: url) }
                } else {
                    throw ChronosError.invalidBundle("injected copy failure")
                }
            }
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: destination.url.path).isEmpty)
        #expect(try await ChronosBundle.open(source.url).identity.files.count == 8)
    }
}
