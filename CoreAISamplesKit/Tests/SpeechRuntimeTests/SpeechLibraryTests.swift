import Foundation
import SpeechCore
import SpeechFeature
import Testing

@testable import SpeechRuntime

@Suite(.timeLimit(.minutes(1)))
struct SpeechLibraryTests {
    @Test("Only visible local directories with a model marker are listed in stable order")
    func directories() async throws {
        let root = try phoneTestDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        for name in ["beta", "alpha", ".hidden", "incomplete"] {
            let directory = root.appendingPathComponent(name, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            if name != "incomplete" { try Data().write(to: directory.appendingPathComponent("irodori.json")) }
        }
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("alias"),
            withDestinationURL: root.appendingPathComponent("alpha"))
        let values = try await SpeechLibrary(directory: root).folders()
        #expect(values.map(\.name) == ["alpha", "beta"])
        await #expect(throws: (any Error).self) {
            try await SpeechLibrary(directory: root.appendingPathComponent("missing")).folders()
        }
    }

    @Test("The folder runner reports failures and suppresses results after cancellation")
    @MainActor
    func runner() async throws {
        let failure = FakeFolderListing()
        await failure.configure(fails: true)
        failure.gate.open()
        var actions: [SpeechLibraryAction] = []
        await SpeechLibraryRunner(library: failure).run(.load(3)) { actions.append($0) }
        #expect(actions == [.failed(3, SpeechError.invalidSamples.localizedDescription)])
        let slow = FakeFolderListing()
        let task = Task { await SpeechLibraryRunner(library: slow).run(.load(4)) { actions.append($0) } }
        defer {
            slow.gate.open()
            task.cancel()
        }
        try await slow.started.wait()
        task.cancel()
        slow.gate.open()
        await task.value
        #expect(actions.count == 1)
    }

    @Test("Check output refuses unsafe names, existing runs and duplicate files")
    func outputProtection() async throws {
        let root = try phoneTestDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let files = SpeechCheckFiles(documents: root)
        for value in ["", ".", "..", "a/b", "a\n", "with space", String(repeating: "a", count: 81)] {
            #expect(!SpeechCheckFiles.validName(value))
            await #expect(throws: (any Error).self) { try await files.reserve(value) }
        }
        let folder = try await files.reserve("run-1")
        try await files.writeWAV(Data([1]), index: 0)
        await #expect(throws: (any Error).self) { try await files.writeWAV(Data([2]), index: 0) }
        #expect(try Data(contentsOf: folder.appendingPathComponent("sample-0.wav")) == Data([1]))
        await #expect(throws: (any Error).self) { try await files.reserve("run-1") }
        await #expect(throws: (any Error).self) { try await files.writeWAV(Data([2]), index: 1) }
        #expect(!FileManager.default.fileExists(atPath: folder.appendingPathComponent("sample-1.wav").path))
        await #expect(throws: (any Error).self) { try await files.modelFolder("../other") }
    }
    @Test("A model named like the check output folder is never modified by diagnostics")
    func modelOutputCollision() async throws {
        let root = try phoneTestDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("SpeechChecks", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let original = Data([1, 2, 3])
        try original.write(to: model.appendingPathComponent("irodori.json"))
        let files = SpeechCheckFiles(documents: root)
        await #expect(throws: (any Error).self) { try await files.reserve("attempt") }
        #expect(try FileManager.default.contentsOfDirectory(atPath: model.path) == ["irodori.json"])
        #expect(try Data(contentsOf: model.appendingPathComponent("irodori.json")) == original)
        for name in ["SpeechChecks", "speechchecks", "SPEECHCHECKS"] {
            await #expect(throws: (any Error).self) { try await files.modelFolder(name) }
        }
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("alias"), withDestinationURL: model)
        await #expect(throws: (any Error).self) { try await files.modelFolder("alias") }
    }

}
