import Foundation
import Testing

@testable import ReceiptRuntime

struct ReceiptExportBoundaryTests {
    @Test("An unrecorded staging entry is neither exported nor deleted")
    func additionalFile() throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let output = try files.folder()
        let identifier = UUID()
        let transaction = try ReceiptExportTransaction(directory: output, identifier: identifier)
        try transaction.write(Data("generated".utf8), path: "receipt.png")
        let staging = output.appending(path: ".ReceiptDesk-" + identifier.uuidString + ".partial")
        let sentinel = staging.appending(path: "keep.txt")
        try Data("keep".utf8).write(to: sentinel)
        #expect(throws: ReceiptRuntimeError.invalidDestination) { try transaction.publish() }
        #expect(throws: ReceiptRuntimeError.cleanupFailed) { try transaction.cleanUp() }
        #expect(try Data(contentsOf: sentinel) == Data("keep".utf8))
    }

    @Test("Publication rejects a same-size rewrite of the retained output inode")
    func changedContents() throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let output = try files.folder()
        let identifier = UUID()
        let transaction = try ReceiptExportTransaction(directory: output, identifier: identifier)
        try transaction.write(Data("original".utf8), path: "receipt.png")
        let staging = output.appending(path: ".ReceiptDesk-" + identifier.uuidString + ".partial")
        let handle = try FileHandle(forWritingTo: staging.appending(path: "receipt.png"))
        try handle.write(contentsOf: Data("modified".utf8))
        try handle.close()
        #expect(throws: ReceiptRuntimeError.sourceMismatch) { try transaction.publish() }
        #expect(
            !FileManager.default.fileExists(atPath: output.appending(path: "ReceiptDesk-" + identifier.uuidString).path)
        )
        try transaction.cleanUp()
    }

    @Test("Replacing staging with a symlink cannot redirect writes or publish that symlink")
    func stagingLink() throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let output = try files.folder()
        let identifier = UUID()
        let transaction = try ReceiptExportTransaction(directory: output, identifier: identifier)
        let staging = output.appending(path: ".ReceiptDesk-" + identifier.uuidString + ".partial")
        let moved = files.root.appending(path: "owned-stage")
        try FileManager.default.moveItem(at: staging, to: moved)
        let unrelated = try files.folder("unrelated")
        let sentinel = unrelated.appending(path: "keep.txt")
        try Data("keep".utf8).write(to: sentinel)
        try FileManager.default.createSymbolicLink(at: staging, withDestinationURL: unrelated)
        #expect(throws: ReceiptRuntimeError.invalidDestination) {
            try transaction.write(Data("ledger".utf8), path: "ledger.csv")
        }
        #expect(throws: ReceiptRuntimeError.invalidDestination) { try transaction.publish() }
        #expect(throws: ReceiptRuntimeError.cleanupFailed) { try transaction.cleanUp() }
        #expect(try Data(contentsOf: sentinel) == Data("keep".utf8))
        #expect(!FileManager.default.fileExists(atPath: unrelated.appending(path: "ledger.csv").path))
    }

    @Test("Cleanup preserves a replacement staging directory and its unrelated files")
    func stagingReplacement() throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let output = try files.folder()
        let identifier = UUID()
        let transaction = try ReceiptExportTransaction(directory: output, identifier: identifier)
        try transaction.write(Data("generated".utf8), path: "ledger.csv")
        let staging = output.appending(path: ".ReceiptDesk-" + identifier.uuidString + ".partial")
        try FileManager.default.moveItem(at: staging, to: files.root.appending(path: "owned-stage"))
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        let sentinel = staging.appending(path: "keep.txt")
        try Data("keep".utf8).write(to: sentinel)
        #expect(throws: ReceiptRuntimeError.cleanupFailed) { try transaction.cleanUp() }
        #expect(try Data(contentsOf: sentinel) == Data("keep".utf8))
    }

    @Test("Replaced nested folders and files cannot be published or recursively removed")
    func nestedReplacement() throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let output = try files.folder()
        let identifier = UUID()
        let transaction = try ReceiptExportTransaction(directory: output, identifier: identifier)
        try transaction.write(Data("generated".utf8), path: "2026/11/receipt.png")
        let staging = output.appending(path: ".ReceiptDesk-" + identifier.uuidString + ".partial")
        let nested = staging.appending(path: "2026")
        try FileManager.default.moveItem(at: nested, to: files.root.appending(path: "owned-nested"))
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: false)
        let sentinel = nested.appending(path: "keep.txt")
        try Data("keep".utf8).write(to: sentinel)
        #expect(throws: ReceiptRuntimeError.invalidDestination) { try transaction.publish() }
        #expect(throws: ReceiptRuntimeError.cleanupFailed) { try transaction.cleanUp() }
        #expect(try Data(contentsOf: sentinel) == Data("keep".utf8))
        let other = UUID()
        let second = try ReceiptExportTransaction(directory: output, identifier: other)
        try second.write(Data("generated".utf8), path: "ledger.csv")
        let target = output.appending(path: ".ReceiptDesk-" + other.uuidString + ".partial/ledger.csv")
        try FileManager.default.removeItem(at: target)
        try Data("replacement".utf8).write(to: target)
        #expect(throws: ReceiptRuntimeError.invalidDestination) { try second.publish() }
        #expect(throws: ReceiptRuntimeError.cleanupFailed) { try second.cleanUp() }
        #expect(try Data(contentsOf: target) == Data("replacement".utf8))
    }
}
