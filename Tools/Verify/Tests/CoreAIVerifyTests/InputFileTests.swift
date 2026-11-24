import Darwin
import Foundation
import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct InputFileTests {
    private func folder() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appending(path: "input-file-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        return url
    }

    @Test("A regular file of the expected size is read")
    func readsARegularFile() throws {
        let base = try folder()
        defer { try? FileManager.default.removeItem(at: base) }
        let bytes = Data([1, 2, 3, 4])
        try bytes.write(to: base.appending(path: "a.bin"))
        #expect(try InputFile.read(base.appending(path: "a.bin"), label: "a.bin", expectedBytes: 4) == bytes)
    }

    @Test("A regular file of any size is read when no size is expected, including an empty one")
    func readsAnySize() throws {
        let base = try folder()
        defer { try? FileManager.default.removeItem(at: base) }
        try Data([1, 2, 3]).write(to: base.appending(path: "a.bin"))
        try Data().write(to: base.appending(path: "empty.bin"))
        #expect(
            try InputFile.read(base.appending(path: "a.bin"), label: "a.bin", expectedBytes: nil) == Data([1, 2, 3]))
        #expect(try InputFile.read(base.appending(path: "empty.bin"), label: "empty.bin", expectedBytes: nil).isEmpty)
    }

    @Test(
        "A regular file that is shorter or longer than the expected size by one byte or more is rejected",
        arguments: [(3, 4), (5, 4), (0, 4), (4096, 4), (4, 0)])
    func rejectsAnotherSize(actual: Int, expected: Int) throws {
        let base = try folder()
        defer { try? FileManager.default.removeItem(at: base) }
        try Data(repeating: 7, count: actual).write(to: base.appending(path: "a.bin"))
        #expect(throws: FunctionError.fileSize("a.bin", expected: expected, actual: actual)) {
            try InputFile.read(base.appending(path: "a.bin"), label: "a.bin", expectedBytes: expected)
        }
    }

    @Test("A folder is rejected as not a regular file")
    func rejectsAFolder() throws {
        let base = try folder()
        defer { try? FileManager.default.removeItem(at: base) }
        try FileManager.default.createDirectory(at: base.appending(path: "d"), withIntermediateDirectories: false)
        #expect(throws: FunctionError.notRegularFile("d")) {
            try InputFile.read(base.appending(path: "d"), label: "d", expectedBytes: 4)
        }
        #expect(throws: FunctionError.notRegularFile("d")) {
            try InputFile.read(base.appending(path: "d"), label: "d", expectedBytes: nil)
        }
    }

    @Test("A named pipe is rejected as not a regular file without waiting for a writer and without reading its bytes")
    func rejectsANamedPipe() throws {
        let base = try folder()
        defer { try? FileManager.default.removeItem(at: base) }
        let path = base.appending(path: "p").path(percentEncoded: false)
        try #require(mkfifo(path, 0o600) == 0)
        let reader = open(path, O_RDONLY | O_NONBLOCK)
        let writer = open(path, O_WRONLY | O_NONBLOCK)
        defer {
            close(reader)
            close(writer)
        }
        try #require(reader >= 0 && writer >= 0)
        try #require(write(writer, "abcd", 4) == 4)
        #expect(throws: FunctionError.notRegularFile("p")) {
            try InputFile.read(base.appending(path: "p"), label: "p", expectedBytes: 4)
        }
        #expect(throws: FunctionError.notRegularFile("p")) {
            try InputFile.read(base.appending(path: "p"), label: "p", expectedBytes: nil)
        }
        var buffer = [UInt8](repeating: 0, count: 8)
        #expect(read(reader, &buffer, 8) == 4)
        #expect(Array(buffer.prefix(4)) == Array("abcd".utf8))
    }

    @Test("A link to a regular file and a link to a named pipe are rejected as not regular files")
    func rejectsALink() throws {
        let base = try folder()
        defer { try? FileManager.default.removeItem(at: base) }
        try Data([1, 2, 3, 4]).write(to: base.appending(path: "a.bin"))
        try FileManager.default.createSymbolicLink(
            at: base.appending(path: "link"), withDestinationURL: base.appending(path: "a.bin"))
        let pipe = base.appending(path: "p").path(percentEncoded: false)
        try #require(mkfifo(pipe, 0o600) == 0)
        try FileManager.default.createSymbolicLink(
            at: base.appending(path: "pipe-link"), withDestinationURL: base.appending(path: "p"))
        try FileManager.default.createSymbolicLink(
            at: base.appending(path: "dangling"), withDestinationURL: base.appending(path: "missing"))
        for name in ["link", "pipe-link", "dangling"] {
            #expect(throws: FunctionError.notRegularFile(name)) {
                try InputFile.read(base.appending(path: name), label: name, expectedBytes: 4)
            }
        }
    }

    @Test("A file that does not exist is rejected with the reason of the system")
    func rejectsAMissingFile() throws {
        let base = try folder()
        defer { try? FileManager.default.removeItem(at: base) }
        #expect(throws: FunctionError.cannot("read missing.bin", String(cString: strerror(ENOENT)))) {
            try InputFile.read(base.appending(path: "missing.bin"), label: "missing.bin", expectedBytes: 4)
        }
    }
}
