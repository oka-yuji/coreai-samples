import ChronosCore
import Darwin
import Foundation
import OSLog

final class ChronosBundleSnapshot: Sendable {
    let bundle: ChronosBundle

    private init(bundle: ChronosBundle) {
        self.bundle = bundle
    }

    deinit {
        do {
            try FileManager.default.removeItem(at: bundle.url)
        } catch {
            Logger(subsystem: "CoreAISamples", category: "ChronosRuntime").error(
                "Chronos temporary bundle cleanup failed.")
        }
    }

    @concurrent
    static func open(
        _ source: URL,
        parent: URL = FileManager.default.temporaryDirectory,
        copy: @Sendable (Int32, URL) throws -> Void = ChronosBundleSnapshot.copyFile
    ) async throws -> ChronosBundleSnapshot {
        let original = try await ChronosBundle.open(source)
        let directory = try ChronosFiles.openDirectory(source)
        defer { Darwin.close(directory) }
        let destination = parent.appendingPathComponent("chronos-model-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: destination, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        do {
            try FileManager.default.createDirectory(
                at: destination.appendingPathComponent("Model.aimodel"), withIntermediateDirectories: false,
                attributes: [.posixPermissions: 0o700])
            for path in ["chronos2.json"] + original.identity.files.keys.sorted() {
                try Task.checkCancellation()
                let descriptor = try ChronosFiles.openFile(path, in: directory)
                do {
                    defer { Darwin.close(descriptor) }
                    try copy(descriptor, destination.appendingPathComponent(path))
                }
            }
            let verified = try await ChronosBundle.open(destination)
            guard verified.identity == original.identity else {
                throw ChronosError.invalidBundle("snapshot identity changed")
            }
            try Task.checkCancellation()
            return ChronosBundleSnapshot(bundle: verified)
        } catch {
            do {
                try FileManager.default.removeItem(at: destination)
            } catch {
                Logger(subsystem: "CoreAISamples", category: "ChronosRuntime").error(
                    "Chronos failed snapshot cleanup failed.")
            }
            throw error
        }
    }

    static func copyFile(_ source: Int32, to destination: URL) throws {
        if fclonefileat(source, AT_FDCWD, destination.path, 0) == 0 {
            guard chflags(destination.path, 0) == 0, chmod(destination.path, 0o600) == 0 else {
                throw ChronosFiles.failure()
            }
            return
        }
        try streamCopy(source, to: destination)
    }

    static func streamCopy(_ source: Int32, to destination: URL) throws {
        let descriptor = Darwin.open(destination.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else { throw ChronosFiles.failure() }
        defer { Darwin.close(descriptor) }
        var info = stat()
        guard fstat(source, &info) == 0, info.st_size >= 0 else { throw ChronosFiles.failure() }
        var total: Int64 = 0
        let input = FileHandle(fileDescriptor: source, closeOnDealloc: false)
        let output = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        while true {
            try Task.checkCancellation()
            let count = try autoreleasepool {
                let data = try input.read(upToCount: ChronosFiles.chunkSize) ?? Data()
                total += Int64(data.count)
                guard total <= info.st_size else { throw ChronosError.invalidBundle("copy source grew") }
                try output.write(contentsOf: data)
                return data.count
            }
            if count == 0 { break }
        }
        guard total == info.st_size else { throw ChronosError.invalidBundle("copy source shrank") }
    }
}
