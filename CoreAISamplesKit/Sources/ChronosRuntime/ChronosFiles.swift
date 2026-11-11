import ChronosCore
import CryptoKit
import Darwin
import Foundation

enum ChronosFiles {
    static let chunkSize = 1_048_576
    static let manifestLimit = 65_536
    static let referenceLimit = 2_097_152
    static let inputLimit = 262_144

    static func validateRelativePath(_ path: String) throws {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !parts.isEmpty, !path.contains("\\"), !path.contains(":"), !path.contains("\0"),
            parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." })
        else { throw ChronosError.invalidBundle("relative path") }
    }

    static func openDirectory(_ url: URL) throws -> Int32 {
        guard url.isFileURL, url.path.hasPrefix("/"), !url.path.contains("\0") else {
            throw ChronosError.invalidBundle("directory URL")
        }
        guard url.path.split(separator: "/").allSatisfy({ $0 != "." && $0 != ".." }) else {
            throw ChronosError.invalidBundle("directory alias")
        }
        let descriptor = Darwin.open(url.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw failure() }
        return descriptor
    }

    static func openFile(_ path: String, in directory: Int32) throws -> Int32 {
        try validateRelativePath(path)
        let parts = path.split(separator: "/").map(String.init)
        var parent = dup(directory)
        guard parent >= 0 else { throw failure() }
        defer { Darwin.close(parent) }
        for part in parts.dropLast() {
            let next = openat(parent, part, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard next >= 0 else { throw failure() }
            Darwin.close(parent)
            parent = next
        }
        guard let name = parts.last else { throw ChronosError.invalidBundle("file name") }
        let descriptor = openat(parent, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard descriptor >= 0 else { throw failure() }
        do {
            let info = try status(descriptor)
            guard info.st_mode & S_IFMT == S_IFREG, info.st_nlink == 1 else {
                throw ChronosError.invalidBundle("regular unaliased file required")
            }
            return descriptor
        } catch {
            Darwin.close(descriptor)
            throw error
        }
    }

    static func names(in descriptor: Int32) throws -> Set<String> {
        let copy = dup(descriptor)
        guard copy >= 0 else { throw failure() }
        guard let directory = fdopendir(copy) else {
            Darwin.close(copy)
            throw failure()
        }
        defer { closedir(directory) }
        rewinddir(directory)
        var result: Set<String> = []
        while true {
            errno = 0
            guard let entry = readdir(directory) else {
                guard errno == 0 else { throw failure() }
                return result
            }
            let name = withUnsafePointer(to: &entry.pointee.d_name) {
                $0.withMemoryRebound(to: CChar.self, capacity: Int(MAXNAMLEN) + 1) { String(cString: $0) }
            }
            if name != ".", name != ".." { result.insert(name) }
        }
    }

    static func read(_ path: String, in directory: Int32, limit: Int) throws -> Data {
        let descriptor = try openFile(path, in: directory)
        defer { Darwin.close(descriptor) }
        let before = try status(descriptor)
        guard before.st_size >= 0, before.st_size <= limit else {
            throw ChronosError.invalidBundle("JSON size limit")
        }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        var result = Data()
        while true {
            try Task.checkCancellation()
            let count = try autoreleasepool {
                let chunk = try handle.read(upToCount: min(chunkSize, limit + 1 - result.count)) ?? Data()
                result.append(chunk)
                return chunk.count
            }
            guard result.count <= limit else { throw ChronosError.invalidBundle("JSON size limit") }
            if count == 0 { break }
        }
        try unchanged(before, descriptor: descriptor)
        guard result.count == before.st_size else { throw ChronosError.invalidBundle("file size changed") }
        return result
    }

    static func read(_ url: URL, limit: Int) throws -> Data {
        let directory = try openDirectory(url.deletingLastPathComponent())
        defer { Darwin.close(directory) }
        return try read(url.lastPathComponent, in: directory, limit: limit)
    }

    static func hash(_ path: String, in directory: Int32, expected: ChronosFileIdentity) throws {
        let descriptor = try openFile(path, in: directory)
        defer { Darwin.close(descriptor) }
        let before = try status(descriptor)
        guard before.st_size == expected.bytes else { throw ChronosError.invalidBundle("file size: \(path)") }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        var hasher = SHA256()
        var count: Int64 = 0
        while true {
            try Task.checkCancellation()
            let length = try autoreleasepool {
                let chunk = try handle.read(upToCount: chunkSize) ?? Data()
                hasher.update(data: chunk)
                return chunk.count
            }
            count += Int64(length)
            guard count <= expected.bytes else { throw ChronosError.invalidBundle("file grew: \(path)") }
            if length == 0 { break }
        }
        try unchanged(before, descriptor: descriptor)
        guard count == expected.bytes, hex(hasher.finalize()) == expected.sha256 else {
            throw ChronosError.invalidBundle("file hash: \(path)")
        }
        try Task.checkCancellation()
    }

    static func identity(_ data: Data) -> ChronosFileIdentity {
        ChronosFileIdentity(bytes: Int64(data.count), sha256: hex(SHA256.hash(data: data)))
    }

    private static func hex(_ digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func status(_ descriptor: Int32) throws -> stat {
        var info = stat()
        guard fstat(descriptor, &info) == 0 else { throw failure() }
        return info
    }

    private static func unchanged(_ before: stat, descriptor: Int32) throws {
        let after = try status(descriptor)
        guard before.st_size == after.st_size, before.st_nlink == after.st_nlink,
            before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
            before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
            before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec,
            before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec
        else { throw ChronosError.invalidBundle("file changed during read") }
    }

    static func failure() -> POSIXError {
        POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
}
