import CryptoKit
import Darwin
import Foundation
import ReceiptCore

final class ReceiptExportTransaction {
    private struct Identity: Equatable {
        let device: dev_t
        let inode: ino_t

        init(_ value: stat) {
            device = value.st_dev
            inode = value.st_ino
        }
    }

    private struct Directory {
        let descriptor: Int32
        let parent: Int32
        let name: String
        let identity: Identity
    }

    private struct File {
        let bytes: Int
        let digest: SHA256.Digest
        let descriptor: Int32
        let parent: Int32
        let name: String
        let identity: Identity
    }

    private let parent: Int32
    private let parentIdentity: Identity
    private let directory: URL
    private let staging: Directory
    private let destinationName: String
    private var directories: [String: Directory] = [:]
    private var files: [File] = []
    private var published = false

    init(directory: URL, identifier: UUID) throws {
        guard directory.isFileURL, !directory.path.contains("\0") else {
            throw ReceiptRuntimeError.invalidDestination
        }
        let parent = open(directory.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard parent >= 0 else { throw ReceiptRuntimeError.invalidDestination }
        do {
            let identity = try Self.identity(parent)
            let name = "ReceiptDesk-" + identifier.uuidString
            var existing = stat()
            guard fstatat(parent, name, &existing, AT_SYMLINK_NOFOLLOW) != 0, errno == ENOENT else {
                throw ReceiptRuntimeError.destinationExists
            }
            let stagingName = "." + name + ".partial"
            guard mkdirat(parent, stagingName, 0o700) == 0 else {
                throw errno == EEXIST ? ReceiptRuntimeError.destinationExists : ReceiptRuntimeError.invalidDestination
            }
            let descriptor = openat(parent, stagingName, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard descriptor >= 0 else { throw ReceiptRuntimeError.cleanupFailed }
            let stagingIdentity: Identity
            do { stagingIdentity = try Self.identity(descriptor) } catch {
                _ = Darwin.close(descriptor)
                throw error
            }
            self.parent = parent
            parentIdentity = identity
            self.directory = directory
            destinationName = name
            staging = Directory(
                descriptor: descriptor, parent: parent, name: stagingName, identity: stagingIdentity)
        } catch {
            _ = Darwin.close(parent)
            throw error
        }
    }

    deinit {
        for file in files { _ = Darwin.close(file.descriptor) }
        for value in directories.values { _ = Darwin.close(value.descriptor) }
        _ = Darwin.close(staging.descriptor)
        _ = Darwin.close(parent)
    }

    func write(_ data: Data, path: String) throws {
        try validateStaging()
        let components = path.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard !components.isEmpty, components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
            !path.contains("\0"), !path.contains("\\"), let name = components.last
        else { throw ReceiptRuntimeError.invalidDestination }
        var descriptor = staging.descriptor
        var key = ""
        for component in components.dropLast() {
            key += "/" + component
            if let child = directories[key] {
                guard Self.matches(child) else { throw ReceiptRuntimeError.invalidDestination }
                descriptor = child.descriptor
            } else {
                guard mkdirat(descriptor, component, 0o700) == 0 else {
                    throw ReceiptRuntimeError.invalidDestination
                }
                let child = openat(descriptor, component, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                guard child >= 0 else { throw ReceiptRuntimeError.cleanupFailed }
                let identity: Identity
                do { identity = try Self.identity(child) } catch {
                    _ = Darwin.close(child)
                    throw error
                }
                directories[key] = Directory(
                    descriptor: child, parent: descriptor, name: component, identity: identity)
                descriptor = child
            }
        }
        let output = openat(descriptor, name, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard output >= 0 else { throw ReceiptRuntimeError.invalidDestination }
        let identity: Identity
        do { identity = try Self.identity(output) } catch {
            _ = Darwin.close(output)
            throw error
        }
        files.append(
            File(
                bytes: data.count, digest: SHA256.hash(data: data), descriptor: output, parent: descriptor, name: name,
                identity: identity))
        try data.withUnsafeBytes { bytes in
            var offset = 0
            while offset < bytes.count {
                try Task.checkCancellation()
                let count = Darwin.write(
                    output, bytes.baseAddress?.advanced(by: offset), min(65_536, bytes.count - offset))
                if count < 0, errno == EINTR { continue }
                guard count > 0 else { throw ReceiptRuntimeError.invalidDestination }
                offset += count
            }
        }
    }

    func writeLedger(_ ledger: ReceiptLedger) throws {
        struct Envelope: Encodable {
            let format: String
            let total: ReceiptAmount
            let receipts: [ReceiptLedgerRow]
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(
            Envelope(format: "receipt-desk/1", total: ReceiptAmount(yen: ledger.totalYen), receipts: ledger.rows))
        try write(data, path: "ledger.json")
    }

    func publish() throws -> URL {
        try validateStaging()
        try verifyInventory()
        for file in files { try Self.verifyContents(file) }
        try validateStaging()
        try verifyInventory()
        guard renameatx_np(parent, staging.name, parent, destinationName, UInt32(RENAME_EXCL)) == 0 else {
            throw errno == EEXIST ? ReceiptRuntimeError.destinationExists : ReceiptRuntimeError.invalidDestination
        }
        published = true
        guard Self.matches(parent: parent, name: destinationName, identity: staging.identity) else {
            throw ReceiptRuntimeError.invalidDestination
        }
        return directory.appending(path: destinationName, directoryHint: .isDirectory)
    }

    func cleanUp() throws {
        guard !published else { return }
        var failed = false
        for file in files.reversed() {
            if Self.matches(parent: file.parent, name: file.name, identity: file.identity) {
                if unlinkat(file.parent, file.name, 0) != 0 { failed = true }
            } else {
                failed = true
            }
        }
        for key in directories.keys.sorted(by: { $0.count > $1.count }) {
            guard let value = directories[key], Self.matches(value) else {
                failed = true
                continue
            }
            if unlinkat(value.parent, value.name, AT_REMOVEDIR) != 0 { failed = true }
        }
        if Self.matches(staging) {
            if unlinkat(parent, staging.name, AT_REMOVEDIR) != 0 { failed = true }
        } else {
            failed = true
        }
        if failed { throw ReceiptRuntimeError.cleanupFailed }
    }

    private func verifyInventory() throws {
        let folders = [staging] + directories.values
        guard folders.allSatisfy(Self.matches),
            files.allSatisfy({ Self.matches(parent: $0.parent, name: $0.name, identity: $0.identity) })
        else { throw ReceiptRuntimeError.invalidDestination }
        for folder in folders {
            let expected = Set(
                directories.values.filter { $0.parent == folder.descriptor }.map(\.name)
                    + files.filter { $0.parent == folder.descriptor }.map(\.name))
            guard try Self.names(in: folder.descriptor) == expected else {
                throw ReceiptRuntimeError.invalidDestination
            }
        }
    }

    private static func names(in descriptor: Int32) throws -> Set<String> {
        let copied = dup(descriptor)
        guard copied >= 0 else { throw ReceiptRuntimeError.invalidDestination }
        guard let stream = fdopendir(copied) else {
            _ = Darwin.close(copied)
            throw ReceiptRuntimeError.invalidDestination
        }
        defer { closedir(stream) }
        rewinddir(stream)
        var names: Set<String> = []
        while true {
            errno = 0
            guard let entry = readdir(stream) else {
                guard errno == 0 else { throw ReceiptRuntimeError.invalidDestination }
                return names
            }
            let name = withUnsafeBytes(of: entry.pointee.d_name) {
                String(decoding: $0.prefix(Int(entry.pointee.d_namlen)), as: UTF8.self)
            }
            if name != ".", name != ".." { names.insert(name) }
        }
    }

    private static func verifyContents(_ file: File) throws {
        var before = stat()
        guard fstat(file.descriptor, &before) == 0, before.st_size == file.bytes else {
            throw ReceiptRuntimeError.sourceMismatch
        }
        var hash = SHA256()
        var buffer = [UInt8](repeating: 0, count: 65_536)
        var offset = 0
        while offset < file.bytes {
            try Task.checkCancellation()
            let count = buffer.withUnsafeMutableBytes {
                pread(file.descriptor, $0.baseAddress, min($0.count, file.bytes - offset), off_t(offset))
            }
            if count < 0, errno == EINTR { continue }
            guard count > 0 else { throw ReceiptRuntimeError.sourceMismatch }
            buffer.withUnsafeBytes { hash.update(bufferPointer: UnsafeRawBufferPointer(rebasing: $0.prefix(count))) }
            offset += count
        }
        var after = stat()
        guard hash.finalize() == file.digest, fstat(file.descriptor, &after) == 0,
            before.st_size == after.st_size,
            before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
            before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
            before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec,
            before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec,
            matches(parent: file.parent, name: file.name, identity: file.identity)
        else { throw ReceiptRuntimeError.sourceMismatch }
    }

    private func validateStaging() throws {
        let current = open(directory.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard current >= 0 else { throw ReceiptRuntimeError.invalidDestination }
        defer { _ = Darwin.close(current) }
        guard try Self.identity(current) == parentIdentity, Self.matches(staging), !published else {
            throw ReceiptRuntimeError.invalidDestination
        }
    }

    private static func identity(_ descriptor: Int32) throws -> Identity {
        var value = stat()
        guard fstat(descriptor, &value) == 0 else { throw ReceiptRuntimeError.invalidDestination }
        return Identity(value)
    }

    private static func matches(_ directory: Directory) -> Bool {
        matches(parent: directory.parent, name: directory.name, identity: directory.identity)
    }

    private static func matches(parent: Int32, name: String, identity: Identity) -> Bool {
        var value = stat()
        return fstatat(parent, name, &value, AT_SYMLINK_NOFOLLOW) == 0 && Identity(value) == identity
    }
}
