import CryptoKit
import Darwin
import Foundation

struct ReceiptSourceCopy: Sendable {
    let name: String
    let fileExtension: String
    let bytes: Data
    let sha256: String
    let text: String
}

enum ReceiptFileReader {
    static let maximumBytes = 20 * 1024 * 1024

    static func read(_ url: URL, beforeFinalStat: () throws -> Void = {}) throws -> ReceiptSourceCopy {
        guard url.isFileURL, !url.path.contains("\0") else { throw ReceiptRuntimeError.invalidSource }
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        let ext = url.pathExtension.lowercased()
        guard ["png", "jpg", "jpeg", "heic", "heif", "tif", "tiff", "pdf"].contains(ext) else {
            throw ReceiptRuntimeError.unsupportedFile
        }
        let descriptor = open(url.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard descriptor >= 0 else { throw ReceiptRuntimeError.invalidSource }
        defer { _ = Darwin.close(descriptor) }
        var before = stat()
        guard fstat(descriptor, &before) == 0, before.st_mode & S_IFMT == S_IFREG, before.st_size >= 0 else {
            throw ReceiptRuntimeError.invalidSource
        }
        guard before.st_size <= maximumBytes else { throw ReceiptRuntimeError.tooLarge }
        var bytes = Data()
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)
        while true {
            try Task.checkCancellation()
            let count = buffer.withUnsafeMutableBytes { Darwin.read(descriptor, $0.baseAddress, $0.count) }
            if count == 0 { break }
            if count < 0 {
                if errno == EINTR { continue }
                throw ReceiptRuntimeError.invalidSource
            }
            guard bytes.count + count <= maximumBytes else { throw ReceiptRuntimeError.tooLarge }
            bytes.append(contentsOf: buffer.prefix(count))
        }
        var after = stat()
        try beforeFinalStat()
        guard fstat(descriptor, &after) == 0, before.st_size == after.st_size,
            before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
            before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
            before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec,
            before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec,
            Int64(bytes.count) == after.st_size
        else { throw ReceiptRuntimeError.sourceChanged }
        guard !bytes.isEmpty else { throw ReceiptRuntimeError.invalidSource }
        return ReceiptSourceCopy(
            name: url.lastPathComponent, fileExtension: ext, bytes: bytes, sha256: digest(bytes), text: "")
    }

    static func digest(_ bytes: Data) -> String {
        SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }
}
