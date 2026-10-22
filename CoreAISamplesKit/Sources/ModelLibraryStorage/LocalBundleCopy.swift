import CryptoKit
import Foundation
import ModelLibraryFeature

func copyLocalBundleFile(
    source: URL, file: ModelArtifactFile, target: URL, progress: (Int64) -> Void
) throws {
    let input = try localSourceHandle(source, path: file.path)
    defer { try? input.close() }
    let descriptor = open(target.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
    guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
    let output = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    defer { try? output.close() }
    var hash = SHA256()
    var copied: Int64 = 0
    while try autoreleasepool(invoking: {
        try Task.checkCancellation()
        guard let data = try input.read(upToCount: 1_048_576), !data.isEmpty else { return false }
        let count = Int64(data.count)
        guard count <= file.bytes - copied else { throw ManagedModelStoreError.sizeMismatch(file.path) }
        try output.write(contentsOf: data)
        hash.update(data: data)
        copied += count
        progress(count)
        return true
    }) {}
    guard copied == file.bytes else { throw ManagedModelStoreError.sizeMismatch(file.path) }
    let actual = hash.finalize().map { String(format: "%02x", $0) }.joined()
    guard actual == file.sha256 else { throw ManagedModelStoreError.checksumMismatch(file.path) }
}

private func localSourceHandle(_ root: URL, path: String) throws -> FileHandle {
    let parts = path.split(separator: "/", omittingEmptySubsequences: false)
    guard !parts.isEmpty, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
        let filename = parts.last
    else { throw ManagedModelStoreError.pathEscape }
    var directory = open(root.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
    guard directory >= 0 else { throw sourceOpenError(path) }
    defer { close(directory) }
    for part in parts.dropLast() {
        let next = openat(directory, String(part), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard next >= 0 else { throw sourceOpenError(path) }
        close(directory)
        directory = next
    }
    let descriptor = openat(directory, String(filename), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
    guard descriptor >= 0 else { throw sourceOpenError(path) }
    let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    var information = stat()
    guard fstat(descriptor, &information) == 0, information.st_mode & S_IFMT == S_IFREG else {
        try? handle.close()
        throw ManagedModelStoreError.missingFile(path)
    }
    return handle
}

private func sourceOpenError(_ path: String) -> any Error {
    switch errno {
    case ELOOP, ENOTDIR: ManagedModelStoreError.symbolicLink
    case ENOENT: ManagedModelStoreError.missingFile(path)
    default: POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
}
