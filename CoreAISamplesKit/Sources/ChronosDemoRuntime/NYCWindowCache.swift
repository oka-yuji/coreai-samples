import ChronosFeature
import Darwin
import Foundation

struct NYCWindowCache {
    static let fileLimit = 48 * 1024 * 1024
    static let totalLimit = 64 * 1024 * 1024
    let directory: URL?
    let capacity: Int

    func read(_ key: String) throws -> APIWindow? {
        guard let directory else { return nil }
        let folder = open(directory.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW)
        guard folder >= 0 else { return nil }
        defer { close(folder) }
        let descriptor = openat(folder, key + ".json", O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
        guard descriptor >= 0 else { return nil }
        let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? file.close() }
        var metadata = stat()
        guard fstat(descriptor, &metadata) == 0, metadata.st_mode & S_IFMT == S_IFREG,
            metadata.st_size > 0, metadata.st_size <= Self.fileLimit
        else { throw ChronosDemoFailure.invalidData }
        let data = try file.read(upToCount: Self.fileLimit + 1) ?? Data()
        guard data.count == metadata.st_size else { throw ChronosDemoFailure.invalidData }
        let envelope = try NYCAPIContract.decode(Envelope.self, data)
        guard NYCAPIContract.hash(envelope.payload) == envelope.sha256 else { throw ChronosDemoFailure.invalidData }
        let value = try NYCAPIContract.decode(APIWindow.self, envelope.payload)
        try NYCWindowValidation.validate(value)
        return value
    }

    func save(_ value: APIWindow, key: String) throws {
        guard let directory else { return }
        let payload = try NYCAPIContract.encode(value)
        let data = try NYCAPIContract.encode(Envelope(payload: payload, sha256: NYCAPIContract.hash(payload)))
        guard data.count <= Self.fileLimit else { throw ChronosDemoFailure.responseTooLarge }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let folder = open(directory.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW)
        guard folder >= 0 else { throw ChronosDemoFailure.persistence }
        defer { close(folder) }
        let temporary = UUID().uuidString + ".tmp"
        let descriptor = openat(folder, temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0o600)
        guard descriptor >= 0 else { throw ChronosDemoFailure.persistence }
        let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer {
            try? file.close()
            unlinkat(folder, temporary, 0)
        }
        try file.write(contentsOf: data)
        guard renameat(folder, temporary, folder, key + ".json") == 0 else { throw ChronosDemoFailure.persistence }
        try prune(folder: folder, directory: directory, keeping: key + ".json", bytes: data.count)
    }

    private func prune(folder: Int32, directory: URL, keeping: String, bytes: Int) throws {
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        let candidates = names.compactMap { name -> (String, stat)? in
            guard name != keeping, name.range(of: "^[0-9a-f]{64}\\.json$", options: .regularExpression) != nil else {
                return nil
            }
            var info = stat()
            guard fstatat(folder, name, &info, AT_SYMLINK_NOFOLLOW) == 0, info.st_mode & S_IFMT == S_IFREG else {
                return nil
            }
            return (name, info)
        }.sorted { $0.1.st_mtimespec.tv_sec > $1.1.st_mtimespec.tv_sec }
        var count = 1
        var total = Int64(bytes)
        for (name, info) in candidates {
            if count < capacity, total + info.st_size <= Self.totalLimit {
                count += 1
                total += info.st_size
            } else if unlinkat(folder, name, 0) != 0 {
                throw ChronosDemoFailure.persistence
            }
        }
    }

    private struct Envelope: Codable {
        let payload: Data
        let sha256: String
    }
}
