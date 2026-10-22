import CryptoKit
import Foundation
import ModelLibraryFeature

public enum ArtifactTransferError: LocalizedError {
    case invalidDefinition
    case insufficientSpace
    case http(Int, String)
    case size(String)
    case checksum(String)
    case transport(String, any Error)

    public var errorDescription: String? {
        switch self {
        case .invalidDefinition: "The download definition is invalid."
        case .insufficientSpace: "There is not enough free space for the download."
        case .http(let status, let path): "HTTP \(status) for \(path)"
        case .size(let path): "Size mismatch for \(path)"
        case .checksum(let path): "Checksum mismatch for \(path)"
        case .transport(let path, _): "The connection was interrupted while downloading \(path)"
        }
    }
}

public actor ArtifactTransfer {
    public typealias FileLoader = @Sendable (URLRequest, URL) async throws -> URLResponse

    private static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieAcceptPolicy = .never
        configuration.httpShouldSetCookies = false
        configuration.urlCredentialStorage = nil
        configuration.urlCache = nil
        return URLSession(configuration: configuration)
    }()

    private let fileLoader: FileLoader
    private let availableCapacity: @Sendable (URL) throws -> Int64

    public init(
        fileLoader: @escaping FileLoader = ArtifactTransfer.load,
        availableCapacity: @escaping @Sendable (URL) throws -> Int64 = {
            try $0.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
                .volumeAvailableCapacityForImportantUsage ?? .max
        }
    ) {
        self.fileLoader = fileLoader
        self.availableCapacity = availableCapacity
    }

    public static func load(_ request: URLRequest, to destination: URL) async throws -> URLResponse {
        let (location, response) = try await session.download(for: request)
        defer { try? FileManager.default.removeItem(at: location) }
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: location, to: destination)
        return response
    }

    public func download(
        model: LibraryModel, artifact: ModelArtifact, to staging: URL,
        progress: @escaping @Sendable (ModelTransferProgress) -> Void
    ) async throws {
        do {
            try await transfer(model, artifact, staging, progress)
        } catch {
            if Task.isCancelled { throw CancellationError() }
            throw error
        }
    }

    private func transfer(
        _ model: LibraryModel, _ artifact: ModelArtifact, _ staging: URL,
        _ progress: @Sendable (ModelTransferProgress) -> Void
    ) async throws {
        guard let source = artifact.download, source.isValid, model.artifact(artifact.id) == artifact,
            !artifact.files.isEmpty, Set(artifact.files.map(\.path)).count == artifact.files.count,
            Set(artifact.files.map(\.path)).isDisjoint(with: artifact.files.map { $0.path + ".part" }),
            source.root.isEmpty || isRelative(source.root), artifact.files.allSatisfy({ isRelative($0.path) })
        else { throw ArtifactTransferError.invalidDefinition }
        let jobs = try artifact.files.map { ($0, try location(source, $0.path)) }
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        try prune(staging, keeping: artifact.files)
        var completed: Int64 = 0
        var pending: [(ModelArtifactFile, URL)] = []
        for (file, url) in jobs {
            let target = staging.appending(path: file.path)
            if try await Self.holds(target, file) {
                completed += file.bytes
            } else {
                try? FileManager.default.removeItem(at: target)
                pending.append((file, url))
            }
        }
        let total = artifact.files.reduce(0) { $0 + $1.bytes }
        guard try availableCapacity(staging) >= total - completed else {
            throw ArtifactTransferError.insufficientSpace
        }
        var reported: Int64 = -1
        func report() {
            guard completed > reported,
                let update = ModelTransferProgress(completedBytes: completed, totalBytes: total)
            else { return }
            reported = completed
            progress(update)
        }
        report()
        let loader = fileLoader
        try await withThrowingTaskGroup(of: Int64.self) { group in
            for (index, (file, url)) in pending.enumerated() {
                if index >= 3, let bytes = try await group.next() {
                    completed += bytes
                    report()
                }
                group.addTask { try await Self.fetch(file, from: url, into: staging, with: loader) }
            }
            for try await bytes in group {
                completed += bytes
                report()
            }
        }
    }

    @concurrent
    private static func holds(_ url: URL, _ file: ModelArtifactFile) async throws -> Bool {
        guard size(of: url) == file.bytes else { return false }
        return try digest(url) == file.sha256
    }

    @concurrent
    private static func fetch(
        _ file: ModelArtifactFile, from url: URL, into staging: URL, with loader: FileLoader
    ) async throws -> Int64 {
        let target = staging.appending(path: file.path)
        let part = target.appendingPathExtension("part")
        try FileManager.default.createDirectory(
            at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: part) }
        let response: URLResponse
        do {
            response = try await loader(URLRequest(url: url), part)
        } catch {
            throw ArtifactTransferError.transport(file.path, (error as? URLError).map { URLError($0.code) } ?? error)
        }
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else { throw ArtifactTransferError.http(status, file.path) }
        guard size(of: part) == file.bytes else { throw ArtifactTransferError.size(file.path) }
        guard try digest(part) == file.sha256 else { throw ArtifactTransferError.checksum(file.path) }
        try FileManager.default.moveItem(at: part, to: target)
        return file.bytes
    }
}

private let unreserved = CharacterSet(
    charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")

private func location(_ source: ModelDownload, _ path: String) throws -> URL {
    let parts = try (source.root.split(separator: "/") + path.split(separator: "/")).map { part in
        guard let encoded = part.addingPercentEncoding(withAllowedCharacters: unreserved) else {
            throw ArtifactTransferError.invalidDefinition
        }
        return encoded
    }
    let address = "https://huggingface.co/\(source.repository)/resolve/\(source.revision)/"
    guard let url = URL(string: address + parts.joined(separator: "/")) else {
        throw ArtifactTransferError.invalidDefinition
    }
    return url
}

private func isRelative(_ path: String) -> Bool {
    path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
}

private func prune(_ staging: URL, keeping files: [ModelArtifactFile]) throws {
    let paths = Set(files.map(\.path))
    let folders = Set(
        files.flatMap { file in
            let parts = file.path.split(separator: "/")
            return parts.indices.dropLast().map { parts[...$0].joined(separator: "/") }
        })
    var removed: [String] = []
    for path in (try? FileManager.default.subpathsOfDirectory(atPath: staging.path)) ?? [] {
        try Task.checkCancellation()
        guard !removed.contains(where: { path.hasPrefix($0 + "/") }) else { continue }
        let item = staging.appending(path: path)
        let type = (try? FileManager.default.attributesOfItem(atPath: item.path))?[.type] as? FileAttributeType
        if type == .typeRegular ? paths.contains(path) : type == .typeDirectory && folders.contains(path) { continue }
        try FileManager.default.removeItem(at: item)
        removed.append(path)
    }
}

private func size(of url: URL) -> Int64? {
    guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
        attributes[.type] as? FileAttributeType == .typeRegular
    else { return nil }
    return (attributes[.size] as? NSNumber)?.int64Value
}

private func digest(_ file: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: file)
    defer { try? handle.close() }
    var hash = SHA256()
    while try autoreleasepool(invoking: {
        guard let data = try handle.read(upToCount: 1_048_576), !data.isEmpty else { return false }
        try Task.checkCancellation()
        hash.update(data: data)
        return true
    }) {}
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}
