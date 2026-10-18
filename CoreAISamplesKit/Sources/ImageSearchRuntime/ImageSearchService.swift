import EmbeddingCore
import EmbeddingRuntime
import Foundation
import ImageIO
import ImageSearchFeature
import UniformTypeIdentifiers

@MainActor
public protocol ImageSearchService {
    func load(_ folder: URL) async throws
    func index(_ urls: [URL], progress: (Int) -> Void) async throws -> [SearchPhoto]
    func search(_ query: String) async throws -> [ImageSearchMatch]
    func waitForIdle() async
    func close() async throws
}

@MainActor
public final class LiveImageSearchService: ImageSearchService {
    private let model = EmbeddingModel()
    private var indexed: [(photo: SearchPhoto, vector: EmbeddingVector)] = []
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    public init() {}

    public func load(_ folder: URL) async throws {
        try begin()
        defer { finish() }
        indexed = []
        try await model.load(directory: folder)
    }

    public func index(_ urls: [URL], progress: (Int) -> Void) async throws -> [SearchPhoto] {
        try begin()
        defer { finish() }
        guard !urls.isEmpty, urls.count <= 50 else { throw EmbeddingError.invalidImage }
        indexed = []
        if let missing = urls.first(where: { !Self.isReadable($0) }) {
            throw CocoaError(.fileReadNoSuchFile, userInfo: [NSURLErrorKey: missing])
        }
        var prepared: [(photo: SearchPhoto, vector: EmbeddingVector)] = []
        for url in urls {
            try Task.checkCancellation()
            let vector = try await model.encode(imageURL: url)
            let photo = SearchPhoto(url: url, thumbnail: try Self.thumbnail(url))
            try Task.checkCancellation()
            prepared.append((photo, vector))
            progress(prepared.count)
        }
        indexed = prepared
        return prepared.map(\.photo)
    }

    public func search(_ query: String) async throws -> [ImageSearchMatch] {
        try begin()
        defer { finish() }
        guard !indexed.isEmpty else { throw EmbeddingError.invalidImage }
        let vector = try await model.encode(query)
        try Task.checkCancellation()
        return indexed.map { ImageSearchMatch(photo: $0.photo, similarity: vector.similarity(to: $0.vector)) }
            .sorted { $0.similarity == $1.similarity ? $0.id.path < $1.id.path : $0.similarity > $1.similarity }
    }

    public func waitForIdle() async {
        while busy { await withCheckedContinuation { waiting.append($0) } }
    }

    public func close() async throws {
        await waitForIdle()
        busy = true
        defer { finish() }
        indexed = []
        try await model.unload()
    }

    private func begin() throws {
        try Task.checkCancellation()
        guard !busy else { throw EmbeddingError.busy }
        busy = true
    }

    private func finish() {
        busy = false
        let continuations = waiting
        waiting = []
        for continuation in continuations { continuation.resume() }
    }

    private static func isReadable(_ url: URL) -> Bool {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        return FileManager.default.isReadableFile(atPath: url.path(percentEncoded: false))
    }

    private static func thumbnail(_ url: URL) throws -> Data {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateThumbnailAtIndex(
                source, 0,
                [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceCreateThumbnailWithTransform: true,
                    kCGImageSourceThumbnailMaxPixelSize: 640,
                ] as CFDictionary)
        else { throw EmbeddingError.invalidImage }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil)
        else {
            throw EmbeddingError.invalidImage
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw EmbeddingError.invalidImage }
        return data as Data
    }
}
