import CoreGraphics
import EmbeddingCore
import EmbeddingRuntime
import Foundation
import ObjectFinderCore

public enum ObjectFinderError: Error, Equatable, Sendable {
    case busy, notPrepared, emptyQuery, invalidImage
}

struct RegionEmbeddingOperations: Sendable {
    var load: @Sendable (URL) async throws -> Void
    var text: @Sendable (String) async throws -> EmbeddingVector
    var image: @Sendable (CGImage) async throws -> EmbeddingVector
    var close: @Sendable () async throws -> Void
}

public actor ObjectRegionFinder {
    private let embedding: RegionEmbeddingOperations
    private let propose: @Sendable (CGImage) throws -> [ObjectRegion]
    private var query: EmbeddingVector?
    private var busy = false

    public init() {
        let model = EmbeddingModel()
        embedding = RegionEmbeddingOperations(
            load: { try await model.load(directory: $0) },
            text: { try await model.encode($0) },
            image: { try await model.encode(image: $0) },
            close: { try await model.unload() })
        propose = ObjectRegionProposer.regions
    }

    init(embedding: RegionEmbeddingOperations, propose: @escaping @Sendable (CGImage) throws -> [ObjectRegion]) {
        self.embedding = embedding
        self.propose = propose
    }

    public func prepare(folder: URL, query text: String) async throws {
        try begin()
        defer { busy = false }
        query = nil
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw ObjectFinderError.emptyQuery }
        do {
            try await embedding.load(folder)
            try Task.checkCancellation()
            let vector = try await embedding.text(trimmed)
            try Task.checkCancellation()
            query = vector
        } catch {
            try await embedding.close()
            throw error
        }
    }

    public func candidates(in image: CGImage) async throws -> [ObjectCandidate] {
        try begin()
        defer { busy = false }
        guard let query else { throw ObjectFinderError.notPrepared }
        guard image.width > 0, image.height > 0, image.width <= 4096, image.height <= 4096 else {
            throw ObjectFinderError.invalidImage
        }
        let regions = try propose(image)
        var matches: [ObjectCandidate] = []
        for (id, region) in regions.prefix(3).enumerated() {
            try Task.checkCancellation()
            guard let crop = ObjectRegionProposer.crop(image, to: region) else { continue }
            let vector = try await embedding.image(crop)
            try Task.checkCancellation()
            matches.append(ObjectCandidate(id: id, region: region, similarity: query.similarity(to: vector)))
        }
        return matches
    }

    public func close() async throws {
        guard !busy else { throw ObjectFinderError.busy }
        busy = true
        defer { busy = false }
        query = nil
        try await embedding.close()
    }

    private func begin() throws {
        guard !busy else { throw ObjectFinderError.busy }
        try Task.checkCancellation()
        busy = true
    }
}
