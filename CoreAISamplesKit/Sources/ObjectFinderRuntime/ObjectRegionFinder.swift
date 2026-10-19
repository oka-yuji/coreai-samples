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
    private let instances: @Sendable (CGImage) throws -> [ObjectRegion]
    private var query: EmbeddingVector?
    private var preparedFolder: URL?
    private var busy = false

    public init() {
        let model = EmbeddingModel()
        embedding = RegionEmbeddingOperations(
            load: { try await model.load(directory: $0) },
            text: { try await model.encode($0) },
            image: { try await model.encode(image: $0) },
            close: { try await model.unload() })
        propose = ObjectRegionProposer.regions
        instances = ObjectRegionProposer.instances
    }

    init(
        embedding: RegionEmbeddingOperations, propose: @escaping @Sendable (CGImage) throws -> [ObjectRegion],
        instances: @escaping @Sendable (CGImage) throws -> [ObjectRegion] = { _ in [] }
    ) {
        self.embedding = embedding
        self.propose = propose
        self.instances = instances
    }

    public func prepare(folder: URL, query text: String) async throws {
        try begin()
        defer { busy = false }
        query = nil
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        do {
            guard !trimmed.isEmpty else { throw ObjectFinderError.emptyQuery }
            if preparedFolder != folder {
                preparedFolder = nil
                try await embedding.load(folder)
                try Task.checkCancellation()
                preparedFolder = folder
            }
            let vector = try await embedding.text(trimmed)
            try Task.checkCancellation()
            query = vector
        } catch {
            preparedFolder = nil
            try await withTaskCancellationShield { try await embedding.close() }
            throw error
        }
    }

    public func candidates(in image: CGImage, mode: ObjectSearchMode = .standard) async throws -> [ObjectCandidate] {
        try begin()
        defer { busy = false }
        guard let query else { throw ObjectFinderError.notPrepared }
        guard image.width > 0, image.height > 0, image.width <= 4096, image.height <= 4096 else {
            throw ObjectFinderError.invalidImage
        }
        let regions =
            switch mode {
            case .standard: Array(try propose(image).prefix(3))
            case .twoStep: try await twoStepRegions(in: image, query: query)
            }
        var matches: [ObjectCandidate] = []
        for (id, region) in regions.enumerated() {
            guard let similarity = try await similarity(of: region, in: image, to: query) else { continue }
            matches.append(ObjectCandidate(id: id, region: region, similarity: similarity))
        }
        return matches
    }

    private func twoStepRegions(in image: CGImage, query: EmbeddingVector) async throws -> [ObjectRegion] {
        var regions = Array(try propose(image).prefix(3)) + (try instances(image))
        var best: (window: ObjectRegion, similarity: Double)?
        for window in ObjectRegion.searchWindows {
            guard let similarity = try await similarity(of: window, in: image, to: query) else { continue }
            if similarity > best?.similarity ?? -.infinity { best = (window, similarity) }
        }
        if let window = best?.window, window != .frame, let crop = ObjectRegionProposer.crop(image, to: window) {
            let local = Array(try propose(crop).prefix(3)) + (try instances(crop))
            regions += local.prefix(3).map { $0.placed(in: window) }
        }
        var unique: [ObjectRegion] = []
        for region in regions where region.isValid && !unique.contains(where: { $0.overlap(with: region) > 0.9 }) {
            unique.append(region)
        }
        return unique
    }

    private func similarity(of region: ObjectRegion, in image: CGImage, to query: EmbeddingVector) async throws
        -> Double?
    {
        try Task.checkCancellation()
        guard let crop = ObjectRegionProposer.crop(image, to: region) else { return nil }
        let vector = try await embedding.image(crop)
        try Task.checkCancellation()
        return query.similarity(to: vector)
    }

    public func close() async throws {
        guard !busy else { throw ObjectFinderError.busy }
        busy = true
        defer { busy = false }
        query = nil
        preparedFolder = nil
        try await withTaskCancellationShield { try await embedding.close() }
    }

    private func begin() throws {
        guard !busy else { throw ObjectFinderError.busy }
        try Task.checkCancellation()
        busy = true
    }
}
