import CoreAI
import Foundation
import GLiNERCore
import GLiNERRuntime
import ProductCore

public struct ProductLabel: Decodable, Sendable {
    public let field: ProductField
    public let label: String
    public let description: String

    public static func bundled() throws -> [ProductLabel] {
        guard let url = Bundle.module.url(forResource: "product-schema", withExtension: "json") else {
            throw ProductError.invalidSchema
        }
        let labels = try JSONDecoder().decode([ProductLabel].self, from: Data(contentsOf: url))
        guard labels.map(\.field) == ProductField.allCases,
            Set(labels.map(\.label)).count == labels.count,
            labels.allSatisfy({ !$0.label.isEmpty && !$0.description.isEmpty })
        else { throw ProductError.invalidSchema }
        return labels
    }
}

public actor ProductExtractor {
    private var extractor: GLiNERExtractor?
    private var loadedFolder: URL?
    private var busy = false

    public init() {}

    public func extract(source: ProductSource, folder: URL) async throws -> ProductMemo {
        guard !busy else { throw ProductError.busy }
        busy = true
        defer { busy = false }
        let access = folder.startAccessingSecurityScopedResource()
        defer { if access { folder.stopAccessingSecurityScopedResource() } }
        let labels = try ProductLabel.bundled()
        let schema = GLiNERSchema.entities(labels.map { EntityLabel(name: $0.label, description: $0.description) })
        if loadedFolder != folder || extractor == nil {
            extractor = nil
            loadedFolder = nil
            let asset = try GLiNERAsset(directory: folder)
            let nested = folder.appending(path: "tokenizer", directoryHint: .isDirectory)
            let tokenizerFolder =
                FileManager.default.fileExists(atPath: nested.appending(path: "tokenizer.json").path)
                ? nested : folder
            let tokenizer = try GLiNERTokenizer(directory: tokenizerFolder)
            try Task.checkCancellation()
            let model = try await GLiNERModel(
                asset: asset, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
            try Task.checkCancellation()
            extractor = GLiNERExtractor(preprocessor: GLiNERPreprocessor(tokenizer: tokenizer), model: model)
            loadedFolder = folder
        }
        guard let extractor else { throw ProductError.invalidSchema }
        var candidates: [ProductCandidate] = []
        var start = 0
        let count = source.text.unicodeScalars.count
        while start < count {
            try Task.checkCancellation()
            var limit = 600
            var chosen: ProductChunk?
            while let chunk = ProductChunks.next(in: source.text, from: start, limit: limit) {
                do {
                    _ = try extractor.preprocessor.prepare(text: chunk.text, schema: schema)
                    chosen = chunk
                    break
                } catch GLiNERPreprocessingError.inputTooLong {
                    guard limit > 32 else { throw ProductError.invalidSchema }
                    limit /= 2
                }
            }
            guard let chunk = chosen else { throw ProductError.invalidSchema }
            let extraction = try await extractor.extract(
                text: chunk.text, labels: labels.map { EntityLabel(name: $0.label, description: $0.description) })
            try Task.checkCancellation()
            for span in extraction.spans {
                guard labels.indices.contains(span.labelIndex) else { throw ProductError.invalidEvidence }
                candidates.append(
                    ProductCandidate(
                        field: labels[span.labelIndex].field,
                        range: (start + span.scalarRange.lowerBound)..<(start + span.scalarRange.upperBound),
                        text: span.text, probability: span.probability))
            }
            if chunk.range.upperBound == count { break }
            start = ProductChunks.followingStart(after: chunk)
        }
        return try ProductMemo(source: source, candidates: candidates)
    }

    public func release() throws {
        guard !busy else { throw ProductError.busy }
        extractor = nil
        loadedFolder = nil
    }
}
