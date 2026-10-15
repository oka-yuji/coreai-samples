import EmbeddingCore
import EmbeddingRuntime
import Foundation

enum SearchCheck {
    struct Plan: Decodable {
        struct ImageInput: Decodable {
            let id: String
            let path: String
        }
        struct Query: Decodable {
            let id: String
            let text: String
        }
        let images: [ImageInput]
        let queries: [Query]
    }

    struct Match: Encodable {
        let id: String
        let similarity: Double
    }

    struct Result: Encodable {
        let id: String
        let matches: [Match]
    }

    static func run(_ arguments: [String]) async throws {
        guard arguments.count == 5 else {
            throw EmbeddingError.invalidBundle("Usage: embedding-verify --search BUNDLE PLAN OUTPUT")
        }
        let output = URL(filePath: arguments[4])
        guard !FileManager.default.fileExists(atPath: output.path) else {
            throw EmbeddingError.invalidBundle("Use a new search output file")
        }
        let plan = try JSONDecoder().decode(Plan.self, from: Data(contentsOf: URL(filePath: arguments[3])))
        guard !plan.images.isEmpty, !plan.queries.isEmpty,
            Set(plan.images.map(\.id)).count == plan.images.count,
            Set(plan.queries.map(\.id)).count == plan.queries.count
        else { throw EmbeddingError.invalidBundle("Search inputs must have unique identifiers") }
        let model = EmbeddingModel()
        try await model.load(directory: URL(filePath: arguments[2]))
        var images: [(String, EmbeddingVector)] = []
        for image in plan.images {
            images.append((image.id, try await model.encode(imageURL: URL(filePath: image.path))))
        }
        var results: [Result] = []
        for query in plan.queries {
            let vector = try await model.encode(query.text)
            var matches: [Match] = []
            for (id, imageVector) in images {
                matches.append(Match(id: id, similarity: vector.similarity(to: imageVector)))
            }
            matches.sort { left, right in
                if left.similarity == right.similarity { return left.id < right.id }
                return left.similarity > right.similarity
            }
            results.append(Result(id: query.id, matches: matches))
        }
        try await model.unload()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(results).write(to: output, options: .atomic)
    }
}
