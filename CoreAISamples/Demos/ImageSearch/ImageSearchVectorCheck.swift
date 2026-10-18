import EmbeddingCore
import EmbeddingRuntime
import Foundation

@MainActor
enum ImageSearchVectorCheck {
    private struct Reference: Decodable {
        let id: String
        let text: String?
        let role: EmbeddingTextRole?
        let image: String?
        let reference: [Float]
    }

    private struct Result: Encodable {
        let id: String
        let vector: [Float]
        let cosine: Double
        let maxDifference: Float
        let passed: Bool
    }

    static func run(folder: URL, output: URL) async throws {
        guard let file = Bundle.main.url(forResource: "image-search-reference", withExtension: "json") else {
            throw EmbeddingError.invalidBundle("Missing vector fixtures")
        }
        let references = try JSONDecoder().decode([Reference].self, from: Data(contentsOf: file))
        let bundle = try EmbeddingBundle(directory: folder)
        let quantized: [String: [Float]]
        if bundle.precision == "int4-int8-ptq-s128-v70" {
            quantized = try JSONDecoder().decode(
                [String: [Float]].self, from: Data(contentsOf: folder.appending(path: "quantized-reference.json")))
            guard quantized.count == references.count, Set(quantized.keys) == Set(references.map(\.id)) else {
                throw EmbeddingError.invalidBundle("Quantized vector fixtures differ")
            }
        } else {
            quantized = [:]
        }
        let model = EmbeddingModel()
        try await model.load(directory: folder)
        do {
            var results: [Result] = []
            for reference in references {
                let vector: EmbeddingVector
                if let name = reference.image {
                    guard let image = Bundle.main.url(forResource: name, withExtension: "png") else {
                        throw EmbeddingError.invalidImage
                    }
                    vector = try await model.encode(imageURL: image)
                } else {
                    vector = try await model.encode(reference.text ?? "", role: reference.role ?? .query)
                }
                let original = try EmbeddingVector(quantized[reference.id] ?? reference.reference)
                let cosine = vector.similarity(to: original)
                let difference = zip(vector.values, original.values).map { abs($0 - $1) }.max() ?? 1
                let passed = cosine >= 0.999 && difference <= 0.01
                results.append(
                    Result(
                        id: reference.id, vector: vector.values, cosine: cosine, maxDifference: difference,
                        passed: passed))
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(results).write(to: output, options: .atomic)
                print("VECTOR \(reference.id) cosine=\(cosine) maxDifference=\(difference) passed=\(passed)")
            }
            guard results.count == 20, results.allSatisfy(\.passed) else { throw EmbeddingError.invalidVector }
            try await model.unload()
        } catch {
            try? await model.unload()
            throw error
        }
    }
}
