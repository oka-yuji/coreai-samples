import EmbeddingCore
import Foundation

@testable import SpokenSearchRuntime

actor SpokenModelProbe: SpokenTextModel {
    var documents: [String] = []
    var queries: [String] = []
    var loads = 0
    var unloads = 0
    let characterLimit: Int

    init(characterLimit: Int = 200) {
        self.characterLimit = characterLimit
    }

    func load(directory: URL) async throws { loads += 1 }

    func encode(_ text: String, role: EmbeddingTextRole) async throws -> EmbeddingVector {
        if role == .document {
            guard text.count <= characterLimit else { throw EmbeddingError.tooManyTokens(text.count) }
            documents.append(text)
        } else {
            queries.append(text)
        }
        let isWeather = role == .query || text.lowercased().hasPrefix("rain") || text.lowercased().hasPrefix("wet")
        var values = Array(repeating: Float.zero, count: 768)
        values[isWeather ? 0 : 1] = 1
        return try EmbeddingVector(values)
    }

    func unload() async throws { unloads += 1 }
}
