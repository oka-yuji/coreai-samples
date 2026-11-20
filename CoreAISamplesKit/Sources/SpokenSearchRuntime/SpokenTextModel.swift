import EmbeddingCore
import EmbeddingRuntime
import Foundation

protocol SpokenTextModel: Sendable {
    func load(directory: URL) async throws
    func encode(_ text: String, role: EmbeddingTextRole) async throws -> EmbeddingVector
    func unload() async throws
}

extension EmbeddingModel: SpokenTextModel {}
