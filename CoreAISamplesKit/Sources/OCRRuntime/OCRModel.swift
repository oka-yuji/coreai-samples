import Foundation
import OCRCore

typealias OCRRotary = (cos: [Float], sin: [Float])

protocol OCRModel: Sendable {
    associatedtype Cache: Sendable

    var windowLimit: Int { get }
    var tokenLimit: Int { get }
    func encode(_ text: String) -> [Int32]
    func decode(_ tokens: [Int32], skipSpecialTokens: Bool) -> String
    func makeCache(capacity: Int) throws -> Cache
    @concurrent func vision(_ input: OCRVisionInput) async throws -> [Float]
    @concurrent func embed(_ ids: [Int32]) async throws -> [Float]
    @concurrent func prefill(_ embeddings: [Float], rotary: OCRRotary, cache: inout Cache) async throws -> [Float]
    @concurrent func step(_ token: Int32, position: Int, rotary: OCRRotary, cache: inout Cache) async throws
        -> [Float]
}
