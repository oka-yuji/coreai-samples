import CTokenizers
import CryptoKit
import Foundation

public final class VoxCPMTokenizer {
    public static let tokenizerSHA256 = "f8984687e4a92a3503d521396d454b7d68e9fdaab2a0288eb3536c7c1aa4bc20"

    public let vocabulary: [Int32: String]
    private let handle: UnsafeMutableRawPointer

    public init(tokenizerJSON: URL) throws {
        let bytes = Array(try Data(contentsOf: tokenizerJSON))
        let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        guard digest == Self.tokenizerSHA256 else { throw VoxCPMTokenizerError.unexpectedSHA256 }
        vocabulary = try Self.vocabulary(Data(bytes))
        var error: UnsafeMutablePointer<CChar>?
        let value = bytes.withUnsafeBufferPointer { iro_tokenizer_create($0.baseAddress, $0.count, &error) }
        defer { iro_error_free(error) }
        guard let value else { throw VoxCPMTokenizerError.tokenizer(Self.message(error)) }
        handle = value
    }

    deinit { iro_tokenizer_destroy(handle) }

    public func tokenIDs(for text: String) throws -> [Int32] {
        let bytes = Array(text.utf8)
        var output = IroTokens()
        var error: UnsafeMutablePointer<CChar>?
        let status = bytes.withUnsafeBufferPointer {
            iro_tokenizer_encode(handle, $0.baseAddress, $0.count, &output, &error)
        }
        defer {
            iro_tokens_free(output)
            iro_error_free(error)
        }
        guard status == 0 else { throw VoxCPMTokenizerError.tokenizer(Self.message(error)) }
        guard output.len == 0 || output.data != nil else { throw VoxCPMTokenizerError.invalidTokenID }
        return try UnsafeBufferPointer(start: output.data, count: output.len).map {
            guard let id = Int32(exactly: $0), vocabulary[id] != nil else { throw VoxCPMTokenizerError.invalidTokenID }
            return id
        }
    }

    private static func vocabulary(_ data: Data) throws -> [Int32: String] {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? NSDictionary,
            let model = root["model"] as? NSDictionary, let tokens = model["vocab"] as? NSDictionary,
            let added = root["added_tokens"] as? NSArray
        else { throw VoxCPMTokenizerError.invalidVocabulary }
        var result: [Int32: String] = [:]
        for (key, value) in tokens {
            guard let token = key as? String, let id = value as? Int32 else {
                throw VoxCPMTokenizerError.invalidVocabulary
            }
            try insert(token, id, into: &result)
        }
        for entry in added {
            guard let entry = entry as? NSDictionary, let token = entry["content"] as? String,
                let id = entry["id"] as? Int32
            else { throw VoxCPMTokenizerError.invalidVocabulary }
            try insert(token, id, into: &result)
        }
        return result
    }

    private static func insert(_ token: String, _ id: Int32, into result: inout [Int32: String]) throws {
        if let existing = result[id], Array(existing.unicodeScalars) != Array(token.unicodeScalars) {
            throw VoxCPMTokenizerError.invalidVocabulary
        }
        guard id >= 0 else { throw VoxCPMTokenizerError.invalidVocabulary }
        result[id] = token
    }

    private static func message(_ pointer: UnsafeMutablePointer<CChar>?) -> String {
        pointer.map { String(cString: $0) } ?? "Tokenizer failed without an error message"
    }
}
