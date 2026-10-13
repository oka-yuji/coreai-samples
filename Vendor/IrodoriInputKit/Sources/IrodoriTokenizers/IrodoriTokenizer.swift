import CTokenizers
import CryptoKit
import Foundation
import IrodoriInputLogic

public final class IrodoriTokenizer {
    private let handle: UnsafeMutableRawPointer

    public init(tokenizerJSON: URL) throws {
        let bytes = Array(try Data(contentsOf: tokenizerJSON))
        let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        guard digest == "6a0734cf21c802169defaffe719bc2ef12bb9d0be37e54b61ed27aa89394723d" else {
            throw IrodoriInputError.tokenizer("Unexpected tokenizer SHA-256 for Irodori v4.1 Small MF")
        }
        var error: UnsafeMutablePointer<CChar>?
        let value = bytes.withUnsafeBufferPointer { iro_tokenizer_create($0.baseAddress, $0.count, &error) }
        defer { iro_error_free(error) }
        guard let value else { throw IrodoriInputError.tokenizer(Self.message(error)) }
        handle = value
    }

    deinit { iro_tokenizer_destroy(handle) }

    public func prepare(_ text: String) throws -> IrodoriTextInput {
        let normalized = try IrodoriText.normalize(text)
        guard !normalized.isEmpty else { throw IrodoriInputError.emptyText }
        let bytes = Array(normalized.utf8)
        var output = IroTokens()
        var error: UnsafeMutablePointer<CChar>?
        let status = bytes.withUnsafeBufferPointer {
            iro_tokenizer_encode(handle, $0.baseAddress, $0.count, &output, &error)
        }
        defer {
            iro_tokens_free(output)
            iro_error_free(error)
        }
        guard status == 0 else { throw IrodoriInputError.tokenizer(Self.message(error)) }
        guard output.len == 0 || output.data != nil else { throw IrodoriInputError.invalidTokenID }
        let raw = UnsafeBufferPointer(start: output.data, count: output.len)
        return try IrodoriTextInput(
            normalizedText: normalized, bodyTokenIDs: Array(raw.prefix(255)), originalTokenCount: raw.count)
    }

    private static func message(_ pointer: UnsafeMutablePointer<CChar>?) -> String {
        pointer.map { String(cString: $0) } ?? "Tokenizer failed without an error message"
    }
}
