import CoreAIBackend
import CryptoKit
import Foundation
import Testing

@testable import coreai_verify

struct TypedDecideRecordTests {
    private func json(_ value: some Encodable) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: JSONEncoder.records.encode(value)) as? [String: Any])
    }

    @Test("A forwarded row encodes the scorer's snake case keys and a rejected row only its error and token count")
    func rowKeys() throws {
        let forwarded = TypedDecideRecord(
            id: "a", kind: .choice, tokenCount: 3, tokensSha256: tokensDigest([1, 2, 3]), capacity: 256,
            logits: Array(repeating: 0.25, count: 24), probabilities: [0.5, 0.5], probabilitiesT1: [0.5, 0.5], top: 0,
            elapsedSeconds: 0.5)
        #expect(
            Set(try json(forwarded).keys) == [
                "id", "kind", "token_count", "tokens_sha256", "capacity", "logits", "probabilities",
                "probabilities_t1", "top", "elapsed_seconds",
            ])
        let rejected = try json(TypedDecideRecord(id: "b", kind: .noul, error: "tooManyTokens", tokenCount: 8_193))
        let expected: NSDictionary = ["id": "b", "kind": "noul", "error": "tooManyTokens", "token_count": 8_193]
        #expect(rejected as NSDictionary == expected)
    }

    @Test("The token digest is the SHA-256 of the decimal IDs joined by commas")
    func tokenDigest() {
        let expected = SHA256.hash(data: Data("151644,0,42".utf8)).map { String(format: "%02x", $0) }.joined()
        #expect(tokensDigest([151_644, 0, 42]) == expected)
        #expect(tokensDigest([1, 23]) != tokensDigest([12, 3]))
    }

    @Test("The input rows decode the request fields and ignore other keys")
    func inputRows() throws {
        let row = #"{"id": "a", "kind": "score", "state": "s", "question": "q", "options": ["0"], "x": 1}"#
        let data = Data(#"{"set": "F", "rows": [\#(row)]}"#.utf8)
        let rows = try JSONDecoder().decode(TypedDecideInput.self, from: data).rows
        #expect(rows.count == 1 && rows[0].id == "a" && rows[0].kind == .score && rows[0].options == ["0"])
    }
}
