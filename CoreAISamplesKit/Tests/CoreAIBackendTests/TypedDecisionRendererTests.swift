import CryptoKit
import Foundation
import Testing
import Tokenizers

@testable import CoreAIBackend

let jevBundleFolder = ProcessInfo.processInfo.environment["JEV_BUNDLE_DIR"] ?? ""
private let jevTokenizer = Task {
    try await AutoTokenizer.from(modelFolder: URL(filePath: jevBundleFolder).appending(path: "tokenizer"))
}

enum TypedDecisionFixture {
    static func url(_ name: String) throws -> URL {
        guard let folder = Bundle.module.url(forResource: "Fixtures", withExtension: nil) else {
            throw CocoaError(.fileNoSuchFile)
        }
        return folder.appending(path: "TypedDecision/\(name).json")
    }

    static func load<Value: Decodable>(_ name: String, as type: Value.Type) throws -> Value {
        try JSONDecoder().decode(type, from: Data(contentsOf: url(name)))
    }
}

struct TypedDecisionRenderRow: Decodable, CustomTestStringConvertible {
    private struct Document: Decodable { let rows: [TypedDecisionRenderRow] }

    enum CodingKeys: String, CodingKey {
        case id, kind, state, question, options, text, tokens, error
        case tokenCount = "token_count"
    }

    let id: String
    let kind: TypedDecisionKind
    let state: String
    let question: String
    let options: [String]
    let text: String?
    let tokens: [Int]?
    let tokenCount: Int?
    let error: String?

    var request: TypedDecisionRequest {
        TypedDecisionRequest(state: state, question: question, kind: kind, options: options)
    }

    var testDescription: String { id }

    static func rows(_ name: String) throws -> [TypedDecisionRenderRow] {
        try TypedDecisionFixture.load(name, as: Document.self).rows
    }
}

struct TypedDecisionRendererTests {
    private static func name(_ error: TypedDecisionError) -> String {
        switch error {
        case .invalidBundle: "invalidBundle"
        case .invalidOptions: "invalidOptions"
        case .tooManyTokens: "tooManyTokens"
        case .nonFinite: "nonFinite"
        case .invalidHidden: "invalidHidden"
        }
    }

    @Test("The typed decision fixtures load from the test bundle with 8 JGLUE rows and at least 14 other rows")
    func fixturesLoadFromTheTestBundle() throws {
        let rows = try TypedDecisionRenderRow.rows("render")
        #expect(rows.filter { $0.id.hasPrefix("jglue:") }.count == 8)
        #expect(rows.filter { !$0.id.hasPrefix("jglue:") }.count >= 14)
        for name in ["render", "reject", "head", "real-head"] {
            #expect(FileManager.default.fileExists(atPath: try TypedDecisionFixture.url(name).path))
        }
    }

    @Test(
        "The rendered text matches the Python bare-v1 text byte for byte",
        arguments: try TypedDecisionRenderRow.rows("render"))
    func renderedTextMatchesPython(_ row: TypedDecisionRenderRow) throws {
        let text = try #require(row.text)
        #expect(Array(try TypedDecisionRenderer.render(row.request).utf8) == Array(text.utf8))
        #expect(row.tokens?.count == row.tokenCount)
    }

    @Test("The template constant renders the same text as the renderer")
    func templateConstantMatchesTheRenderer() throws {
        let request = TypedDecisionRequest(state: "S", question: "Q", kind: .choice, options: ["x", "y"])
        let expected = TypedDecisionRenderer.template
            .replacingOccurrences(of: "{kind}", with: "choice")
            .replacingOccurrences(of: "{state}", with: "S")
            .replacingOccurrences(of: "{question}", with: "Q")
            .replacingOccurrences(of: "{options}", with: "A) x\nB) y")
        #expect(try TypedDecisionRenderer.render(request) == expected)
    }

    @Test(
        "Invalid option lists fail with the listed error before tokenizing",
        arguments: try TypedDecisionRenderRow.rows("reject"))
    func invalidOptionListsFailBeforeTokenizing(_ row: TypedDecisionRenderRow) {
        let error = #expect(throws: TypedDecisionError.self) {
            try TypedDecisionRenderer.tokenIDs(for: row.request, limit: .max) { _ in
                Issue.record("Tokenized \(row.id)")
                return []
            }
        }
        #expect(error.map(Self.name) == row.error)
    }

    @Test("A fake encoder accepts exactly the token limit and refuses one more token")
    func theTokenLimitIsInclusive() throws {
        let request = TypedDecisionRequest(
            state: "state", question: "question", kind: .noul, options: ["false", "true"])
        let encode: (String) -> [Int] = { $0.unicodeScalars.map { Int($0.value) } }
        let count = try TypedDecisionRenderer.render(request).unicodeScalars.count
        #expect(try TypedDecisionRenderer.tokenIDs(for: request, limit: count, encode: encode).count == count)
        #expect(throws: TypedDecisionError.tooManyTokens(count: count, limit: count - 1)) {
            try TypedDecisionRenderer.tokenIDs(for: request, limit: count - 1, encode: encode)
        }
    }

    @Test("The bundled tokenizer is the file that tokenized the Python fixture", .enabled(if: !jevBundleFolder.isEmpty))
    func theBundledTokenizerMadeTheFixture() throws {
        struct Document: Decodable {
            enum CodingKeys: String, CodingKey { case tokenizerSHA256 = "tokenizer_sha256" }
            let tokenizerSHA256: String
        }
        let expected = try TypedDecisionFixture.load("render", as: Document.self).tokenizerSHA256
        let data = try Data(contentsOf: URL(filePath: jevBundleFolder).appending(path: "tokenizer/tokenizer.json"))
        #expect(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == expected)
    }

    @Test(
        "Token IDs from the bundled tokenizer match the Python fixture",
        .enabled(if: !jevBundleFolder.isEmpty),
        arguments: try TypedDecisionRenderRow.rows("render"))
    func tokenIDsMatchPython(_ row: TypedDecisionRenderRow) async throws {
        let tokenizer = try await jevTokenizer.value
        #expect(try TypedDecisionRenderer.tokenIDs(for: row.request, tokenizer: tokenizer, limit: 8_192) == row.tokens)
    }
}
