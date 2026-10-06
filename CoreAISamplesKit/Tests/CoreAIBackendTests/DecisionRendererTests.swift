import Foundation
import Testing
import Tokenizers

@testable import CoreAIBackend

private let clefTokenizerFolder = ProcessInfo.processInfo.environment["CLEF_TOKENIZER_DIR"] ?? ""
private let clefTokenizer = Task { try await AutoTokenizer.from(modelFolder: URL(filePath: clefTokenizerFolder)) }

private func scalarIDs(_ text: String) -> [Int] { text.unicodeScalars.map { Int($0.value) } }

struct DecisionFixture: Decodable, CustomTestStringConvertible {
    struct Question: Decodable {
        enum CodingKeys: String, CodingKey { case id, type, instructions, criteria }

        let question: DecisionQuestion

        init(from decoder: any Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let criteria: DecisionQuestion.Criteria =
                switch try container.decode(String.self, forKey: .type) {
                case "noul": .noul(try container.decodeIfPresent([String: String?].self, forKey: .criteria) ?? [:])
                case "choice": .choice(try container.decode([String: String?].self, forKey: .criteria))
                case "score": .score(try container.decode([String?].self, forKey: .criteria))
                case let type:
                    throw DecodingError.dataCorruptedError(forKey: .type, in: container, debugDescription: type)
                }
            question = DecisionQuestion(
                id: try container.decode(String.self, forKey: .id),
                instructions: try container.decodeIfPresent(String.self, forKey: .instructions),
                criteria: criteria
            )
        }
    }

    struct Request: Decodable {
        let state: String
        let questions: [Question]
    }

    struct Expected: Decodable {
        let fragments: [String]
        let ids: [Int]
        let questionSpans: [[Int]]
        let optionSpans: [[[Int]]]
        let optionIDs: [[String]]
        let questionTypes: [Int]
    }

    let id: String
    let request: Request
    let expected: Expected

    var decisionRequest: DecisionRequest {
        DecisionRequest(state: request.state, questions: request.questions.map(\.question))
    }

    var testDescription: String { id }

    static func all() throws -> [DecisionFixture] {
        guard let folder = Bundle.module.url(forResource: "Fixtures", withExtension: nil) else {
            throw CocoaError(.fileNoSuchFile)
        }
        return try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .map { try JSONDecoder().decode(DecisionFixture.self, from: Data(contentsOf: $0)) }
    }
}

struct DecisionRendererTests {
    @Test func fixturesAreThirtyLabRecordsAndTwoEdgeCases() throws {
        let ids = try DecisionFixture.all().map(\.id)
        #expect(ids.count == 32)
        #expect(ids.filter { $0.hasPrefix("edge_") }.count == 2)
    }

    @Test(arguments: try DecisionFixture.all())
    func fragmentsOptionIDsAndTypesMatchThePythonRendering(_ fixture: DecisionFixture) throws {
        let request = fixture.decisionRequest
        #expect(DecisionRenderer.fragments(for: request).map(\.text) == fixture.expected.fragments)
        let rendered = try DecisionRenderer.render(request, limits: DecisionLimits()) { _ in [] }
        #expect(rendered.optionIDs == fixture.expected.optionIDs)
        #expect(rendered.questionTypes == fixture.expected.questionTypes)
    }

    @Test(.enabled(if: !clefTokenizerFolder.isEmpty), arguments: try DecisionFixture.all())
    func tokensAndSpansMatchThePythonRendering(_ fixture: DecisionFixture) async throws {
        let tokenizer = try await clefTokenizer.value
        let rendered = try DecisionRenderer.render(fixture.decisionRequest, tokenizer: tokenizer)
        #expect(rendered.tokenIDs == fixture.expected.ids)
        #expect(rendered.questionSpans.map { [$0.lowerBound, $0.upperBound] } == fixture.expected.questionSpans)
        #expect(rendered.optionSpans.map { $0.map { [$0.lowerBound, $0.upperBound] } } == fixture.expected.optionSpans)
        #expect(rendered.optionIDs == fixture.expected.optionIDs)
        #expect(rendered.questionTypes == fixture.expected.questionTypes)
        #expect(try DecisionRenderer.render(fixture.decisionRequest, tokenizer: tokenizer) == rendered)
    }

    @Test(.enabled(if: !clefTokenizerFolder.isEmpty))
    func aDocumentPastTheTokenLimitIsRefusedWithTheClefTokenizer() async throws {
        let tokenizer = try await clefTokenizer.value
        let fixture = try #require(try DecisionFixture.all().first { $0.id == "s2e_ctr9k_03" })
        var request = fixture.decisionRequest
        #expect(try DecisionRenderer.render(request, tokenizer: tokenizer).tokenIDs.count == fixture.expected.ids.count)
        request.state += String(repeating: " more", count: 20)
        let error = #expect(throws: DecisionError.self) { try DecisionRenderer.render(request, tokenizer: tokenizer) }
        switch error {
        case .tooManyTokens(let count, let limit)?: #expect(limit == 9_216 && count > limit)
        default: Issue.record("Expected tooManyTokens, got \(String(describing: error))")
        }
    }

    @Test func questionAndOptionLimitsAreCheckedBeforeTokenizing() throws {
        func render(_ questions: [DecisionQuestion]) throws -> RenderedDecision {
            try DecisionRenderer.render(DecisionRequest(state: "s", questions: questions), limits: DecisionLimits()) {
                Issue.record("Tokenized \($0)")
                return []
            }
        }
        let options = Dictionary(uniqueKeysWithValues: (1...9).map { ("o\($0)", Optional("option \($0)")) })
        #expect(throws: DecisionError.tooManyQuestions(count: 17, limit: 16)) {
            try render((1...17).map { DecisionQuestion(id: "q\($0)", criteria: .noul()) })
        }
        #expect(throws: DecisionError.tooManyOptions(question: "pick", count: 9, limit: 8)) {
            try render([DecisionQuestion(id: "pick", criteria: .choice(options))])
        }
        #expect(throws: DecisionError.tooManyOptions(question: "rate", count: 9, limit: 8)) {
            try render([DecisionQuestion(id: "rate", criteria: .score(Array(repeating: "level", count: 9)))])
        }
        #expect(throws: DecisionError.noQuestions) { try render([]) }
        #expect(throws: DecisionError.noOptions(question: "pick")) {
            try render([DecisionQuestion(id: "pick", criteria: .choice([:]))])
        }
        #expect(throws: DecisionError.noOptions(question: "rate")) {
            try render([DecisionQuestion(id: "rate", criteria: .score([]))])
        }
        #expect(throws: DecisionError.duplicateQuestion("q")) {
            try render([DecisionQuestion(id: "q", criteria: .noul()), DecisionQuestion(id: "q", criteria: .noul())])
        }
        for (error, limit) in [
            (DecisionError.tooManyQuestions(count: 17, limit: 16), "at most 16 questions"),
            (.tooManyOptions(question: "pick", count: 9, limit: 8), "at most 8 options per question"),
            (.tooManyTokens(count: 9_217, limit: 9_216), "at most 9216 tokens"),
        ] {
            #expect(error.localizedDescription.contains(limit))
        }
    }

    @Test func atMostSixteenQuestionsEightOptionsAndTheTokenLimitAreAccepted() throws {
        func request(stateLength: Int) -> DecisionRequest {
            let options = Dictionary(uniqueKeysWithValues: (1...8).map { ("o\($0)", String?.none) })
            let pick = DecisionQuestion(id: "pick", criteria: .choice(options))
            let levels = [String?](repeating: nil, count: 8)
            let rates = (2...16).map { DecisionQuestion(id: "q\($0)", criteria: .score(levels)) }
            return DecisionRequest(state: String(repeating: "a", count: stateLength), questions: [pick] + rates)
        }
        let fixed = try DecisionRenderer.render(request(stateLength: 0), limits: DecisionLimits(), encode: scalarIDs)
        let longest = try DecisionRenderer.render(
            request(stateLength: 9_216 - fixed.tokenIDs.count), limits: DecisionLimits(), encode: scalarIDs)
        #expect(longest.tokenIDs.count == 9_216)
        #expect(longest.optionSpans.map(\.count) == Array(repeating: 8, count: 16))
        #expect(throws: DecisionError.tooManyTokens(count: 9_217, limit: 9_216)) {
            try DecisionRenderer.render(
                request(stateLength: 9_217 - fixed.tokenIDs.count), limits: DecisionLimits(), encode: scalarIDs)
        }
    }

    @Test func renderingTwiceGivesTheSameResultWhateverTheDictionaryOrder() throws {
        let pairs = (0..<8).map { ("option \($0)", $0.isMultiple(of: 3) ? nil : "meaning \($0)") }
        var forward: [String: String?] = [:]
        var backward: [String: String?] = [:]
        backward.reserveCapacity(1_000)
        for (key, value) in pairs { forward.updateValue(value, forKey: key) }
        for (key, value) in pairs.reversed() { backward.updateValue(value, forKey: key) }
        func render(_ criteria: [String: String?], _ noul: [String: String?]) throws -> RenderedDecision {
            let questions = [
                DecisionQuestion(id: "pick", criteria: .choice(criteria)),
                DecisionQuestion(id: "check", criteria: .noul(noul)),
            ]
            return try DecisionRenderer.render(
                DecisionRequest(state: "s", questions: questions), limits: DecisionLimits(), encode: scalarIDs)
        }
        let first = try render(forward, ["true": "yes", "false": nil])
        #expect(try render(forward, ["true": "yes", "false": nil]) == first)
        #expect(try render(backward, ["false": nil, "true": "yes"]) == first)
        #expect(first.optionIDs == [pairs.map(\.0), ["true", "false"]])
    }
}
