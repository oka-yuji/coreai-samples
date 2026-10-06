import Foundation
import GLiNERCore
import GLiNERRuntime
import Testing

private let environment = ProcessInfo.processInfo.environment
private let fixturesDirectory = URL(
    filePath: environment["GLINER_FIXTURES_DIR"] ?? NSHomeDirectory() + "/Desktop/SelfApps/CoreLLM/models/coreai/gliner"
)
private let tokenizerDirectory = environment["GLINER_TOKENIZER_DIR"].map { URL(filePath: $0) }
private let fixtureFiles = [
    "fixtures/g1_method_i_input_N.json", "fixtures/g1_method_i_input_W.json", "work/g4_swift_fixtures.json",
    "data/g1_preregistration.json",
]
private let hasFixtures = fixtureFiles.allSatisfy {
    FileManager.default.fileExists(atPath: fixturesDirectory.appending(path: $0).path(percentEncoded: false))
}

private struct InputFixture: Decodable {
    struct EntitySpec: Decodable {
        let label: String
        let description: String
    }

    struct Case: Decodable {
        let id: Int
        let modelText: String
        let words: [String]
        let wordOffsetsCodepoint: [[Int]]
        let wordOffsetsUtf16: [[Int]]
        let inputIds: [Int32]
        let textWordIndices: [Int32]
        let queryMarkerIndices: [Int32]

        var wordSpans: [WordSpan] {
            zip(words, wordOffsetsCodepoint).map { WordSpan(text: $0, scalarRange: $1[0]..<$1[1]) }
        }
    }

    let entitySpecs: [EntitySpec]
    let cases: [Case]

    var schema: GLiNERSchema {
        .entities(entitySpecs.map { EntityLabel(name: $0.label, description: $0.description) })
    }
}

private struct SwiftFixture: Decodable {
    struct Inputs: Decodable {
        let inputIds: [Int32]
        let attentionMask: [Int32]
        let wordIdx: [Int32]
        let wordMask: [Int32]
        let queryIdx: [Int32]
        let queryMask: [Int32]
        let clsIdx: [Int32]
        let clsMask: [Int32]
    }

    struct Entry: Decodable {
        let id: String
        let bucket: Int
        let call: String
        let inputs: Inputs
        let queryNames: [String]
        let labels: [String]
        let startMap: [Int]
        let endMap: [Int]
        let text: String
        let sourceText: String?
    }

    let entries: [Entry]
}

private struct Preregistration: Decodable {
    struct Field: Decodable {
        let key: String
        let name: String
    }

    let form: [Field]
}

private func load<Value: Decodable>(_ type: Value.Type, _ path: String) throws -> Value {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(type, from: Data(contentsOf: fixturesDirectory.appending(path: path)))
}

private func difference<Element: Equatable>(_ name: String, _ actual: [Element], _ expected: [Element]) -> String? {
    let index =
        zip(actual, expected).enumerated().first { $0.element.0 != $0.element.1 }?.offset
        ?? (actual.count == expected.count ? nil : min(actual.count, expected.count))
    guard let index else { return nil }
    func value(_ values: [Element]) -> String { values.indices.contains(index) ? "\(values[index])" : "none" }
    return "\(name) differs first at \(index): \(value(actual)) != \(value(expected))"
}

private func padded(_ values: [Int32], to capacity: Int) -> [Int32] {
    values + Array(repeating: 0, count: max(0, capacity - values.count))
}

@Suite(.enabled(if: hasFixtures, "GLINER_FIXTURES_DIR must hold the GL1 and GL4 fixtures"))
struct FixtureParityTests {
    @Test("The word splitter matches the words and offsets of fixture N in all 30 sentences")
    func wordsMatchSplitterN() throws {
        let cases = try load(InputFixture.self, "fixtures/g1_method_i_input_N.json").cases
        try #require(cases.count == 30)
        var mismatches: [String] = []
        for item in cases {
            let words = JapaneseWordSplitter.words(in: item.modelText)
            let utf16 = words.map { item.modelText.utf16Range(fromScalarRange: $0.scalarRange) }
            let expectedUTF16 = item.wordOffsetsUtf16.map { Optional($0[0]..<$0[1]) }
            for problem in [
                difference("words", words.map(\.text), item.words),
                difference("scalar ranges", words.map(\.scalarRange), item.wordSpans.map(\.scalarRange)),
                difference("UTF-16 ranges", utf16, expectedUTF16),
            ] {
                if let problem { mismatches.append("case \(item.id): \(problem)") }
            }
        }
        #expect(mismatches.isEmpty, "\(mismatches.joined(separator: "\n"))")
    }

    @Test(
        "The tokenizer matches the fixture input_ids in all 60 cases of 30 sentences x method (i)",
        .enabled(if: tokenizerDirectory != nil, "set GLINER_TOKENIZER_DIR to the folder with tokenizer.json"))
    func inputIDsMatchFixtures() throws {
        let tokenizer = try GLiNERTokenizer(directory: #require(tokenizerDirectory))
        var compared = 0
        var mismatches: [String] = []
        for splitter in ["N", "W"] {
            let fixture = try load(InputFixture.self, "fixtures/g1_method_i_input_\(splitter).json")
            for item in fixture.cases {
                let words = item.wordSpans
                let input = try PreparedInput(
                    text: item.modelText, schema: fixture.schema, splitWords: { _ in words },
                    tokenize: tokenizer.tokenIDs(for:))
                let capacity = input.tokenCapacity
                for problem in [
                    difference("input_ids", input.inputIDs, padded(item.inputIds, to: capacity)),
                    difference(
                        "attention_mask", input.attentionMask, padded(item.inputIds.map { _ in 1 }, to: capacity)),
                    difference("word_idx", input.wordIndex, padded(item.textWordIndices, to: capacity)),
                    difference("query_idx", input.queryIndex, padded(item.queryMarkerIndices, to: 16)),
                ] {
                    if let problem { mismatches.append("\(splitter) case \(item.id): \(problem)") }
                }
                compared += 1
            }
        }
        #expect(compared == 60)
        #expect(mismatches.isEmpty, "\(mismatches.joined(separator: "\n"))")
    }

    @Test(
        "The layout and the word/query/cls index arrays and masks match the GL4 Swift fixtures at both T=256 and 512",
        .enabled(if: tokenizerDirectory != nil, "set GLINER_TOKENIZER_DIR to the folder with tokenizer.json"))
    func arraysMatchGL4() throws {
        let preprocessor = try GLiNERPreprocessor(tokenizer: GLiNERTokenizer(directory: #require(tokenizerDirectory)))
        let entries = try load(SwiftFixture.self, "work/g4_swift_fixtures.json").entries
        let specs = try load(InputFixture.self, "fixtures/g1_method_i_input_N.json").entitySpecs
        let descriptions = Dictionary(uniqueKeysWithValues: specs.map { ($0.label, $0.description) })
        let form = try load(Preregistration.self, "data/g1_preregistration.json").form
        let names = Dictionary(uniqueKeysWithValues: form.map { ($0.key, $0.name) })
        try #require(entries.count == 418)
        #expect(Set(entries.map(\.bucket)) == [256, 512])
        var mismatches: [String] = []
        for entry in entries {
            let schema: GLiNERSchema
            if entry.call == "x" {
                schema = .entities(entry.queryNames.map { EntityLabel(name: $0, description: descriptions[$0]) })
            } else {
                schema = .classification(
                    name: try #require(names[String(entry.call.dropFirst(2))]), labels: entry.labels)
            }
            let source = entry.sourceText ?? entry.text
            let input = try preprocessor.prepare(text: source, schema: schema, tokenCapacities: [entry.bucket])
            let expected = entry.inputs
            for problem in [
                difference("input_ids", input.inputIDs, expected.inputIds),
                difference("attention_mask", input.attentionMask, expected.attentionMask),
                difference("word_idx", input.wordIndex, expected.wordIdx),
                difference("word_mask", input.wordMask, expected.wordMask),
                difference("query_idx", input.queryIndex, expected.queryIdx),
                difference("query_mask", input.queryMask, expected.queryMask),
                difference("cls_idx", input.classIndex, expected.clsIdx),
                difference("cls_mask", input.classMask, expected.clsMask),
                difference("start_map", input.words.map(\.scalarRange.lowerBound), entry.startMap),
                difference("end_map", input.words.map(\.scalarRange.upperBound), entry.endMap),
                difference("model text", [source + (input.appendedPeriod ? "." : "")], [entry.text]),
                difference("T", [input.tokenCapacity], [entry.bucket]),
            ] {
                if let problem { mismatches.append("\(entry.id) T=\(entry.bucket) \(entry.call): \(problem)") }
            }
        }
        #expect(mismatches.isEmpty, "\(mismatches.joined(separator: "\n"))")
    }
}
