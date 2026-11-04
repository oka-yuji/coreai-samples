import CryptoKit
import Foundation
import Testing

@testable import CoreAIBackend

enum TypedDecisionHeadFile {
    static func data(header: String, payload: Data) -> Data {
        var text = header
        while !(8 + text.utf8.count).isMultiple(of: 8) { text += " " }
        var length = UInt64(text.utf8.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(Data(text.utf8))
        data.append(payload)
        return data
    }

    static func payload(_ values: [Float]) -> Data {
        var data = Data()
        for value in values {
            var bits = value.bitPattern.littleEndian
            data.append(withUnsafeBytes(of: &bits) { Data($0) })
        }
        return data
    }

    static func data(weight: [[Float]], bias: [Float]) -> Data {
        let width = weight.first?.count ?? 0
        let biasBytes = bias.count * 4
        let header = """
            {"proj.bias":{"dtype":"F32","shape":[\(bias.count)],"data_offsets":[0,\(biasBytes)]},\
            "proj.weight":{"dtype":"F32","shape":[\(weight.count),\(width)],\
            "data_offsets":[\(biasBytes),\(biasBytes + weight.count * width * 4)]}}
            """
        return data(header: header, payload: payload(bias + weight.flatMap { $0 }))
    }
}

struct TypedDecisionHeadCase: Decodable, CustomTestStringConvertible {
    let id: String
    let kind: TypedDecisionKind
    let count: Int
    let hidden: [Float]
    let logits: [Float]
    let temperature: Double
    let plain: [Double]
    let shipping: [Double]
    let top: Int

    var testDescription: String { id }
}

struct TypedDecisionSyntheticHead: Decodable {
    let width: Int
    let weight: [[Float]]
    let bias: [Float]
    let rows: [TypedDecisionHeadCase]

    static func load() throws -> TypedDecisionSyntheticHead {
        try TypedDecisionFixture.load("head", as: TypedDecisionSyntheticHead.self)
    }
}

struct TypedDecisionRealHead: Decodable {
    struct Row: Decodable, CustomTestStringConvertible {
        enum CodingKeys: String, CodingKey {
            case id, kind, options, tokens, hidden, logits1m, p1m
            case p1mT = "p1m_T"
        }

        let id: String
        let kind: TypedDecisionKind
        let options: [String]
        let tokens: [Int]
        let hidden: [Float]
        let logits1m: [Double]
        let p1m: [Double]
        let p1mT: [Double]

        var testDescription: String { id }
    }

    enum CodingKeys: String, CodingKey {
        case temperatures, rows
        case headSHA256 = "head_sha256"
    }

    let headSHA256: String
    let temperatures: [String: Double]
    let rows: [Row]

    static func load() throws -> TypedDecisionRealHead {
        try TypedDecisionFixture.load("real-head", as: TypedDecisionRealHead.self)
    }
}

struct TypedDecisionHeadDefect: Sendable, CustomTestStringConvertible {
    let testDescription: String
    let data: Data

    static let bias = TypedDecisionHeadFile.payload([Float](repeating: 0, count: 24))
    static let weight = TypedDecisionHeadFile.payload([Float](repeating: 0, count: 96))
    static let biasTensor = #""proj.bias":{"dtype":"F32","shape":[24],"data_offsets":[0,96]}"#

    static func file(dtype: String = "F32", shape: String = "[24,4]", offsets: String, payload: Data? = nil) -> Data {
        TypedDecisionHeadFile.data(
            header:
                "{\(biasTensor),\"proj.weight\":{\"dtype\":\"\(dtype)\",\"shape\":\(shape),\"data_offsets\":\(offsets)}}",
            payload: payload ?? bias + weight)
    }

    static let all: [TypedDecisionHeadDefect] = [
        .init(testDescription: "truncated header", data: Data([1, 2, 3])),
        .init(testDescription: "header size", data: Data([255, 255, 255, 255, 0, 0, 0, 0]) + weight),
        .init(testDescription: "file too small for the width", data: file(offsets: "[96,480]", payload: bias)),
        .init(
            testDescription: "missing weight",
            data: TypedDecisionHeadFile.data(header: "{\(biasTensor)}", payload: bias + weight)),
        .init(testDescription: "dtype", data: file(dtype: "F16", offsets: "[96,480]")),
        .init(testDescription: "shape", data: file(shape: "[4,24]", offsets: "[96,480]")),
        .init(testDescription: "end beyond the file", data: file(offsets: "[192,576]")),
        .init(testDescription: "negative start", data: file(offsets: "[-4,380]")),
        .init(testDescription: "end before start", data: file(offsets: "[1,-9223372036854775808]")),
        .init(testDescription: "length", data: file(offsets: "[96,476]")),
        .init(testDescription: "alignment", data: file(offsets: "[98,482]", payload: bias + weight + Data([0, 0]))),
        .init(
            testDescription: "bias shape",
            data: TypedDecisionHeadFile.data(
                header: #"{"proj.bias":{"dtype":"F32","shape":[23],"data_offsets":[0,96]},"#
                    + #""proj.weight":{"dtype":"F32","shape":[24,4],"data_offsets":[96,480]}}"#,
                payload: bias + weight)),
    ]
}

private func largestDifference(_ actual: [Float], _ expected: [Double]) -> Double {
    guard actual.count == expected.count else { return .infinity }
    return zip(actual, expected).map { abs(Double($0) - $1) }.max() ?? 0
}

private func temporaryFile(_ data: Data) throws -> URL {
    let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    let url = folder.appending(path: "head.safetensors")
    try data.write(to: url)
    return url
}

struct TypedDecisionHeadTests {
    private func syntheticHead() throws -> (TypedDecisionHead, TypedDecisionSyntheticHead) {
        let fixture = try TypedDecisionSyntheticHead.load()
        let url = try temporaryFile(TypedDecisionHeadFile.data(weight: fixture.weight, bias: fixture.bias))
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        return (try TypedDecisionHead(url: url, width: fixture.width), fixture)
    }

    @Test(
        "Synthetic head logits and probabilities at temperature 1 and the shipping temperature match Python",
        arguments: try TypedDecisionSyntheticHead.load().rows)
    func syntheticHeadMatchesPython(_ fixture: TypedDecisionHeadCase) throws {
        let (head, _) = try syntheticHead()
        let logits = try head.logits(hidden: fixture.hidden)
        #expect(largestDifference(logits, fixture.logits.map(Double.init)) <= 1e-6)
        let result = try head.probabilities(
            logits: logits, kind: fixture.kind, count: fixture.count, temperature: Float(fixture.temperature))
        #expect(largestDifference(result.plain, fixture.plain) <= 1e-6)
        #expect(largestDifference(result.shipping, fixture.shipping) <= 1e-6)
        #expect(TypedDecisionHead.top(of: result.shipping) == fixture.top)
    }

    @Test("The synthetic cases cover noul, score, choice with 2, 5 and 16 options and a tie")
    func syntheticCasesCoverEveryKind() throws {
        let rows = try TypedDecisionSyntheticHead.load().rows
        #expect(
            rows.map { "\($0.kind.rawValue)-\($0.count)" }.sorted() == [
                "choice-16", "choice-2", "choice-4", "choice-5", "noul-2", "score-6",
            ])
        let tie = try #require(rows.first { $0.id == "tie" })
        #expect(tie.shipping.filter { $0 == tie.shipping.max() }.count == 2 && tie.top == 1)
    }

    @Test("Non-finite logits and temperatures that are not positive and finite are refused")
    func nonFiniteValuesAreRefused() throws {
        let (head, _) = try syntheticHead()
        let logits = [Float](repeating: 0, count: 24)
        for (index, value) in [(9, Float.nan), (0, .nan), (23, .infinity), (8, -.infinity)] {
            var changed = logits
            changed[index] = value
            #expect(throws: TypedDecisionError.nonFinite("The head logits must be finite.")) {
                try head.probabilities(logits: changed, kind: .choice, count: 2, temperature: 1)
            }
        }
        for temperature in [Float(0), -0.0, -1, .nan, .infinity, -.infinity] {
            #expect(throws: TypedDecisionError.nonFinite("The temperature must be a positive finite number.")) {
                try head.probabilities(logits: logits, kind: .noul, count: 2, temperature: temperature)
            }
        }
    }

    @Test("Option counts outside the slots of a kind and vectors of the wrong length are refused")
    func countsAndLengthsAreRefused() throws {
        let (head, _) = try syntheticHead()
        let logits = [Float](repeating: 0, count: 24)
        for (kind, count) in [
            (TypedDecisionKind.noul, 1), (.noul, 3), (.score, 5), (.score, 7), (.choice, 1), (.choice, 17),
        ] {
            let error = #expect(throws: TypedDecisionError.self) {
                try head.probabilities(logits: logits, kind: kind, count: count, temperature: 1)
            }
            #expect(error == .invalidOptions("\(kind.rawValue) cannot take \(count) options."))
        }
        #expect(throws: TypedDecisionError.invalidHidden(count: 23, expected: 24)) {
            try head.probabilities(logits: Array(logits.dropLast()), kind: .noul, count: 2, temperature: 1)
        }
        #expect(throws: TypedDecisionError.invalidHidden(count: 3, expected: 4)) {
            try head.logits(hidden: [1, 2, 3])
        }
        #expect(throws: TypedDecisionError.invalidHidden(count: 5, expected: 4)) {
            try head.logits(hidden: [1, 2, 3, 4, 5])
        }
    }

    @Test("Ties select the earliest of the largest values")
    func tiesSelectTheEarliestValue() {
        #expect(TypedDecisionHead.top(of: [0.25, 0.5, 0.25, 0.5]) == 1)
        #expect(TypedDecisionHead.top(of: [0.5, 0.5]) == 0)
        #expect(TypedDecisionHead.top(of: [0.1, 0.2, 0.7]) == 2)
    }

    @Test("A head file built the same way as the refused files opens")
    func aWellFormedHeadFileOpens() throws {
        let url = try temporaryFile(TypedDecisionHeadDefect.file(offsets: "[96,480]"))
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let head = try TypedDecisionHead(url: url, width: 4)
        #expect(try head.logits(hidden: [1, 1, 1, 1]) == Array(repeating: 0, count: 24))
    }

    @Test(
        "Head files with a wrong type, shape, range, alignment or length are refused",
        arguments: TypedDecisionHeadDefect.all)
    func malformedHeadFilesAreRefused(_ defect: TypedDecisionHeadDefect) throws {
        let url = try temporaryFile(defect.data)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let error = #expect(throws: TypedDecisionError.self) { try TypedDecisionHead(url: url, width: 4) }
        guard case .invalidBundle? = error else {
            Issue.record("Expected invalidBundle, got \(String(describing: error))")
            return
        }
    }

    @Test(
        "The bundled head reproduces the stage 1' logits within 1e-4 and probabilities within 1e-5",
        .enabled(if: !jevBundleFolder.isEmpty))
    func bundledHeadMatchesTheMergedReference() throws {
        let fixture = try TypedDecisionRealHead.load()
        let url = URL(filePath: jevBundleFolder).appending(path: "head.safetensors")
        let digest = SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
        #expect(digest == fixture.headSHA256)
        let head = try TypedDecisionHead(url: url, width: 5_120)
        #expect(Set(fixture.rows.map(\.kind)) == Set(TypedDecisionKind.allCases))
        #expect(fixture.rows.contains { $0.id.hasPrefix("jglue:") })
        var largest = (logits: 0.0, probabilities: 0.0)
        for row in fixture.rows {
            let logits = try head.logits(hidden: row.hidden)
            let temperature = try #require(fixture.temperatures[row.kind.rawValue])
            let result = try head.probabilities(
                logits: logits, kind: row.kind, count: row.options.count, temperature: Float(temperature))
            let logitDifference = largestDifference(logits, row.logits1m)
            let probabilityDifference = max(
                largestDifference(result.plain, row.p1m), largestDifference(result.shipping, row.p1mT))
            #expect(logitDifference <= 1e-4, "\(row.id)")
            #expect(probabilityDifference <= 1e-5, "\(row.id)")
            largest = (max(largest.logits, logitDifference), max(largest.probabilities, probabilityDifference))
            print("real-head \(row.id) logits \(logitDifference) probabilities \(probabilityDifference)")
        }
        print(
            "real-head rows \(fixture.rows.count) largest logits \(largest.logits) probabilities \(largest.probabilities)"
        )
    }
}
