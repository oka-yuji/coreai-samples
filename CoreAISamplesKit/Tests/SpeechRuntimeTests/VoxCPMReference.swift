import CryptoKit
import Foundation

@testable import SpeechRuntime

struct VoxCPMReference: Sendable {
    struct Index: Decodable, Sendable {
        let format: Int
        let seed: UInt64
        let samplesPerPatch: Int
        let sentences: [String]
    }

    struct Entry: Decodable, Sendable {
        let file: String
        let type: String
        let shape: [Int]
        let sha256: String
    }

    struct Call: Decodable, Sendable {
        let name: String
        let index: Int

        init(from decoder: any Decoder) throws {
            var container = try decoder.unkeyedContainer()
            name = try container.decode(String.self)
            index = try container.decode(Int.self)
        }
    }

    struct Record: Decodable, Sendable {
        let name: String
        let seed: UInt64
        let patches: Int
        let stoppedPatch: Int
        let ids: [Int32]
        let calls: [Call]
        let arrays: [String: Entry]
    }

    let record: Record
    private let words: [String: [UInt32]]

    static func index(_ root: URL) throws -> Index {
        let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf: root.appending(path: "reference.json")))
        guard index.format == 1, index.samplesPerPatch > 0, !index.sentences.isEmpty else {
            throw VoxCPMGenerationError.invalidInput("reference.json")
        }
        return index
    }

    init(_ folder: URL, index: Index) throws {
        record = try JSONDecoder().decode(Record.self, from: Data(contentsOf: folder.appending(path: "case.json")))
        guard record.seed == index.seed, record.name == folder.lastPathComponent, index.sentences.contains(record.name)
        else { throw VoxCPMGenerationError.invalidInput("case.json") }
        var words: [String: [UInt32]] = [:]
        for (key, entry) in record.arrays {
            let data = try Data(contentsOf: folder.appending(path: entry.file))
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            guard digest == entry.sha256, data.count == entry.shape.reduce(4, *),
                ["float32", "int32"].contains(entry.type)
            else { throw VoxCPMGenerationError.invalidInput(key) }
            words[key] = data.withUnsafeBytes { bytes in
                (0..<(data.count / 4)).map {
                    UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))
                }
            }
        }
        self.words = words
    }

    func slice(_ key: String, _ index: Int) throws -> [UInt32] {
        guard let entry = record.arrays[key], let values = words[key], index >= 0, index < entry.shape[0] else {
            throw VoxCPMGenerationError.invalidInput(key)
        }
        let size = entry.shape.dropFirst().reduce(1, *)
        return Array(values[(index * size)..<((index + 1) * size)])
    }

    func floats(_ key: String, _ index: Int) throws -> [Float] {
        try slice(key, index).map(Float.init(bitPattern:))
    }

    func floats(_ key: String) throws -> [Float] {
        guard let values = words[key] else { throw VoxCPMGenerationError.invalidInput(key) }
        return values.map(Float.init(bitPattern:))
    }

    func input(_ key: String, _ index: Int) throws -> VoxCPMInput {
        let values = try slice(key, index)
        return record.arrays[key]?.type == "int32"
            ? .integers(values.map(Int32.init(bitPattern:))) : .floats(values.map(Float.init(bitPattern:)))
    }

    static func error(_ reference: [Float], _ actual: [Float]) -> Double? {
        guard reference.count == actual.count, !reference.isEmpty, actual.allSatisfy(\.isFinite) else { return nil }
        var difference = 0.0
        var peak = 0.0
        for (expected, value) in zip(reference, actual) {
            difference = max(difference, abs(Double(expected) - Double(value)))
            peak = max(peak, abs(Double(expected)))
        }
        return difference / max(peak, 1e-8)
    }
}
