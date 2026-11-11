import ChronosCore
import Foundation

public typealias ChronosFileIdentity = ChronosCore.ChronosFileIdentity
public typealias ChronosSource = ChronosCore.ChronosSource

struct ChronosABI: Codable, Equatable, Sendable {
    struct Tensor: Codable, Equatable, Sendable {
        let name: String
        let dtype: String
        let shape: [Int]
        var minLength: Int?
        var maxLength: Int?
        var lengthMultiple: Int?
    }

    let asset: String
    let entrypoint: String
    let inputs: [Tensor]
    let outputs: [Tensor]
    let stateNames: [String]

    static let expected = Self(
        asset: "Model.aimodel", entrypoint: "main",
        inputs: [
            Tensor(
                name: "context", dtype: "float32", shape: [1, -1],
                minLength: 16, maxLength: 8192, lengthMultiple: 16)
        ],
        outputs: [Tensor(name: "quantiles", dtype: "float32", shape: ChronosContract.outputShape)],
        stateNames: [])
}

struct ChronosManifest: Codable, Sendable {
    struct RawContext: Codable, Equatable, Sendable {
        let minLength: Int
        let maxLength: Int
        let missing: String
        let padding: String

        static let expected = Self(minLength: 1, maxLength: 8192, missing: "NaN", padding: "NaN on the left")
    }

    let format: String
    let source: ChronosSource
    let precision: String
    let abi: ChronosABI
    let quantiles: [Double]
    let rawContext: RawContext
    let files: [String: ChronosFileIdentity]

    static let fileNames: Set<String> = [
        "LICENSE", "NOTICE", "SOURCE_README.md", "reference.json", "verification.json",
        "Model.aimodel/main.hash", "Model.aimodel/main.mlirb", "Model.aimodel/metadata.json",
    ]

    func validate() throws {
        guard format == ChronosContract.format, source == .expected, precision == "float32",
            abi == .expected, quantiles == ChronosContract.quantileLevels, rawContext == .expected,
            Set(files.keys) == Self.fileNames
        else { throw ChronosError.invalidBundle("manifest contract") }
        for (name, identity) in files {
            try ChronosFiles.validateRelativePath(name)
            try identity.validate()
        }
    }
}

enum ChronosJSON {
    static func decodeExact<Value: Codable>(_ type: Value.Type, from data: Data) throws -> Value {
        let value = try decode(type, from: data)
        let original = try JSONSerialization.jsonObject(with: data)
        let encoded = try JSONSerialization.jsonObject(with: encode(value))
        try checkKeys(original, encoded)
        return value
    }

    static func decode<Value: Decodable>(_ type: Value.Type, from data: Data) throws -> Value {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(type, from: data)
    }

    static func encode(_ value: some Encodable) throws -> Data {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes]
        return try encoder.encode(value)
    }

    private static func checkKeys(_ original: Any, _ encoded: Any) throws {
        if let object = original as? [String: Any], let expected = encoded as? [String: Any] {
            guard Set(object.keys) == Set(expected.keys) else { throw ChronosError.invalidBundle("JSON fields") }
            for key in object.keys {
                if let value = object[key], let expectedValue = expected[key] { try checkKeys(value, expectedValue) }
            }
        } else if let array = original as? [Any], let expected = encoded as? [Any] {
            guard array.count == expected.count else { throw ChronosError.invalidBundle("JSON array") }
            for (value, expectedValue) in zip(array, expected) { try checkKeys(value, expectedValue) }
        }
    }
}
