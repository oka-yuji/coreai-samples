import ChronosCore
import Foundation

struct ChronosReferenceIdentity: Codable, Equatable, Sendable {
    let source: ChronosSource
    let model: ChronosFileIdentity
    let fixtures: ChronosFileIdentity

    func validate() throws {
        guard source == .expected else { throw ChronosError.invalidReference }
        try model.validate()
        try fixtures.validate()
    }
}

public struct ChronosReference: Codable, Sendable {
    struct Case: Codable, Sendable {
        let id: String
        let context: [Float?]
        let quantiles: [[[Float]]]

        func input() throws -> ChronosContext {
            try ChronosContext(context.map { $0 ?? .nan })
        }

        func prediction() throws -> ChronosPrediction {
            guard quantiles.count == 1, let rows = quantiles.first else { throw ChronosError.invalidReference }
            return try ChronosPrediction(quantiles: rows)
        }
    }

    let format: String
    let identity: ChronosReferenceIdentity
    let precision: String
    let abi: ChronosABI
    let quantiles: [Double]
    let atol: Double
    let rtol: Double
    let cases: [Case]

    static let caseIDs = [
        "trend-1", "trend-15", "trend-16", "trend-17", "trend-32", "trend-64", "trend-1008", "trend-8192",
        "constant-zero", "constant-positive", "periodic", "negative-scale", "missing-16", "missing-1008",
        "missing-8192", "single-observed", "large-counts", "leading-missing",
    ]
    static let rawCounts = [
        1, 15, 16, 17, 32, 64, 1008, 8192, 1008, 1008, 1008, 1008, 16, 1008, 8192, 1008, 1008, 1008,
    ]

    func validate() throws {
        try identity.validate()
        guard format == "coreai-samples-chronos2-reference/1", precision == "float32",
            abi == .expected, quantiles == ChronosContract.quantileLevels, atol == 0.0001, rtol == 0.0001,
            cases.map(\.id) == Self.caseIDs, cases.map({ $0.context.count }) == Self.rawCounts
        else { throw ChronosError.invalidReference }
        for item in cases {
            _ = try item.input()
            _ = try item.prediction()
        }
    }
}

struct ChronosPriorVerification: Codable {
    struct Check: Codable {
        let id: String
        let count: Int
        let passed: Bool
        let maxAbsoluteError: Double
    }

    struct Runtime: Codable, Equatable {
        let functionNames: [String]
        let inputNames: [String]
        let outputNames: [String]
        let stateNames: [String]
    }

    struct Inspection: Codable {
        struct Storage: Codable {
            let name: String
            let count: Int
        }

        struct Metadata: Codable, Equatable {
            let author: String
            let license: String
            let producer: String
            let sourceRevision: String
        }

        let abi: ChronosABI
        let compatibilityVersion: String
        let computeTypes: [String]
        let storageTypes: [Storage]
        let metadata: Metadata

        func validate() throws {
            let allowed: Set<String> = ["Bool", "Float32", "Int32", "Int64", "UInt32", "UInt64"]
            let storage = storageTypes.map(\.name)
            guard abi == .expected, compatibilityVersion == "27.0",
                Set(computeTypes).isSubset(of: allowed), Set(storage).isSubset(of: allowed),
                computeTypes.contains("Float32"), storage.contains("Float32"),
                Set(computeTypes).count == computeTypes.count, Set(storage).count == storage.count,
                storageTypes.allSatisfy({ $0.count > 0 }),
                metadata
                    == Metadata(
                        author: "Amazon", license: "Apache-2.0", producer: "coreai-core 1.0.0b3",
                        sourceRevision: ChronosContract.revision)
            else { throw ChronosError.invalidBundle("verification inspection") }
        }
    }

    let identity: ChronosReferenceIdentity
    let atol: Double
    let rtol: Double
    let count: Int
    let elements: Int
    let passed: Bool
    let cases: [Check]
    let runtime: Runtime
    let inspection: Inspection

    func validate(reference: ChronosReference) throws {
        guard identity == reference.identity, atol == reference.atol, rtol == reference.rtol,
            count == 18, elements == 6048, passed, cases.map(\.id) == ChronosReference.caseIDs,
            cases.allSatisfy({
                $0.count == 336 && $0.passed && $0.maxAbsoluteError.isFinite && $0.maxAbsoluteError >= 0
            }),
            runtime
                == Runtime(functionNames: ["main"], inputNames: ["context"], outputNames: ["quantiles"], stateNames: [])
        else { throw ChronosError.invalidBundle("verification contract") }
        try inspection.validate()
    }
}
