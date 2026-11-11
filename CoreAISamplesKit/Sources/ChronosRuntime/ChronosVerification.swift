import ChronosCore
import Foundation

public struct ChronosCaseCheck: Codable, Sendable {
    public let id: String
    public let count: Int
    public let passed: Bool
    public let maxAbsoluteError: Double
    public let quantiles: [[[Float]]]
    public let seconds: Double
    public let loaded: Bool
    public let timingBoundary: String

    static func compare(id: String, result: ChronosResult, expected: ChronosPrediction) -> Self {
        var count = 0
        var passed = true
        var maximum = 0.0
        for quantile in 0..<21 {
            for step in 0..<16 {
                let reference = Double(expected.quantiles[quantile][step])
                let error = abs(Double(result.prediction.quantiles[quantile][step]) - reference)
                count += 1
                maximum = max(maximum, error)
                if !error.isFinite || error > 0.0001 + 0.0001 * abs(reference) { passed = false }
            }
        }
        return Self(
            id: id, count: count, passed: passed, maxAbsoluteError: maximum,
            quantiles: [result.prediction.quantiles], seconds: result.seconds, loaded: result.loaded,
            timingBoundary: result.timingBoundary)
    }
}

public struct ChronosVerificationReport: Codable, Sendable {
    public let format: String
    public let identity: ChronosBundleIdentity
    public let shape: [Int]
    public let quantileLevels: [Double]
    public let atol: Double
    public let rtol: Double
    public let count: Int
    public let elements: Int
    public let passed: Bool
    public let cases: [ChronosCaseCheck]
    public let reload: ChronosCaseCheck
    public let reloadMatchesFirst: Bool
}

public enum ChronosVerification {
    public static func run(bundle: ChronosBundle, runtime: ChronosRuntime) async throws -> ChronosVerificationReport {
        do {
            let result = try await compare(bundle: bundle, runtime: runtime)
            await runtime.release()
            return result
        } catch {
            await runtime.release()
            throw error
        }
    }

    private static func compare(bundle: ChronosBundle, runtime: ChronosRuntime) async throws
        -> ChronosVerificationReport
    {
        try bundle.reference.validate()
        var checks: [ChronosCaseCheck] = []
        for item in bundle.reference.cases {
            try Task.checkCancellation()
            let result = try await runtime.predict(item.input())
            guard result.identity == bundle.identity else {
                throw ChronosError.invalidBundle("runtime identity changed")
            }
            checks.append(.compare(id: item.id, result: result, expected: try item.prediction()))
        }
        await runtime.release()
        try Task.checkCancellation()
        guard let first = bundle.reference.cases.first, let firstCheck = checks.first else {
            throw ChronosError.invalidReference
        }
        let reloaded = try await runtime.predict(first.input())
        guard reloaded.loaded, reloaded.identity == bundle.identity else {
            throw ChronosError.invalidBundle("reload identity changed")
        }
        let reload = ChronosCaseCheck.compare(id: first.id, result: reloaded, expected: try first.prediction())
        let repeated = ChronosCaseCheck.compare(
            id: first.id, result: reloaded, expected: try ChronosPrediction(quantiles: firstCheck.quantiles[0]))
        let elements = checks.reduce(0) { $0 + $1.count }
        return ChronosVerificationReport(
            format: "coreai-samples-chronos2-swift-verification/1", identity: bundle.identity,
            shape: ChronosContract.outputShape, quantileLevels: ChronosContract.quantileLevels,
            atol: 0.0001, rtol: 0.0001, count: checks.count, elements: elements,
            passed: checks.count == 18 && elements == 6048 && checks.allSatisfy(\.passed) && reload.passed
                && repeated.passed,
            cases: checks, reload: reload, reloadMatchesFirst: repeated.passed)
    }
}
