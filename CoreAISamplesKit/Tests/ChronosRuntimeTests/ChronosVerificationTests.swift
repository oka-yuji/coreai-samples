import ChronosCore
import Foundation
import Testing

@testable import ChronosRuntime

@Suite(.timeLimit(.minutes(1)))
struct ChronosVerificationTests {
    @Test("Verification compares every value in every ordered case and repeats after reload")
    func fullComparison() async throws {
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        try folder.writeBundle()
        let bundle = try await ChronosBundle.open(folder.url)
        let factory = VerificationFactory(identity: bundle.identity)
        let runtime = ChronosRuntime(factory: { await factory.make() })
        let report = try await ChronosVerification.run(bundle: bundle, runtime: runtime)
        #expect(report.passed)
        #expect(report.count == 18)
        #expect(report.elements == 6048)
        #expect(report.cases.map(\.id) == ChronosReference.caseIDs)
        #expect(report.cases.map(\.quantiles) == bundle.reference.cases.map(\.quantiles))
        #expect(report.reload.quantiles == report.cases[0].quantiles)
        #expect(report.reload.loaded)
        #expect(await factory.inputs == ChronosReference.rawCounts + [1])
        #expect(await factory.loads == 2)
        #expect(try await runtime.predict(ChronosContext([0])).loaded)
        await runtime.release()
        let saved = try ChronosJSON.encode(report)
        let decoded = try ChronosJSON.decode(ChronosVerificationReport.self, from: saved)
        #expect(decoded.passed)
        #expect(decoded.cases.last?.quantiles[0][20][15] == 17_335)
    }

    @Test("A mismatch in the last quantile or reload prevents verification success", arguments: [18, 19])
    func mismatch(call: Int) async throws {
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        try folder.writeBundle()
        let bundle = try await ChronosBundle.open(folder.url)
        let factory = VerificationFactory(identity: bundle.identity, corruptCall: call)
        let runtime = ChronosRuntime(factory: { await factory.make() })
        let report = try await ChronosVerification.run(bundle: bundle, runtime: runtime)
        #expect(!report.passed)
        #expect(report.elements == 6048)
        if call == 18 { #expect(report.cases.last?.passed == false) } else { #expect(!report.reload.passed) }
        #expect(try await runtime.predict(ChronosContext([0])).loaded)
        await runtime.release()
    }

    @Test("Incomplete inference and changed runtime identity fail and release the runtime", arguments: [false, true])
    func incomplete(changedIdentity: Bool) async throws {
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        try folder.writeBundle()
        let bundle = try await ChronosBundle.open(folder.url)
        let factory = VerificationFactory(
            identity: changedIdentity ? ChronosTestFolder.identity : bundle.identity,
            failCall: changedIdentity ? nil : 18)
        let runtime = ChronosRuntime(factory: { await factory.make() })
        await #expect(throws: (any Error).self) { try await ChronosVerification.run(bundle: bundle, runtime: runtime) }
        #expect(try await runtime.predict(ChronosContext([0])).loaded)
        await runtime.release()
    }

    @Test("The frozen tolerance includes the relative term and checks all 336 values")
    func tolerance() throws {
        let expected = try ChronosPrediction(quantiles: Array(repeating: Array(repeating: 1000, count: 16), count: 21))
        var rows = expected.quantiles
        rows[20][15] += 0.09
        let result = ChronosResult(
            identity: ChronosTestFolder.identity, prediction: try ChronosPrediction(quantiles: rows), rawCount: 1,
            paddedCount: 16, seconds: 0, loaded: false, timingBoundary: "test")
        let check = ChronosCaseCheck.compare(id: "test", result: result, expected: expected)
        #expect(check.passed)
        #expect(check.count == 336)
        #expect(check.maxAbsoluteError > 0.08)
    }
}

private actor VerificationFactory {
    let identity: ChronosBundleIdentity
    let corruptCall: Int?
    let failCall: Int?
    private(set) var loads = 0
    private(set) var inputs: [Int] = []

    init(identity: ChronosBundleIdentity, corruptCall: Int? = nil, failCall: Int? = nil) {
        self.identity = identity
        self.corruptCall = corruptCall
        self.failCall = failCall
    }

    func make() -> any ChronosPredictor {
        loads += 1
        return ChronosFakePredictor(identity: identity, run: { try await self.predict($0) })
    }

    private func predict(_ context: ChronosContext) throws -> ChronosPrediction {
        inputs.append(context.rawCount)
        if inputs.count == failCall { throw ChronosError.invalidModel }
        let offset = Int(context.paddedValues.last ?? 0)
        var rows = ChronosTestFolder.rows(offset: offset)
        if inputs.count == corruptCall { rows[20][15] += 100 }
        return try ChronosPrediction(quantiles: rows)
    }
}
