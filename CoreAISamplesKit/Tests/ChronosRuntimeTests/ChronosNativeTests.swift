import ChronosCore
import CoreAI
import Foundation
import Testing

@testable import ChronosRuntime

struct ChronosNativeTests {
    @Test("Native array extraction reads all values using the actual strides")
    func strides() throws {
        let expected = ChronosTestFolder.rows()
        var values = Array(repeating: Float.nan, count: 2016)
        for row in 0..<21 {
            for step in 0..<16 { values[row * 96 + step * 3] = expected[row][step] }
        }
        let view = NDArray.View(span: values.span, shape: [1, 21, 16], strides: [2016, 96, 3])
        #expect(try ChronosNativePredictor.extract(view).quantiles == expected)
        var invalid = values
        invalid[20 * 96 + 15 * 3] = .nan
        #expect(throws: ChronosError.invalidPrediction) {
            let invalidView = NDArray.View(span: invalid.span, shape: [1, 21, 16], strides: [2016, 96, 3])
            _ = try ChronosNativePredictor.extract(invalidView)
        }
    }

    @Test("Native array extraction rejects incorrect type shape and interleaving")
    func invalidArrays() {
        #expect(throws: ChronosError.invalidPrediction) {
            try ChronosNativePredictor.validateOutput(shape: [1, 21, 16], scalarType: .float16, interleaveLayout: nil)
        }
        #expect(throws: ChronosError.invalidPrediction) {
            try ChronosNativePredictor.validateOutput(shape: [21, 16], scalarType: .float32, interleaveLayout: nil)
        }
        #expect(throws: ChronosError.invalidPrediction) {
            try ChronosNativePredictor.validateOutput(
                shape: [1, 21, 16], scalarType: .float32, interleaveLayout: .init(dimension: 2, factor: 2))
        }
    }

    @Test(
        "Native Chronos matches all fixed references and reloads the first case",
        .timeLimit(.minutes(20)),
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_CHRONOS2_BUNDLE"] != nil))
    func nativeReference() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_CHRONOS2_BUNDLE"])
        let url = URL(fileURLWithPath: path)
        let bundle = try await ChronosBundle.open(url)
        let runtime = ChronosRuntime(bundleURL: url)
        let report = try await ChronosVerification.run(bundle: bundle, runtime: runtime)
        #expect(report.count == 18)
        #expect(report.elements == 6048)
        #expect(report.reload.count == 336)
        #expect(report.passed)
    }
}
