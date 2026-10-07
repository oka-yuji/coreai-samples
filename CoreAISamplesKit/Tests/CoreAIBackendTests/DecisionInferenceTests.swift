import Darwin
import Foundation
import Testing

@testable import CoreAIBackend

struct DecisionInferenceFixture: Decodable {
    let id: String
    let requestJSON: String
    let tokens: Int
    let probabilities: [[Float]]
    let reference: [[Float]]

    var request: DecisionRequest { get throws { try DecisionRequest(json: Data(requestJSON.utf8)) } }

    static func all() throws -> [Self] {
        guard let folder = Bundle.module.url(forResource: "Fixtures", withExtension: nil) else {
            throw DecisionError.invalidRequest("Missing inference fixtures.")
        }
        return try JSONDecoder().decode(
            [Self].self, from: Data(contentsOf: folder.appendingPathComponent("DecisionInference/verification.json")))
    }
}

private let decisionTestBundle = ProcessInfo.processInfo.environment["COREAI_DECISION_BUNDLE"]

@Suite(.serialized, .enabled(if: decisionTestBundle != nil))
struct DecisionInferenceTests {
    @Test("Ten documents match the lab probabilities and the fixed PyTorch fidelity limits")
    func fixtureParity() async throws {
        let path = try #require(decisionTestBundle)
        let model = try await DecisionModel.load(bundle: URL(filePath: path))
        let fixtures = try DecisionInferenceFixture.all()
        #expect(fixtures.count == 10)
        var bands: [Int: [Float]] = [:]
        var metrics: [[String: Any]] = []
        for fixture in fixtures {
            let result = try await model.decide(fixture.request)
            #expect(result.inputTokens == fixture.tokens)
            try #require(result.answers.count == fixture.probabilities.count)
            var labMax: Float = 0
            var referenceMax: Float = 0
            for index in result.answers.indices {
                let probabilities = result.answers[index].distribution.map(\.probability)
                try #require(probabilities.count == fixture.probabilities[index].count)
                try #require(probabilities.count == fixture.reference[index].count)
                #expect(probabilities.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
                #expect(abs(probabilities.reduce(0, +) - 1) <= 1e-3)
                let labError = zip(probabilities, fixture.probabilities[index]).map { abs($0 - $1) }.max() ?? 0
                let referenceError = zip(probabilities, fixture.reference[index]).map { abs($0 - $1) }.max() ?? 0
                #expect(labError <= 1e-3, "\(fixture.id) question \(index): lab \(labError)")
                labMax = max(labMax, labError)
                referenceMax = max(referenceMax, referenceError)
                let band = [512, 2_048, 4_096, 8_192, 9_216].first { result.inputTokens <= $0 } ?? 9_216
                bands[band, default: []].append(referenceError)
            }
            if fixture.id == "s2d_ctr8k_01" { #expect(result.elapsedSeconds <= 20) }
            metrics.append([
                "id": fixture.id, "inputTokens": result.inputTokens, "seconds": result.elapsedSeconds,
                "maximumLabError": labMax, "maximumReferenceError": referenceMax,
                "footprint": try Self.footprint(),
            ])
        }
        for (band, errors) in bands {
            #expect((errors.max() ?? 1) <= 0.03, "Band \(band)")
            #expect(errors.reduce(0, +) / Float(errors.count) <= 0.01, "Band \(band)")
        }
        try Self.writeMetrics(
            [
                "fixtures": metrics,
                "referenceBands": Dictionary(
                    uniqueKeysWithValues: bands.map { (String($0.key), $0.value.map(Double.init)) }),
            ], name: "parity")
    }

    @Test(
        "Alternating twenty 2K and 8K documents grows footprint by less than 0.1 GB per document",
        .enabled(if: ProcessInfo.processInfo.environment["DECIDE_MEMORY_TEST"] == "1"))
    func alternatingMemory() async throws {
        let path = try #require(decisionTestBundle)
        let model = try await DecisionModel.load(bundle: URL(filePath: path))
        let fixtures = try DecisionInferenceFixture.all()
        let short = try #require(fixtures.first { $0.id == "s2d_ctr2k_01" })
        let long = try #require(fixtures.first { $0.id == "s2d_ctr8k_01" })
        _ = try await model.decide(short.request)
        _ = try await model.decide(long.request)
        let baseline = try Self.footprint()
        var metrics: [[String: Any]] = []
        var measurements: [UInt64] = []
        for index in 0..<20 {
            let fixture = index.isMultiple(of: 2) ? short : long
            let result = try await model.decide(fixture.request)
            let footprint = try Self.footprint()
            measurements.append(footprint)
            metrics.append(["index": index, "id": fixture.id, "seconds": result.elapsedSeconds, "footprint": footprint])
        }
        let firstShort = try #require(measurements.first)
        let last = try #require(measurements.last)
        let lastShort = measurements[18]
        let growth = (Double(last) - Double(baseline)) / Double(measurements.count)
        let shortGrowth = (Double(lastShort) - Double(firstShort)) / 18
        try Self.writeMetrics(
            [
                "documents": metrics, "baselineAfter8K": baseline,
                "bytesPerDocument": growth, "shortBytesPerDocument": shortGrowth,
                "measurement":
                    "Two warm-up documents; compare matching document lengths across twenty measured documents.",
            ], name: "memory")
        #expect(growth < 100_000_000)
        #expect(shortGrowth < 100_000_000)
    }

    static func footprint() throws -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { throw DecisionError.invalidOutput("Cannot measure footprint: \(result).") }
        return info.phys_footprint
    }

    static func writeMetrics(_ values: [String: Any], name: String) throws {
        guard let path = ProcessInfo.processInfo.environment["DECIDE_METRICS_DIR"] else { return }
        let folder = URL(filePath: path)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: values, options: [.prettyPrinted, .sortedKeys])
            .write(to: folder.appendingPathComponent(name + ".json"), options: .atomic)
    }
}
