import ArgumentParser
import Foundation
import Synchronization
import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct TypedDecideThermalTests {
    private let required = ["--bundle", "/not-loaded", "--input", "/in.json", "--output", "/out.json"]

    private func run(_ states: [ProcessInfo.ThermalState], limit: Duration) async throws -> (ThermalStart, [Duration]) {
        let elapsed = Mutex(Duration.zero)
        let reads = Mutex(0)
        let slept = Mutex<[Duration]>([])
        let base = ContinuousClock.now
        let gate = ThermalGate(
            limit: limit,
            state: {
                let index = reads.withLock { count in
                    count += 1
                    return count - 1
                }
                return states[min(index, states.count - 1)]
            },
            now: { base + elapsed.withLock { $0 } },
            sleep: { duration in
                slept.withLock { $0.append(duration) }
                elapsed.withLock { $0 += duration }
            })
        let start = try await gate.wait()
        return (start, slept.withLock { $0 })
    }

    @Test("The thermal wait is off by default and takes a limit of zero or more seconds")
    func option() throws {
        #expect(try TypedDecide.parse(required).waitNominalThermal == nil)
        #expect(try TypedDecide.parse(required + ["--wait-nominal-thermal", "0"]).waitNominalThermal == 0)
        #expect(try TypedDecide.parse(required + ["--wait-nominal-thermal", "600"]).waitNominalThermal == 600)
        let error = #expect(throws: (any Error).self) {
            try TypedDecide.parse(required + ["--wait-nominal-thermal=-1"])
        }
        #expect(TypedDecide.message(for: try #require(error)) == "--wait-nominal-thermal takes zero or more seconds.")
    }

    @Test("A nominal thermal state starts the row without waiting")
    func nominalAtOnce() async throws {
        let (start, slept) = try await run([.nominal], limit: .seconds(600))
        #expect(start == ThermalStart(waitedSeconds: 0, state: .nominal))
        #expect(slept.isEmpty)
    }

    @Test("The gate checks the thermal state every second until it becomes nominal")
    func waitsUntilNominal() async throws {
        let (start, slept) = try await run([.serious, .fair, .nominal], limit: .seconds(600))
        #expect(start == ThermalStart(waitedSeconds: 2, state: .nominal))
        #expect(slept == [.seconds(1), .seconds(1)])
    }

    @Test("The gate stops waiting at its limit and reports the state the row starts with")
    func stopsAtTheLimit() async throws {
        let (start, slept) = try await run([.fair], limit: .seconds(3))
        #expect(start == ThermalStart(waitedSeconds: 3, state: .fair))
        #expect(slept.count == 3)
    }

    @Test("Thermal states are named nominal, fair, serious and critical")
    func stateNames() {
        let states: [ProcessInfo.ThermalState] = [.nominal, .fair, .serious, .critical]
        let names = states.map { ThermalStart(waitedSeconds: 0, state: $0).stateName }
        #expect(names == ["nominal", "fair", "serious", "critical"])
    }

    @Test("A gated row encodes its start time, thermal wait and starting state, and an ungated row has none of them")
    func encoding() throws {
        var row = TypedDecideRecord(
            id: "a", kind: .noul, tokenCount: 3, tokensSha256: tokensDigest([1, 2, 3]), capacity: 256,
            logits: Array(repeating: 0, count: 24), probabilities: [0.5, 0.5], probabilitiesT1: [0.5, 0.5], top: 0,
            elapsedSeconds: 0.5)
        let plain = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder.records.encode(row)) as? [String: Any])
        #expect(plain["started"] == nil && plain["thermal_wait_seconds"] == nil)
        #expect(plain["thermal_state_at_start"] == nil)
        row.setStart(ThermalStart(waitedSeconds: 2.5, state: .fair), at: Date(timeIntervalSince1970: 0))
        let gated = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder.records.encode(row)) as? [String: Any])
        #expect(gated["started"] as? String == "1970-01-01T00:00:00Z")
        #expect(gated["thermal_wait_seconds"] as? Double == 2.5)
        #expect(gated["thermal_state_at_start"] as? String == "fair")
    }
}
