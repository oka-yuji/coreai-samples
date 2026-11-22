import Foundation
import Testing

@testable import CoreAIBackend

@Suite(.timeLimit(.minutes(1))) struct PreparedCopyConcurrencyTests {
    private typealias Fixture = PreparedCopyFixture

    @Test(
        "Sixteen concurrent normal loads of one key leave all sixteen marks in the ledger and an empty record after they finish"
    )
    func concurrentLoadsKeepEveryMark() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let gate = PreparedCopyGate(expected: Fixture.crowd.count)
        let recoveries = Fixture.crowd.map { fixture.recovery($0) }
        let results = try await withThrowingTaskGroup(of: Bool.self) { group in
            for recovery in recoveries {
                group.addTask {
                    var entered = false
                    do {
                        let loaded = try await recovery.load(
                            delete: {},
                            load: {
                                entered = true
                                await gate.arrive()
                                return 1
                            }, warmUp: { _ in })
                        return loaded.recovered
                    } catch {
                        if !entered { await gate.arrive(waiting: false) }
                        throw error
                    }
                }
            }
            await gate.allArrived()
            let record = fixture.recovery().record()
            #expect(record.marks.count == Fixture.crowd.count)
            #expect(Set(record.marks.map(\.pid)) == Set(Fixture.crowd.map(\.pid)))
            await gate.open()
            return try await group.reduce(into: [Bool]()) { $0.append($1) }
        }
        #expect(results == Array(repeating: false, count: Fixture.crowd.count))
        #expect(fixture.recovery().record() == Fixture.empty)
    }

    @Test(
        "An actor loads a value that is not sendable with closures that change its own state and reads both afterwards")
    func actorLoadsNonSendableValue() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let runtime = PreparedCopyRuntimeFake()
        let events = try await runtime.open(fixture.recovery())
        #expect(events == ["load", "warmUp", "warmed"])
        #expect(fixture.recovery().record() == Fixture.empty)
    }
}
