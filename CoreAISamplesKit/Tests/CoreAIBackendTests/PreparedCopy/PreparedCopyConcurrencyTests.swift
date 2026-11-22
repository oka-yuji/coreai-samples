import Foundation
import Testing

@testable import CoreAIBackend

@Suite(.serialized, .timeLimit(.minutes(1))) struct PreparedCopyConcurrencyTests {
    private typealias Fixture = PreparedCopyFixture

    @Test(
        "Eight concurrent normal loads of one key leave all eight marks in the ledger and an empty record after they finish"
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
                                try await gate.arrive()
                                return 1
                            }, warmUp: { _ in })
                        return loaded.recovered
                    } catch {
                        if !entered { try await gate.arrive(waiting: false) }
                        throw error
                    }
                }
            }
            try await gate.allArrived()
            #expect(fixture.lockIsFree())
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
        "Eight concurrent loads that see one leftover mark start exactly one recovery, the others return preparing and the entry is deleted once"
    )
    func concurrentLoadsStartOneRecovery() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.write(Fixture.record([Fixture.dead]))
        let gate = PreparedCopyGate(expected: Fixture.crowd.count)
        let recoveries = Fixture.crowd.map { fixture.recovery($0) }
        let outcomes = try await withThrowingTaskGroup(of: (String, Int).self) { group in
            for recovery in recoveries {
                group.addTask {
                    var entered = false
                    var deletes = 0
                    do {
                        let loaded = try await recovery.load(
                            delete: { deletes += 1 },
                            load: {
                                entered = true
                                try await gate.arrive()
                                return 1
                            }, warmUp: { _ in })
                        return (loaded.recovered ? "recovered" : "normal", deletes)
                    } catch PreparedCopyError.preparing where !entered {
                        try await gate.arrive(waiting: false)
                        return ("preparing", deletes)
                    } catch {
                        if !entered { try await gate.arrive(waiting: false) }
                        throw error
                    }
                }
            }
            try await gate.allArrived()
            #expect(fixture.lockIsFree())
            let entered = await gate.entered
            #expect(entered == 1)
            await gate.open()
            return try await group.reduce(into: [(String, Int)]()) { $0.append($1) }
        }
        #expect(outcomes.filter { $0.0 == "recovered" }.count == 1)
        #expect(outcomes.filter { $0.0 == "preparing" }.count == Fixture.crowd.count - 1)
        #expect(outcomes.map(\.1).reduce(0, +) == 1)
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
