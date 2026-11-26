import Foundation
import FoundationModels
import Testing

@_spi(Verification) @testable import CoreAIBackend

@Suite(.timeLimit(.minutes(1)))
struct RuntimePreparedCopyTests {
    @Test("Loading with the default policy never requests a recovery ledger, even with a complete hash")
    func disabled() async throws {
        let model = try await ReloadTestModel.make(policy: .init(), hashLength: 32) { _ in
            Issue.record("A disabled load requested the recovery ledger")
            throw ReloadTestFailure.load
        }
        defer { try? FileManager.default.removeItem(at: model.folder) }
        #expect(!model.model.loadStats.recoveredPreparedCopy)
        #expect(!model.control.events.contains("warmUp:1"))
    }

    @Test(
        "A missing, short or long asset hash bypasses recovery on an opted-in load", arguments: [nil, 31, 33] as [Int?])
    func invalidHash(length: Int?) async throws {
        let fixture = try PreparedCopyFixture()
        defer { fixture.remove() }
        let model = try await ReloadTestModel.make(policy: .init(), hashLength: length, recoverPreparedCopy: true) {
            bundle in
            try PreparedCopyRecovery.forBundle(bundle).map { _ in fixture.recovery() }
        }
        defer { try? FileManager.default.removeItem(at: model.folder) }
        #expect(!PreparedCopyFixture.exists(fixture.directory))
        #expect(!model.control.events.contains("warmUp:1"))
        #expect(!model.model.loadStats.recoveredPreparedCopy)
    }

    @Test(
        "A cached load error repairs once and reports whether the repair succeeded",
        arguments: ["success", "delete failure", "load failure", "cancelled"])
    func recoverableLoad(outcome: String) async throws {
        let fixture = try PreparedCopyFixture()
        defer { fixture.remove() }
        let folder = try tinyBundle()
        defer { try? FileManager.default.removeItem(at: folder) }
        let bundle = try ModelBundle(url: folder)
        let recovery = fixture.recovery()
        let control = ReloadTestControl()
        control.state.withLock { $0.failNextLoad = true }
        do {
            let runtime = try await Runtime(
                bundle: bundle, contextLimit: nil, overflow: .error, settings: .init(), recoverPreparedCopy: true,
                onRecovery: { control.state.withLock { $0.events.append("repair") } },
                makeRecovery: { _ in recovery },
                makeEngine: { bundle, limit in
                    do {
                        return try await control.make(bundle, limit)
                    } catch {
                        return try Engine.loadingFunction(markFailure: true) { throw error }
                    }
                },
                deletePreparedEntry: { _ in
                    control.state.withLock {
                        $0.events.append("delete")
                        $0.failNextLoad = outcome == "load failure"
                    }
                    if outcome == "delete failure" { throw ReloadTestFailure.load }
                    if outcome == "cancelled" { throw CancellationError() }
                })
            #expect(outcome == "success")
            #expect(runtime.loadStats.recoveredPreparedCopy)
            #expect(control.events == ["load:1", "repair", "delete", "load:2", "warmUp:2"])
            #expect(recovery.record().marks.isEmpty)
            _ = withExtendedLifetime(runtime) {
                #expect(throws: PreparedCopyError.inUse) { try recovery.clear {} }
            }
        } catch is CancellationError {
            #expect(outcome == "cancelled")
            #expect(recovery.record() == PreparedCopyFixture.empty)
        } catch let error as PreparedCopyError {
            #expect(outcome == "delete failure" || outcome == "load failure")
            #expect(error == .recoveryFailed)
            #expect(recovery.record().failed)
        }
        #expect(control.events.filter { $0 == "repair" }.count == 1)
        #expect(control.events.filter { $0 == "delete" }.count == 1)
        try recovery.clear {}
    }

    @Test(
        "A reload protects the real input read and releases its mark on success, failure and cancellation",
        arguments: ["success", "failure", "cancellation", "masked"])
    func protectsFirstRead(outcome: String) async throws {
        let fixture = try PreparedCopyFixture()
        defer { fixture.remove() }
        let recovery = fixture.recovery()
        let model = try await ReloadTestModel.make(
            policy: .init(readInterval: 0, footprintLimitBytes: 1), hashLength: 32,
            recoverPreparedCopy: true, makeRecovery: { _ in recovery })
        defer { try? FileManager.default.removeItem(at: model.folder) }
        #expect(recovery.record().marks.isEmpty)
        #expect(throws: PreparedCopyError.inUse) { try recovery.clear {} }
        #expect(!model.model.loadStats.recoveredPreparedCopy)
        try fixture.write(PreparedCopyFixture.record([PreparedCopyFixture.alive]))
        model.control.state.withLock {
            $0.hasOnlyMaskedLogits = outcome == "masked"
            $0.footprint = 1
            $0.blockNextRead = true
            $0.readIgnoresCancellation = outcome == "cancellation"
            $0.failNextRead = outcome == "failure"
        }
        try await withTaskCancellationHandler {
            try await withThrowingTaskGroup(of: Bool.self) { group in
                defer { model.control.resumeRead() }
                group.addTask {
                    do {
                        if outcome == "success" {
                            _ = try await model.request("b")
                            return true
                        }
                        let request = directReloadRequest(
                            Transcript(entries: [.prompt(.init(segments: [.text(.init(content: "b"))]))]))
                        try await model.runtime.respond(to: request, streamingInto: .init())
                        return true
                    } catch is CancellationError {
                        #expect(outcome == "cancellation")
                        return false
                    } catch EngineError.missingOutput {
                        #expect(outcome == "masked")
                        return false
                    } catch ReloadTestFailure.read {
                        #expect(outcome == "failure")
                        return false
                    }
                }
                var reads = model.control.reading.stream.makeAsyncIterator()
                try #require(await reads.next() == 2)
                let marks = recovery.record().marks
                #expect(marks.map(\.stage) == [.load, .warmUp])
                #expect(throws: PreparedCopyError.inUse) { try recovery.clear {} }
                if outcome == "cancellation" { group.cancelAll() }
                model.control.resumeRead()
                #expect(try await group.next() == (outcome == "success"))
            }
        } onCancel: {
            model.control.resumeRead()
        }
        #expect(recovery.record().marks == [PreparedCopyFixture.alive])
        #expect(model.control.events.filter { $0.hasPrefix("warmUp:") } == ["warmUp:1"])
        if outcome == "success" {
            #expect(throws: PreparedCopyError.inUse) { try recovery.clear {} }
        } else {
            #expect(await model.model.processedTokens().isEmpty)
            try recovery.clear {}
            model.control.state.withLock { $0.hasOnlyMaskedLogits = false }
            _ = try await model.request("a")
            #expect(model.control.loads == 3)
            #expect(recovery.record().marks.isEmpty)
        }
    }
}
