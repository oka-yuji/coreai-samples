import Foundation
import Testing

@testable import CoreAIBackend

@Suite struct PreparedCopyRetryTests {
    @Test("An initial cached function load error reserves one repair and repeats the load once")
    func repairsOnce() async throws {
        let fixture = try PreparedCopyFixture()
        defer { fixture.remove() }
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        let loaded = try await recovery.load(
            shouldRetry: { ($0 as? POSIXError)?.code == .EIO },
            onRecovery: { recorder.note("reserve") },
            delete: {
                #expect(recovery.record().recoveries == 1)
                #expect(recovery.record().marks.first?.kind == .recovery)
                recorder.note("delete")
            },
            load: {
                recorder.note("load")
                if recorder.events.count == 1 { throw POSIXError(.EIO) }
                return 7
            }, warmUp: { _ in recorder.note("warmUp") })
        #expect(loaded.value == 7 && loaded.recovered)
        #expect(recorder.events == ["load", "reserve", "delete", "load", "warmUp"])
        #expect(recovery.record() == PreparedCopyFixture.empty)
    }

    @Test("A second load error leaves a terminal repair and no subsequent load retries it")
    func failedRepair() async throws {
        let fixture = try PreparedCopyFixture()
        defer { fixture.remove() }
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        await #expect(throws: POSIXError(.EIO)) {
            _ = try await recovery.load(
                shouldRetry: { _ in true }, delete: { recorder.note("delete") },
                load: { throw POSIXError(.EIO) }, warmUp: { (_: Int) in })
        }
        await #expect(throws: PreparedCopyError.recoveryFailed) {
            _ = try await recorder.load(recovery)
        }
        #expect(recovery.record().failed)
        #expect(recorder.events == ["delete"])
    }

    @Test("An already loaded model keeps its lease when a second load requests an immediate repair")
    func inUse() async throws {
        let fixture = try PreparedCopyFixture()
        defer { fixture.remove() }
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        let held = try await recorder.load(recovery)
        await #expect(throws: PreparedCopyError.preparing) {
            _ = try await recovery.load(
                shouldRetry: { _ in true }, delete: { recorder.note("delete") },
                load: { throw POSIXError(.EIO) }, warmUp: { (_: Int) in })
        }
        #expect(recovery.record() == PreparedCopyFixture.empty)
        #expect(recorder.events == ["load", "warmUp"])
        withExtendedLifetime(held) {}
    }

    @Test("A warm-up error never triggers the cached function load retry")
    func warmUpError() async throws {
        let fixture = try PreparedCopyFixture()
        defer { fixture.remove() }
        let recovery = fixture.recovery()
        let recorder = PreparedCopyRecorder()
        await #expect(throws: POSIXError(.EIO)) {
            _ = try await recovery.load(
                shouldRetry: { _ in true }, delete: { recorder.note("delete") },
                load: { 1 }, warmUp: { _ in throw POSIXError(.EIO) })
        }
        #expect(recorder.events.isEmpty)
        #expect(recovery.record() == PreparedCopyFixture.empty)
    }
}
