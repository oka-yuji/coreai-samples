import Foundation
import Testing

@testable import CoreAIBackend

@Suite struct PreparedCopyCancellationTests {
    @Test("Cancellation at the repair notification preserves the old mark without deleting the entry")
    func cancelsBeforeDeletion() async throws {
        let fixture = try PreparedCopyFixture()
        defer { fixture.remove() }
        let recovery = fixture.recovery()
        let original = PreparedCopyFixture.record([PreparedCopyFixture.dead])
        try fixture.write(original)
        let task = Task {
            _ = try await recovery.load(
                onRecovery: { withUnsafeCurrentTask { $0?.cancel() } },
                delete: { Issue.record("Deleting the entry after the request was cancelled") },
                load: { 1 }, warmUp: { _ in })
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(recovery.record() == original)
        let result = try await PreparedCopyRecorder().load(recovery)
        #expect(result.recovered)
    }

    @Test("A cancelled recovery that receives an SDK error from its load never becomes terminal")
    func cancelledLoadError() async throws {
        let fixture = try PreparedCopyFixture()
        defer { fixture.remove() }
        let recovery = fixture.recovery()
        try fixture.write(PreparedCopyFixture.record([PreparedCopyFixture.dead]))
        let task = Task {
            _ = try await recovery.load(
                delete: {},
                load: {
                    withUnsafeCurrentTask { $0?.cancel() }
                    throw POSIXError(.EIO)
                }, warmUp: { (_: Int) in })
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(!recovery.record().failed)
        #expect(recovery.record().marks.isEmpty)
        let result = try await PreparedCopyRecorder().load(recovery)
        #expect(!result.recovered)
    }
}
