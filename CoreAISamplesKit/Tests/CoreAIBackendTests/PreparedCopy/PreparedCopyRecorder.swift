@testable import CoreAIBackend

final class PreparedCopyRecorder {
    private(set) var events: [String] = []

    func note(_ event: String) {
        events.append(event)
    }

    func load(_ recovery: PreparedCopyRecovery, reload: Bool = false, label: String = "") async throws
        -> PreparedCopyLoad<Int>
    {
        try await recovery.load(
            reload: reload, onRecovery: { note(label + "onRecovery") }, delete: { note(label + "delete") },
            load: {
                note(label + "load")
                return 1
            }, warmUp: { _ in note(label + "warmUp") })
    }
}
