@testable import CoreAIBackend

actor PreparedCopyRuntimeFake {
    private var events: [String] = []

    func open(_ recovery: PreparedCopyRecovery) async throws -> [String] {
        let loaded = try await recovery.load(
            delete: { events.append("delete") },
            load: {
                events.append("load")
                return PreparedCopyModelFake()
            },
            warmUp: { model in
                events.append("warmUp")
                model.warmed = true
            })
        events.append(loaded.value.warmed ? "warmed" : "cold")
        return events
    }
}
