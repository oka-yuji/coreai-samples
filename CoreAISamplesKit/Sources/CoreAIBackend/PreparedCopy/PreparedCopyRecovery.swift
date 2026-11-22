import Foundation

struct PreparedCopyRecovery: Sendable {
    private let directory: URL
    private let key: PreparedCopyKey
    private let process: PreparedCopyProcess
    private let liveness: @Sendable (PreparedCopyMark) -> PreparedCopyLiveness
    private let now: @Sendable () -> Double

    init(
        directory: URL, key: PreparedCopyKey, process: PreparedCopyProcess,
        liveness: @escaping @Sendable (PreparedCopyMark) -> PreparedCopyLiveness = PreparedCopyProcess.liveness(of:),
        now: @escaping @Sendable () -> Double = { Date().timeIntervalSince1970 }
    ) {
        self.directory = directory
        self.key = key
        self.process = process
        self.liveness = liveness
        self.now = now
    }

    var file: URL {
        directory.appending(path: "ledger.json")
    }

    func record() -> PreparedCopyRecord {
        PreparedCopyRecord(marks: [], recoveries: 0, suspects: [], failed: false)
    }

    func load<Loaded>(
        isolation: isolated (any Actor)? = #isolation,
        reload: Bool = false, onRecovery: () -> Void = {}, delete: () throws -> Void,
        load: () async throws -> Loaded, warmUp: (Loaded) async throws -> Void
    ) async throws -> PreparedCopyLoad<Loaded> {
        throw POSIXError(.ENOSYS)
    }

    func clear(delete: () throws -> Void) throws {
        throw POSIXError(.ENOSYS)
    }
}
