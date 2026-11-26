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
        store.file
    }

    private var store: PreparedCopyStore {
        PreparedCopyStore(directory: directory, key: key)
    }

    func record() -> PreparedCopyRecord {
        store.record()
    }

    func load<Loaded>(
        isolation: isolated (any Actor)? = #isolation,
        reload: Bool = false, shouldRetry: (any Error) -> Bool = { _ in false },
        onRecovery: () -> Void = {}, delete: () throws -> Void,
        load: () async throws -> Loaded, warmUp: (Loaded) async throws -> Void
    ) async throws -> PreparedCopyLoad<Loaded> {
        var attempt = try begin(reload: reload)
        while true {
            if attempt.recovering {
                onRecovery()
                try discard(with: delete)
            }
            let value: Loaded
            do {
                value = try await load()
            } catch {
                if !reload, !attempt.recovering, !Task.isCancelled, shouldRetry(error) {
                    do {
                        attempt = try reserveRepair(after: attempt)
                    } catch {
                        abandon(attempt, after: error)
                        throw error
                    }
                    continue
                }
                abandon(attempt, after: error)
                throw error
            }
            do {
                try advance(attempt)
                try await warmUp(value)
            } catch {
                abandon(attempt, after: error)
                throw error
            }
            try finish(attempt)
            return PreparedCopyLoad(value: value, recovered: attempt.recovering, use: attempt.use)
        }
    }

    func beginReload() throws -> PreparedCopyAttempt {
        try begin(reload: true)
    }

    func loadedReload(_ attempt: PreparedCopyAttempt) throws {
        try advance(attempt)
    }

    func finishReload(_ attempt: PreparedCopyAttempt) throws {
        try finish(attempt)
    }

    func abandonReload(_ attempt: PreparedCopyAttempt, after error: any Error) {
        abandon(attempt, after: error)
    }

    private func reserveRepair(after attempt: PreparedCopyAttempt) throws -> PreparedCopyAttempt {
        try store.locked {
            try store.modify { record in
                guard !record.failed, record.recoveries < 1 else { throw PreparedCopyError.recoveryFailed }
                let others = record.marks.filter { $0.attempt != attempt.mark.attempt }
                guard others.allSatisfy({ liveness($0) == .dead }), try attempt.use.takeExclusive() else {
                    throw PreparedCopyError.preparing
                }
                var mark = attempt.mark
                mark.kind = .recovery
                mark.at = now()
                record.marks = [mark]
                record.recoveries = 1
                record.suspects = others + [attempt.mark]
                return PreparedCopyAttempt(mark: mark, use: attempt.use)
            }
        }
    }

    func clear(delete: () throws -> Void) throws {
        try store.locked {
            guard let use = try store.use(exclusive: true) else { throw PreparedCopyError.inUse }
            try withExtendedLifetime(use) {
                try delete()
                try store.removeLedger()
            }
        }
    }

    private func begin(reload: Bool) throws -> PreparedCopyAttempt {
        try store.locked {
            try store.modify { record in
                let suspects = try plan(record, reload: reload)
                let recovering = !suspects.isEmpty
                guard let use = try store.use(exclusive: recovering) else { throw PreparedCopyError.preparing }
                let mark = PreparedCopyMark(
                    attempt: UUID().uuidString, pid: process.pid, start: process.start,
                    kind: recovering ? .recovery : .normal, stage: .load, at: now())
                if recovering {
                    record.marks = [mark]
                    record.recoveries += 1
                    record.suspects = suspects
                } else {
                    record.marks.append(mark)
                }
                return PreparedCopyAttempt(mark: mark, use: use)
            }
        }
    }

    private func plan(_ record: PreparedCopyRecord, reload: Bool) throws -> [PreparedCopyMark] {
        let dead = record.marks.filter { liveness($0) == .dead }
        let living = record.marks.filter { !dead.contains($0) }
        if record.failed { throw PreparedCopyError.recoveryFailed }
        if living.contains(where: { $0.kind == .recovery }) {
            let stalled = try store.use(exclusive: false) != nil
            throw stalled ? PreparedCopyError.recoveryFailed : PreparedCopyError.preparing
        }
        if dead.isEmpty { return [] }
        if !living.isEmpty { throw PreparedCopyError.preparing }
        if record.recoveries >= 1 { throw PreparedCopyError.recoveryFailed }
        if reload { throw PreparedCopyError.reloadRequired }
        return dead
    }

    private func discard(with delete: () throws -> Void) throws {
        do {
            try delete()
        } catch {
            try? update { $0.failed = true }
            throw error
        }
    }

    private func advance(_ attempt: PreparedCopyAttempt) throws {
        try update { record in
            for index in record.marks.indices where record.marks[index].attempt == attempt.mark.attempt {
                record.marks[index].stage = .warmUp
                record.marks[index].at = now()
            }
        }
    }

    private func abandon(_ attempt: PreparedCopyAttempt, after error: any Error) {
        let terminal = attempt.recovering && !(error is CancellationError)
        try? update { record in
            if terminal {
                record.failed = true
            } else {
                record.marks.removeAll { $0.attempt == attempt.mark.attempt }
            }
        }
    }

    private func finish(_ attempt: PreparedCopyAttempt) throws {
        try store.locked {
            try store.modify { record in
                record.marks.removeAll { $0.attempt == attempt.mark.attempt || liveness($0) == .dead }
                record.recoveries = 0
                record.suspects = []
            }
            if attempt.recovering { try attempt.use.share() }
        }
    }

    private func update(_ change: (inout PreparedCopyRecord) -> Void) throws {
        try store.locked { try store.modify(change) }
    }
}
