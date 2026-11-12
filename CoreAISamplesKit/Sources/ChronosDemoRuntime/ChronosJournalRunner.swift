import ChronosFeature
import Darwin
import FeatureStore
import Foundation
import ModelLibraryRuntime

@MainActor
public final class ChronosJournalRunner {
    private struct Pending {
        let ticket: Int
        let id: UUID
        let savedAt: Date
        let batch: ChronosForecastBatch
        let device: ChronosRecordDevice
        var prepared: ChronosJournalFiles.Prepared?
    }

    private let files: ChronosJournalFiles
    private let actuals: @Sendable (Date, [String]) async throws -> APIWindow
    private let permits: @MainActor () -> Bool
    private let retainPermission: @MainActor () -> DemoExecutionPermission?
    private let now: @MainActor () -> Date
    private let makeID: @MainActor () -> UUID
    private let device: @MainActor () -> ChronosRecordDevice
    private var pending: Pending?
    private weak var journal: Store<ChronosJournalFeature>?
    private var tracksJournal = false

    public init(
        files: ChronosJournalFiles,
        actuals: @escaping @Sendable (Date, [String]) async throws -> APIWindow,
        permits: @escaping @MainActor () -> Bool = { true },
        retainPermission: (@MainActor () -> DemoExecutionPermission?)? = nil,
        now: @escaping @MainActor () -> Date = { Date() }, makeID: @escaping @MainActor () -> UUID = { UUID() },
        device: @escaping @MainActor () -> ChronosRecordDevice = { ChronosJournalRunner.currentDevice() }
    ) {
        self.files = files
        self.actuals = actuals
        self.permits = permits
        self.retainPermission = retainPermission ?? { DemoExecutionPermission(isPermitted: permits) }
        self.now = now
        self.makeID = makeID
        self.device = device
    }

    func attach(to journal: Store<ChronosJournalFeature>) {
        self.journal = journal
        tracksJournal = true
    }

    public func run(_ effect: ChronosJournalFeature.Effect, send: (ChronosJournalFeature.Action) -> Void) async {
        let ticket: Int
        switch effect {
        case .save(let value, _), .list(let value), .detail(let value, _), .check(let value, _): ticket = value
        }
        do {
            try allowed(effect)
            let action = try await perform(effect)
            try allowed(effect)
            send(action)
        } catch {
            guard !Task.isCancelled, permits(), current(effect) else { return }
            if error is CancellationError {
                send(.cancelled(ticket))
            } else if case .save = effect {
                send(.saveFailed(ticket, .persistence))
            } else {
                send(.failed(ticket, (error as? ChronosDemoFailure) ?? .persistence))
            }
        }
    }

    private func perform(_ effect: ChronosJournalFeature.Effect) async throws -> ChronosJournalFeature.Action {
        switch effect {
        case .save(let ticket, let batch):
            if pending?.ticket != ticket {
                pending = Pending(ticket: ticket, id: makeID(), savedAt: now(), batch: batch, device: device())
            }
            guard let capture = pending else { throw ChronosDemoFailure.invalidData }
            let prepared: ChronosJournalFiles.Prepared
            if let existing = capture.prepared {
                prepared = existing
            } else {
                prepared = try await files.prepare(
                    id: capture.id, savedAt: capture.savedAt, batch: capture.batch, device: capture.device)
                try allowed(effect)
                guard pending?.ticket == ticket else { throw CancellationError() }
                pending?.prepared = prepared
            }
            do { try await publish(prepared, for: effect) } catch is CancellationError {
                throw CancellationError()
            } catch {
                let original = error
                guard (try? await files.matches(prepared)) == true else { throw original }
            }
            return .saved(ticket, prepared.recordID)
        case .list(let ticket): return .listed(ticket, try await files.list())
        case .detail(let ticket, let id): return .loaded(ticket, try await files.detail(id))
        case .check(let ticket, let id):
            let record = try await files.record(id)
            try allowed(effect)
            let window = try await actuals(record.input.snapshot.origin, record.areas.map(\.areaID))
            try allowed(effect)
            let prepared = try await files.prepare(id: makeID(), record: record, checkedAt: now(), actuals: window)
            try await publish(prepared, for: effect)
            try allowed(effect)
            return .loaded(ticket, try await files.detail(id))
        }
    }

    private func publish(_ prepared: ChronosJournalFiles.Prepared, for effect: ChronosJournalFeature.Effect)
        async throws
    {
        while true {
            try allowed(effect)
            guard let permission = retainPermission() else { throw CancellationError() }
            let attempt = try ChronosPublicationAttempt(permission: permission, journal: journal, effect: effect)
            do {
                try await withTaskCancellationHandler {
                    try Task.checkCancellation()
                    try await files.publish(prepared, permit: attempt.permit)
                } onCancel: { [permit = attempt.permit] in
                    permit.revoke()
                }
                attempt.finish()
                return
            } catch {
                attempt.finish()
                if error is ChronosPublicationInvalidated { continue }
                throw error
            }
        }
    }

    private func current(_ effect: ChronosJournalFeature.Effect) -> Bool {
        guard tracksJournal else { return true }
        guard let journal else { return false }
        return ChronosJournalFeature.permits(effect, in: journal.state)
    }

    private func allowed(_ effect: ChronosJournalFeature.Effect) throws {
        try Task.checkCancellation()
        guard permits(), current(effect) else { throw CancellationError() }
    }

    public static func currentDevice() -> ChronosRecordDevice {
        #if os(macOS)
            let system = "macOS"
            let key = "hw.model"
        #else
            let system = "iOS"
            let key = "hw.machine"
        #endif
        var count = 0
        var hardware = "unknown"
        if sysctlbyname(key, nil, &count, nil, 0) == 0, (1...128).contains(count) {
            var bytes = [CChar](repeating: 0, count: count)
            if sysctlbyname(key, &bytes, &count, nil, 0) == 0 {
                let candidate = String(
                    decoding: bytes.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }, as: UTF8.self)
                if ChronosJournalValidation.acceptsHardwareModel(candidate) {
                    hardware = candidate
                }
            }
        }
        let version = ProcessInfo.processInfo.operatingSystemVersion
        return ChronosRecordDevice(
            hardwareModel: hardware, operatingSystem: system,
            operatingSystemVersion: "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)")
    }
}

@MainActor
public final class ChronosJournalAssembly {
    public static let shared = ChronosJournalAssembly()
    public let files: ChronosJournalFiles
    public let api: NYCAPIClient

    public init(files: ChronosJournalFiles? = nil, api: NYCAPIClient? = nil) {
        let support = URL.applicationSupportDirectory.appending(path: "coreai-samples")
        self.files = files ?? ChronosJournalFiles(directory: support.appending(path: "ChronosJournal-v1"))
        self.api = api ?? NYCAPIClient(cacheDirectory: URL.cachesDirectory.appending(path: "ChronosAPI"))
    }

    public func makeStore(
        permits: @escaping @MainActor () -> Bool = { true },
        retainPermission: (@MainActor () -> DemoExecutionPermission?)? = nil,
        observe: @escaping @MainActor (ChronosJournalFeature.Action) -> Void = { _ in }
    ) -> Store<ChronosJournalFeature> {
        let api = api
        let runner = ChronosJournalRunner(
            files: files, actuals: { try await api.window(start: $0, count: 4, areaIDs: $1, refresh: true) },
            permits: permits, retainPermission: retainPermission)
        let store = Store<ChronosJournalFeature>(initialState: ChronosJournalFeature.State()) { effect, send in
            await runner.run(effect) { action in
                send(action)
                observe(action)
            }
        }
        runner.attach(to: store)
        return store
    }
}
