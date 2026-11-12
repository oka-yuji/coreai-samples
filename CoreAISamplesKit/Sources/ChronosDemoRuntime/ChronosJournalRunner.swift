import ChronosFeature
import Darwin
import FeatureStore
import Foundation

@MainActor
public final class ChronosJournalRunner {
    private let files: ChronosJournalFiles
    private let actuals: @Sendable (Date, [String]) async throws -> APIWindow
    private let permits: @MainActor () -> Bool
    private let now: @MainActor () -> Date
    private let makeID: @MainActor () -> UUID
    private let device: @MainActor () -> ChronosRecordDevice
    private var pending: (ticket: Int, record: ChronosForecastRecord)?

    public init(
        files: ChronosJournalFiles,
        actuals: @escaping @Sendable (Date, [String]) async throws -> APIWindow,
        permits: @escaping @MainActor () -> Bool = { true },
        now: @escaping @MainActor () -> Date = { Date() }, makeID: @escaping @MainActor () -> UUID = { UUID() },
        device: @escaping @MainActor () -> ChronosRecordDevice = { ChronosJournalRunner.currentDevice() }
    ) {
        self.files = files
        self.actuals = actuals
        self.permits = permits
        self.now = now
        self.makeID = makeID
        self.device = device
    }

    public func run(_ effect: ChronosJournalFeature.Effect, send: (ChronosJournalFeature.Action) -> Void) async {
        let permit = ChronosPublicationPermit()
        let ticket: Int
        switch effect {
        case .save(let value, _), .list(let value), .detail(let value, _), .check(let value, _): ticket = value
        }
        do {
            let action = try await withTaskCancellationHandler {
                try allowed()
                return try await perform(effect, permit: permit)
            } onCancel: {
                permit.revoke()
            }
            try allowed()
            send(action)
        } catch {
            permit.revoke()
            guard !Task.isCancelled, permits() else { return }
            if error is CancellationError {
                send(.cancelled(ticket))
            } else if case .save = effect {
                send(.saveFailed(ticket, .persistence))
            } else {
                send(.failed(ticket, (error as? ChronosDemoFailure) ?? .persistence))
            }
        }
    }

    private func perform(
        _ effect: ChronosJournalFeature.Effect, permit: ChronosPublicationPermit
    ) async throws -> ChronosJournalFeature.Action {
        switch effect {
        case .save(let ticket, let batch):
            if pending?.ticket != ticket {
                if let snapshot = batch.inputSnapshot {
                    guard batch.inputSHA256 == NYCAPIContract.hash(try NYCAPIContract.encode(snapshot)) else {
                        throw ChronosDemoFailure.invalidData
                    }
                }
                pending = (
                    ticket, try ChronosForecastRecord(id: makeID(), savedAt: now(), batch: batch, device: device())
                )
            }
            guard let record = pending?.record else { throw ChronosDemoFailure.invalidData }
            do { try await files.create(record, permit: permit) } catch is CancellationError {
                throw CancellationError()
            } catch {
                let original = error
                guard let existing = try? await files.record(record.id), existing == record else { throw original }
            }
            try allowed()
            return .saved(ticket, record.id)
        case .list(let ticket): return .listed(ticket, try await files.list())
        case .detail(let ticket, let id): return .loaded(ticket, try await files.detail(id))
        case .check(let ticket, let id):
            let record = try await files.record(id)
            try allowed()
            let window = try await actuals(record.input.snapshot.origin, record.areas.map(\.areaID))
            try allowed()
            let check = try ChronosActualCheck(id: makeID(), record: record, checkedAt: now(), actuals: window)
            try await files.append(check, permit: permit)
            try allowed()
            return .loaded(ticket, try await files.detail(id))
        }
    }

    private func allowed() throws {
        try Task.checkCancellation()
        guard permits() else { throw CancellationError() }
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
                if candidate.range(of: "^(Mac|iPhone|iPad)[0-9]+,[0-9]+$", options: .regularExpression) != nil {
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
        observe: @escaping @MainActor (ChronosJournalFeature.Action) -> Void = { _ in }
    ) -> Store<ChronosJournalFeature> {
        let api = api
        let runner = ChronosJournalRunner(
            files: files, actuals: { try await api.window(start: $0, count: 4, areaIDs: $1, refresh: true) },
            permits: permits)
        return Store<ChronosJournalFeature>(initialState: ChronosJournalFeature.State()) { effect, send in
            await runner.run(effect) { action in
                send(action)
                observe(action)
            }
        }
    }
}
