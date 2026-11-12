import ChronosFeature
import Darwin
import Foundation
import Synchronization

public final class ChronosPublicationPermit: Sendable {
    private enum State { case allowed, invalidated, revoked }
    private let state = Mutex(State.allowed)
    public init() {}
    public func revoke() { state.withLock { $0 = .revoked } }

    func invalidate() {
        state.withLock { if case .allowed = $0 { $0 = .invalidated } }
    }

    func publish(_ operation: () throws -> Void) throws {
        try state.withLock {
            try Task.checkCancellation()
            switch $0 {
            case .allowed: break
            case .invalidated: throw ChronosPublicationInvalidated()
            case .revoked: throw CancellationError()
            }
            try operation()
        }
    }
}

struct ChronosPublicationInvalidated: Error {}

public actor ChronosJournalFiles {
    struct Prepared: Sendable {
        fileprivate let record: ChronosForecastRecord
        fileprivate let check: ChronosActualCheck?
        fileprivate let data: Data

        var recordID: UUID { record.id }
    }

    public static let payloadLimit = 192 * 1024 * 1024
    static let fileLimit = 256 * 1024 * 1024 + 4096
    private let directory: URL
    private let profile: NYCVerificationProfile?
    private let write: @Sendable (Data, FileHandle) throws -> Void
    private let beforePublish: @Sendable (ChronosPublicationPermit) throws -> Void

    public init(
        directory: URL, verification: NYCVerificationProfile? = nil,
        write: @escaping @Sendable (Data, FileHandle) throws -> Void = { try $1.write(contentsOf: $0) },
        beforePublish: @escaping @Sendable (ChronosPublicationPermit) throws -> Void = { _ in }
    ) {
        self.directory = directory
        profile = verification
        self.write = write
        self.beforePublish = beforePublish
    }

    public func create(_ record: ChronosForecastRecord, permit: ChronosPublicationPermit) throws {
        try publish(prepare(record), permit: permit)
    }

    func prepare(
        id: UUID, savedAt: Date, batch: ChronosForecastBatch, device: ChronosRecordDevice
    ) throws -> Prepared {
        if let snapshot = batch.inputSnapshot {
            guard batch.inputSHA256 == NYCAPIContract.hash(try NYCAPIContract.encode(snapshot)) else {
                throw ChronosDemoFailure.invalidData
            }
        }
        return try prepare(ChronosForecastRecord(id: id, savedAt: savedAt, batch: batch, device: device))
    }

    private func prepare(_ record: ChronosForecastRecord) throws -> Prepared {
        try Task.checkCancellation()
        try ChronosJournalValidation.record(record, profile: profile)
        let data = try envelope(record, id: record.id, recordID: nil)
        return Prepared(record: record, check: nil, data: data)
    }

    func prepare(
        id: UUID, record: ChronosForecastRecord, checkedAt: Date, actuals: APIWindow
    ) throws -> Prepared {
        try prepare(ChronosActualCheck(id: id, record: record, checkedAt: checkedAt, actuals: actuals), record: record)
    }

    private func prepare(_ check: ChronosActualCheck, record: ChronosForecastRecord) throws -> Prepared {
        try Task.checkCancellation()
        try ChronosJournalValidation.check(check, record: record, profile: profile)
        return try Prepared(record: record, check: check, data: envelope(check, id: check.id, recordID: record.id))
    }

    func matches(_ prepared: Prepared) throws -> Bool {
        try record(prepared.recordID) == prepared.record
    }

    func publish(_ prepared: Prepared, permit: ChronosPublicationPermit) throws {
        if let check = prepared.check {
            try append(prepared, check: check, permit: permit)
        } else {
            try create(prepared, permit: permit)
        }
    }

    private func create(_ prepared: Prepared, permit: ChronosPublicationPermit) throws {
        try withRoot { root in
            try cleanup(root)
            let pending = ".pending-" + UUID().uuidString
            guard mkdirat(root, pending, 0o700) == 0 else { throw ChronosDemoFailure.persistence }
            let folder = try ChronosJournalBoundary.folder(pending, in: root)
            var published = false
            defer {
                if !published { unlinkat(folder, "prediction.json", 0) }
                close(folder)
                unlinkat(root, pending, AT_REMOVEDIR)
            }
            try writeFile(prepared.data, name: "prediction.json", in: folder)
            try beforePublish(permit)
            try permit.publish {
                guard renameatx_np(root, pending, root, prepared.recordID.uuidString, UInt32(RENAME_EXCL)) == 0 else {
                    throw ChronosDemoFailure.persistence
                }
                published = true
            }
        }
    }

    public func append(_ check: ChronosActualCheck, permit: ChronosPublicationPermit) throws {
        let record = try record(check.recordID)
        try publish(prepare(check, record: record), permit: permit)
    }

    private func append(
        _ prepared: Prepared, check: ChronosActualCheck, permit: ChronosPublicationPermit
    ) throws {
        try withRoot { root in
            let record = try readRecord(check.recordID, in: root)
            guard record == prepared.record else { throw ChronosDemoFailure.invalidData }
            let folder = try ChronosJournalBoundary.folder(check.recordID.uuidString, in: root)
            defer { close(folder) }
            guard mkdirat(folder, "checks", 0o700) == 0 || errno == EEXIST else {
                throw ChronosDemoFailure.persistence
            }
            let checks = try ChronosJournalBoundary.folder("checks", in: folder)
            defer { close(checks) }
            try cleanup(checks)
            let pending = ".pending-" + UUID().uuidString + ".tmp"
            defer { unlinkat(checks, pending, 0) }
            try writeFile(prepared.data, name: pending, in: checks)
            try beforePublish(permit)
            try permit.publish {
                guard renameatx_np(checks, pending, checks, check.id.uuidString + ".json", UInt32(RENAME_EXCL)) == 0
                else { throw ChronosDemoFailure.persistence }
            }
        }
    }

    public func record(_ id: UUID) throws -> ChronosForecastRecord {
        try withRoot { try readRecord(id, in: $0) }
    }

    public func list() throws -> [ChronosJournalEntry] {
        try withRoot { root in
            try ChronosJournalBoundary.names(root).compactMap { name in
                try Task.checkCancellation()
                guard let id = UUID(uuidString: name), id.uuidString == name else { return nil }
                do {
                    return ChronosJournalEntry(id: id, record: try readRecord(id, in: root))
                } catch is CancellationError {
                    throw CancellationError()
                } catch { return ChronosJournalEntry(id: id, record: nil) }
            }.sorted { ($0.savedAt ?? .distantPast, $0.id.uuidString) > ($1.savedAt ?? .distantPast, $1.id.uuidString) }
        }
    }

    public func detail(_ id: UUID) throws -> ChronosJournalDetail {
        try withRoot { root in
            let record = try readRecord(id, in: root)
            let folder = try ChronosJournalBoundary.folder(id.uuidString, in: root)
            defer { close(folder) }
            if try !ChronosJournalBoundary.names(folder).contains("checks") {
                return ChronosJournalDetail(record: record, checks: [])
            }
            let checks = try ChronosJournalBoundary.folder("checks", in: folder)
            defer { close(checks) }
            let values: [ChronosActualCheck] = try ChronosJournalBoundary.names(checks).compactMap { name in
                if name.hasPrefix(".pending-") { return nil }
                guard name.hasSuffix(".json"), let checkID = UUID(uuidString: String(name.dropLast(5))),
                    name == checkID.uuidString + ".json"
                else { throw ChronosDemoFailure.invalidData }
                let data = try readEnvelope(name, in: checks, id: checkID, recordID: id)
                let value = try ChronosJournalCodec.decode(ChronosActualCheck.self, data)
                guard value.id == checkID else { throw ChronosDemoFailure.invalidData }
                try ChronosJournalValidation.check(value, record: record, profile: profile)
                return value
            }
            return ChronosJournalDetail(
                record: record,
                checks: values.sorted { ($0.checkedAt, $0.id.uuidString) > ($1.checkedAt, $1.id.uuidString) })
        }
    }

    private func readRecord(_ id: UUID, in root: Int32) throws -> ChronosForecastRecord {
        let folder = try ChronosJournalBoundary.folder(id.uuidString, in: root)
        defer { close(folder) }
        let data = try readEnvelope("prediction.json", in: folder, id: id, recordID: nil)
        let record = try ChronosJournalCodec.decode(ChronosForecastRecord.self, data)
        guard record.id == id else { throw ChronosDemoFailure.invalidData }
        try ChronosJournalValidation.record(record, profile: profile)
        return record
    }

    private func envelope<T: Encodable>(_ value: T, id: UUID, recordID: UUID?) throws -> Data {
        let payload = try ChronosJournalCodec.encode(value)
        guard payload.count <= Self.payloadLimit else { throw ChronosDemoFailure.invalidData }
        return try ChronosJournalCodec.encode(
            Envelope(
                schema: "chronos-journal-envelope/2", id: id, recordID: recordID, payload: payload,
                sha256: NYCAPIContract.hash(payload)))
    }

    private func readEnvelope(_ name: String, in folder: Int32, id: UUID, recordID: UUID?) throws -> Data {
        let data = try ChronosJournalBoundary.read(name, in: folder, limit: Self.fileLimit)
        let value = try ChronosJournalCodec.decode(Envelope.self, data)
        guard value.schema == "chronos-journal-envelope/2", value.id == id, value.recordID == recordID,
            value.payload.count <= Self.payloadLimit, NYCAPIContract.hash(value.payload) == value.sha256
        else { throw ChronosDemoFailure.invalidData }
        return value.payload
    }

    private func writeFile(_ data: Data, name: String, in folder: Int32) throws {
        try Task.checkCancellation()
        guard data.count <= Self.fileLimit else { throw ChronosDemoFailure.invalidData }
        let descriptor = openat(folder, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else { throw ChronosDemoFailure.persistence }
        let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? file.close() }
        try write(data, file)
        try file.synchronize()
        guard try ChronosJournalBoundary.read(name, in: folder, limit: Self.fileLimit) == data else {
            throw ChronosDemoFailure.persistence
        }
    }

    private func withRoot<T>(_ operation: (Int32) throws -> T) throws -> T {
        try Task.checkCancellation()
        guard directory.isFileURL, directory.path.hasPrefix("/"), !directory.path.contains("\0") else {
            throw ChronosDemoFailure.invalidData
        }
        let parent = directory.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let ancestor = open(parent.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard ancestor >= 0 else { throw ChronosDemoFailure.persistence }
        defer { close(ancestor) }
        let name = directory.lastPathComponent
        guard !["", ".", ".."].contains(name), !name.contains("\0"),
            mkdirat(ancestor, name, 0o700) == 0 || errno == EEXIST
        else { throw ChronosDemoFailure.persistence }
        let root = try ChronosJournalBoundary.folder(name, in: ancestor)
        defer { close(root) }
        let lock = openat(root, ".lock", O_CREAT | O_RDWR | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, 0o600)
        guard lock >= 0 else { throw ChronosDemoFailure.persistence }
        defer { close(lock) }
        var info = stat()
        guard fstat(lock, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_nlink == 1,
            flock(lock, LOCK_EX) == 0
        else { throw ChronosDemoFailure.persistence }
        defer { flock(lock, LOCK_UN) }
        try Task.checkCancellation()
        return try operation(root)
    }

    private func cleanup(_ root: Int32) throws {
        for name in try ChronosJournalBoundary.names(root) where name.hasPrefix(".pending-") {
            let suffix = String(name.dropFirst(9))
            var info = stat()
            guard fstatat(root, name, &info, AT_SYMLINK_NOFOLLOW) == 0 else { continue }
            if name.hasSuffix(".tmp"), UUID(uuidString: String(suffix.dropLast(4))) != nil,
                info.st_mode & S_IFMT == S_IFREG, info.st_nlink == 1
            {
                guard unlinkat(root, name, 0) == 0 else { throw ChronosDemoFailure.persistence }
            } else if UUID(uuidString: suffix) != nil, info.st_mode & S_IFMT == S_IFDIR {
                let folder = try ChronosJournalBoundary.folder(name, in: root)
                defer { close(folder) }
                let names = try ChronosJournalBoundary.names(folder)
                guard names.isSubset(of: ["prediction.json"]) else { continue }
                if names.contains("prediction.json") {
                    var file = stat()
                    guard fstatat(folder, "prediction.json", &file, AT_SYMLINK_NOFOLLOW) == 0,
                        file.st_mode & S_IFMT == S_IFREG, file.st_nlink == 1
                    else { continue }
                    guard unlinkat(folder, "prediction.json", 0) == 0 else { throw ChronosDemoFailure.persistence }
                }
                guard unlinkat(root, name, AT_REMOVEDIR) == 0 else { throw ChronosDemoFailure.persistence }
            }
        }
    }

    private struct Envelope: Codable {
        let schema: String
        let id: UUID
        let recordID: UUID?
        let payload: Data
        let sha256: String
    }
}
