import CryptoKit
import Foundation

@testable import CoreAIBackend

struct PreparedCopyFixture {
    static let key = PreparedCopyKey(
        namespace: "com.example.samples", osBuild: "27A100", assetHash: String(repeating: "5a", count: 32),
        options: "default")
    static let otherKey = PreparedCopyKey(
        namespace: key.namespace, osBuild: key.osBuild, assetHash: key.assetHash, options: "cpuOnly")
    static let first = PreparedCopyProcess(pid: 101, start: 1_000)
    static let second = PreparedCopyProcess(pid: 102, start: 2_000)
    static let alive = PreparedCopyMark(attempt: "alive", pid: 102, start: 2_000, kind: .normal, stage: .load, at: 20)
    static let dead = PreparedCopyMark(attempt: "dead", pid: 201, start: 3_000, kind: .normal, stage: .load, at: 10)
    static let lateDead = PreparedCopyMark(
        attempt: "late-dead", pid: 202, start: 4_000, kind: .normal, stage: .warmUp, at: 30)
    static let deadRecovery = PreparedCopyMark(
        attempt: "dead-recovery", pid: 203, start: 5_000, kind: .recovery, stage: .warmUp, at: 40)
    static let unknown = PreparedCopyMark(
        attempt: "unknown", pid: 301, start: 6_000, kind: .normal, stage: .load, at: 15)
    static let aliveRecovery = PreparedCopyMark(
        attempt: "alive-recovery", pid: 102, start: 2_000, kind: .recovery, stage: .load, at: 25)
    static let crowd = (0..<8).map { PreparedCopyProcess(pid: 401 + Int32($0), start: 7_000 + Int64($0)) }
    static let liveness: [Int32: PreparedCopyLiveness] = {
        var table: [Int32: PreparedCopyLiveness] = [
            101: .alive, 102: .alive, 201: .dead, 202: .dead, 203: .dead, 301: .unknown,
        ]
        for process in crowd {
            table[process.pid] = .alive
        }
        return table
    }()
    static let now = 50.0
    static let empty = PreparedCopyRecord(marks: [], recoveries: 0, suspects: [], failed: false)

    let root: URL
    let directory: URL

    init() throws {
        root = FileManager.default.temporaryDirectory.appending(path: "PreparedCopyTests-" + UUID().uuidString)
        directory = root.appending(path: "ledger")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    static func record(
        _ marks: [PreparedCopyMark], recoveries: Int = 0, suspects: [PreparedCopyMark] = [], failed: Bool = false
    ) -> PreparedCopyRecord {
        PreparedCopyRecord(marks: marks, recoveries: recoveries, suspects: suspects, failed: failed)
    }

    static func own(_ attempt: String, _ kind: PreparedCopyMark.Kind, _ stage: PreparedCopyMark.Stage)
        -> PreparedCopyMark
    {
        PreparedCopyMark(attempt: attempt, pid: first.pid, start: first.start, kind: kind, stage: stage, at: now)
    }

    static func ledger(
        _ record: PreparedCopyRecord, namespace: String = PreparedCopyFixture.key.namespace,
        osBuild: String = PreparedCopyFixture.key.osBuild, assetHash: String = PreparedCopyFixture.key.assetHash
    ) -> PreparedCopyLedger {
        PreparedCopyLedger(namespace: namespace, osBuild: osBuild, assetHash: assetHash, records: [key.options: record])
    }

    static func base(_ key: PreparedCopyKey) -> String {
        SHA256.hash(
            data: Data((key.namespace + "\n" + key.osBuild + "\n" + key.assetHash + "\n" + key.options).utf8)
        )
        .map { String(format: "%02x", $0) }.joined()
    }

    static func exists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path(percentEncoded: false))
    }

    func recovery(
        _ process: PreparedCopyProcess = PreparedCopyFixture.first, key: PreparedCopyKey = PreparedCopyFixture.key
    ) -> PreparedCopyRecovery {
        let table = Self.liveness
        let now = Self.now
        return PreparedCopyRecovery(
            directory: directory, key: key, process: process, liveness: { table[$0.pid] ?? .unknown }, now: { now })
    }

    func lockIsFree() -> Bool {
        let path = directory.appending(path: Self.base(Self.key) + ".lock").path(percentEncoded: false)
        let descriptor = open(path, O_RDONLY | O_CLOEXEC)
        guard descriptor >= 0 else { return false }
        defer { close(descriptor) }
        return flock(descriptor, LOCK_EX | LOCK_NB) == 0
    }

    func write(_ record: PreparedCopyRecord, key: PreparedCopyKey = PreparedCopyFixture.key) throws {
        let ledger = PreparedCopyLedger(
            namespace: key.namespace, osBuild: key.osBuild, assetHash: key.assetHash, records: [key.options: record])
        try write(ledger: ledger, key: key)
    }

    func write(ledger: PreparedCopyLedger, key: PreparedCopyKey = PreparedCopyFixture.key) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try write(raw: encoder.encode(ledger), key: key)
    }

    func write(raw data: Data, key: PreparedCopyKey = PreparedCopyFixture.key) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try data.write(to: recovery(key: key).file)
    }

    func bytes(key: PreparedCopyKey = PreparedCopyFixture.key) throws -> Data {
        try Data(contentsOf: recovery(key: key).file)
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }
}
