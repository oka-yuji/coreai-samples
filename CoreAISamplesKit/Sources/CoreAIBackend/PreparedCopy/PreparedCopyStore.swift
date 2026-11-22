import CryptoKit
import Darwin
import Foundation

struct PreparedCopyStore: Sendable {
    let directory: URL
    let key: PreparedCopyKey

    var file: URL {
        sibling("json")
    }

    private var base: String {
        let text = key.namespace + "\n" + key.osBuild + "\n" + key.assetHash
        return SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private var blank: PreparedCopyRecord {
        PreparedCopyRecord(marks: [], recoveries: 0, suspects: [], failed: false)
    }

    func record() -> PreparedCopyRecord {
        read().records[key.options] ?? blank
    }

    func locked<Value>(_ body: () throws -> Value) throws -> Value {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let descriptor = try PreparedCopyDescriptor.make(sibling("lock"))
        defer { close(descriptor) }
        try PreparedCopyDescriptor.lock(descriptor, LOCK_EX)
        return try body()
    }

    func modify<Value>(_ change: (inout PreparedCopyRecord) throws -> Value) throws -> Value {
        var ledger = read()
        var record = ledger.records[key.options] ?? blank
        let value = try change(&record)
        ledger.records[key.options] = record
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(ledger).write(to: file, options: .atomic)
        return value
    }

    func use(exclusive: Bool) throws -> PreparedCopyUse? {
        try PreparedCopyUse.take(sibling("use"), exclusive: exclusive)
    }

    func removeLedger() throws {
        guard FileManager.default.fileExists(atPath: file.path(percentEncoded: false)) else { return }
        try FileManager.default.removeItem(at: file)
    }

    private func sibling(_ suffix: String) -> URL {
        directory.appending(path: base + "." + suffix)
    }

    private func read() -> PreparedCopyLedger {
        let empty = PreparedCopyLedger(
            namespace: key.namespace, osBuild: key.osBuild, assetHash: key.assetHash, records: [:])
        guard let data = try? Data(contentsOf: file),
            let ledger = try? JSONDecoder().decode(PreparedCopyLedger.self, from: data),
            ledger.namespace == key.namespace, ledger.osBuild == key.osBuild, ledger.assetHash == key.assetHash
        else { return empty }
        return ledger
    }
}
