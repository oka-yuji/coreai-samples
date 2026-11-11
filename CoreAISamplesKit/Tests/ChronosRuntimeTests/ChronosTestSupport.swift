import ChronosCore
import Foundation
import Synchronization
import Testing

@testable import ChronosRuntime

struct ChronosTestFolder {
    let url: URL

    init() throws {
        url = FileManager.default.temporaryDirectory.resolvingSymlinksInPath()
            .appendingPathComponent("chronos-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    func remove() throws {
        try FileManager.default.removeItem(at: url)
    }

    func clean() {
        do { try remove() } catch { Issue.record("Temporary folder cleanup failed: \(error)") }
    }

    func refreshInventory(_ name: String) throws {
        let identity = ChronosFiles.identity(try Data(contentsOf: url.appendingPathComponent(name)))
        let encoded = try JSONSerialization.jsonObject(with: ChronosJSON.encode(identity))
        try changeJSON("chronos2.json") { object in
            var files = object["files"] as? [String: Any] ?? [:]
            files[name] = encoded
            object["files"] = files
        }
    }

    func writeBundle() throws {
        try FileManager.default.createDirectory(
            at: url.appendingPathComponent("Model.aimodel"), withIntermediateDirectories: false)
        let reference = Self.reference()
        var files: [String: ChronosFileIdentity] = [:]
        for name in ChronosManifest.fileNames {
            let data: Data
            switch name {
            case "reference.json": data = try ChronosJSON.encode(reference)
            case "verification.json": data = try Self.verification(reference)
            default: data = Data(name.utf8)
            }
            try data.write(to: url.appendingPathComponent(name))
            files[name] = ChronosFiles.identity(data)
        }
        let manifest = ChronosManifest(
            format: ChronosContract.format, source: .expected, precision: "float32", abi: .expected,
            quantiles: ChronosContract.quantileLevels, rawContext: .expected, files: files)
        try ChronosJSON.encode(manifest).write(to: url.appendingPathComponent("chronos2.json"))
    }

    func changeJSON(_ name: String, change: (inout [String: Any]) -> Void) throws {
        let path = url.appendingPathComponent(name)
        var object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
        change(&object)
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: path)
    }

    static func reference() -> ChronosReference {
        ChronosReference(
            format: "coreai-samples-chronos2-reference/1",
            identity: ChronosReferenceIdentity(
                source: .expected, model: ChronosFiles.identity(Data("model".utf8)),
                fixtures: ChronosFiles.identity(Data("fixtures".utf8))),
            precision: "float32", abi: .expected, quantiles: ChronosContract.quantileLevels,
            atol: 0.0001, rtol: 0.0001,
            cases: zip(ChronosReference.caseIDs, ChronosReference.rawCounts).enumerated().map { index, item in
                ChronosReference.Case(
                    id: item.0, context: Array(repeating: Float(index), count: item.1),
                    quantiles: [rows(offset: index)])
            })
    }

    static func rows(offset: Int = 0) -> [[Float]] {
        (0..<21).map { row in (0..<16).map { Float(offset * 1000 + row * 16 + $0) } }
    }

    static var identity: ChronosBundleIdentity {
        ChronosBundleIdentity(source: .expected, manifest: ChronosFiles.identity(Data("manifest".utf8)), files: [:])
    }

    private static func verification(_ reference: ChronosReference) throws -> Data {
        let identity = try JSONSerialization.jsonObject(with: ChronosJSON.encode(reference.identity))
        let abi = try JSONSerialization.jsonObject(with: ChronosJSON.encode(ChronosABI.expected))
        return try JSONSerialization.data(
            withJSONObject: [
                "identity": identity, "atol": 0.0001, "rtol": 0.0001, "count": 18, "elements": 6048, "passed": true,
                "cases": ChronosReference.caseIDs.map {
                    ["id": $0, "count": 336, "passed": true, "max_absolute_error": 0]
                },
                "runtime": [
                    "function_names": ["main"], "input_names": ["context"], "output_names": ["quantiles"],
                    "state_names": [],
                ],
                "inspection": [
                    "abi": abi, "compatibility_version": "27.0", "compute_types": ["Float32"],
                    "storage_types": [["name": "Float32", "count": 1]],
                    "metadata": [
                        "author": "Amazon", "license": "Apache-2.0", "producer": "coreai-core 1.0.0b3",
                        "source_revision": ChronosContract.revision,
                    ],
                ],
            ], options: [.sortedKeys])
    }
}

final class ChronosGate: Sendable {
    private struct State {
        var result: Result<Void, any Error>?
        var waiters: [CheckedContinuation<Void, any Error>] = []
    }

    private let state = Mutex(State())

    func wait() async throws {
        try await withCheckedThrowingContinuation { continuation in
            let result = state.withLock { value -> Result<Void, any Error>? in
                if let result = value.result { return result }
                value.waiters.append(continuation)
                return nil
            }
            if let result { continuation.resume(with: result) }
        }
    }

    func finish(_ result: Result<Void, any Error> = .success(())) {
        let waiters = state.withLock { value in
            guard value.result == nil else { return [CheckedContinuation<Void, any Error>]() }
            value.result = result
            let waiters = value.waiters
            value.waiters.removeAll()
            return waiters
        }
        for waiter in waiters { waiter.resume(with: result) }
    }
}

struct ChronosFakePredictor: ChronosPredictor {
    let identity: ChronosBundleIdentity
    var run: @Sendable (ChronosContext) async throws -> ChronosPrediction = { _ in
        try ChronosPrediction(quantiles: ChronosTestFolder.rows())
    }

    func predict(_ context: ChronosContext) async throws -> ChronosPrediction {
        try await run(context)
    }
}
