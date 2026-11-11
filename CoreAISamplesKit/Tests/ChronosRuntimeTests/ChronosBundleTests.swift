import ChronosCore
import Darwin
import Foundation
import Testing

@testable import ChronosRuntime

@Suite(.timeLimit(.minutes(1)))
struct ChronosBundleTests {
    @Test("A complete bundle validates all files and records the actual manifest identity")
    func validBundle() async throws {
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        try folder.writeBundle()
        let bundle = try await ChronosBundle.open(folder.url)
        #expect(bundle.identity.source == .expected)
        #expect(bundle.identity.files.count == 8)
        #expect(bundle.reference.cases.count == 18)
        #expect(
            bundle.identity.manifest
                == ChronosFiles.identity(try Data(contentsOf: folder.url.appendingPathComponent("chronos2.json"))))
    }

    @Test("A bundle remains usable below an operating system or user selected parent alias")
    func parentAlias() async throws {
        let folder = try ChronosTestFolder()
        let outside = try ChronosTestFolder()
        defer {
            folder.clean()
            outside.clean()
        }
        try folder.writeBundle()
        let alias = outside.url.appendingPathComponent("parent")
        try FileManager.default.createSymbolicLink(
            at: alias, withDestinationURL: folder.url.deletingLastPathComponent())
        let bundle = try await ChronosBundle.open(alias.appendingPathComponent(folder.url.lastPathComponent))
        #expect(bundle.identity.source == .expected)
    }

    @Test(
        "Missing extra linked aliased and nonregular bundle entries are rejected",
        arguments: [
            "missing", "extra", "empty-directory", "symlink", "directory-link", "hardlink", "file-directory", "fifo",
            "root-link",
        ])
    func fileFailures(kind: String) async throws {
        let folder = try ChronosTestFolder()
        let outside = try ChronosTestFolder()
        defer {
            folder.clean()
            outside.clean()
        }
        try folder.writeBundle()
        let manager = FileManager.default
        let file = folder.url.appendingPathComponent("NOTICE")
        var bundleURL = folder.url
        switch kind {
        case "missing": try manager.removeItem(at: file)
        case "extra": try Data([1]).write(to: folder.url.appendingPathComponent("extra"))
        case "empty-directory":
            try manager.createDirectory(
                at: folder.url.appendingPathComponent("extra"), withIntermediateDirectories: false)
        case "symlink":
            try manager.moveItem(at: file, to: outside.url.appendingPathComponent("NOTICE"))
            try manager.createSymbolicLink(at: file, withDestinationURL: outside.url.appendingPathComponent("NOTICE"))
        case "directory-link":
            let asset = folder.url.appendingPathComponent("Model.aimodel")
            let moved = outside.url.appendingPathComponent("Model.aimodel")
            try manager.moveItem(at: asset, to: moved)
            try manager.createSymbolicLink(at: asset, withDestinationURL: moved)
        case "hardlink": try manager.linkItem(at: file, to: outside.url.appendingPathComponent("NOTICE"))
        case "file-directory":
            try manager.removeItem(at: file)
            try manager.createDirectory(at: file, withIntermediateDirectories: false)
        case "fifo":
            try manager.removeItem(at: file)
            try #require(mkfifo(file.path, 0o600) == 0)
        case "root-link":
            bundleURL = outside.url.appendingPathComponent("linked")
            try manager.createSymbolicLink(at: bundleURL, withDestinationURL: folder.url)
        default: Issue.record("Unknown fixture mutation")
        }
        await #expect(throws: (any Error).self) { try await ChronosBundle.open(bundleURL) }
    }

    @Test(
        "Manifest path aliases and escapes are rejected",
        arguments: [
            "../NOTICE", "/NOTICE", "./NOTICE", "Model.aimodel//main.hash", "Model.aimodel/../NOTICE",
            "notice", "Model.aimodel\\main.hash", "file:NOTICE", "NOTICE\0extra",
        ])
    func paths(path: String) async throws {
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        try folder.writeBundle()
        try folder.changeJSON("chronos2.json") { object in
            var files = object["files"] as? [String: Any] ?? [:]
            files[path] = files.removeValue(forKey: "NOTICE")
            object["files"] = files
        }
        await #expect(throws: (any Error).self) { try await ChronosBundle.open(folder.url) }
    }

    @Test(
        "Manifest source precision ABI quantiles raw bounds and identity types are validated",
        arguments: [
            "format", "source", "precision", "quantiles", "raw_context", "abi", "hash", "size", "size-type", "unknown",
            "key-alias",
        ])
    func metadata(field: String) async throws {
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        try folder.writeBundle()
        try folder.changeJSON("chronos2.json") { object in
            switch field {
            case "format": object["format"] = "coreai-samples-chronos2/2"
            case "source": object["source"] = ["model_id": ChronosContract.source, "revision": "wrong"]
            case "precision": object["precision"] = "float16"
            case "unknown": object["extra"] = "ignored"
            case "key-alias": object["rawContext"] = object.removeValue(forKey: "raw_context")
            case "quantiles": object["quantiles"] = Array(ChronosContract.quantileLevels.reversed())
            case "raw_context":
                object["raw_context"] = [
                    "min_length": 0, "max_length": 8192, "missing": "NaN", "padding": "NaN on the left",
                ]
            case "abi":
                var abi = object["abi"] as? [String: Any] ?? [:]
                abi["state_names"] = ["state"]
                object["abi"] = abi
            default:
                var files = object["files"] as? [String: Any] ?? [:]
                var entry = files["NOTICE"] as? [String: Any] ?? [:]
                if field == "hash" { entry["sha256"] = String(repeating: "z", count: 64) }
                if field == "size" { entry["bytes"] = -1 }
                if field == "size-type" { entry["bytes"] = true }
                files["NOTICE"] = entry
                object["files"] = files
            }
        }
        await #expect(throws: (any Error).self) { try await ChronosBundle.open(folder.url) }
    }

    @Test("Hashing validates bytes beyond the first chunk and rejects size changes")
    func streamedHash() async throws {
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        try folder.writeBundle()
        let url = folder.url.appendingPathComponent("Model.aimodel/main.mlirb")
        var data = Data(repeating: 37, count: ChronosFiles.chunkSize * 2 + 19)
        try data.write(to: url)
        try folder.refreshInventory("Model.aimodel/main.mlirb")
        _ = try await ChronosBundle.open(folder.url)
        data[data.count - 1] = 38
        try data.write(to: url)
        await #expect(throws: (any Error).self) { try await ChronosBundle.open(folder.url) }
        data.append(0)
        try data.write(to: url)
        await #expect(throws: (any Error).self) { try await ChronosBundle.open(folder.url) }
    }

    @Test("Bounded JSON reads reject oversized manifests before decoding")
    func boundedManifest() async throws {
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        try folder.writeBundle()
        try Data(repeating: 32, count: ChronosFiles.manifestLimit + 1).write(
            to: folder.url.appendingPathComponent("chronos2.json"))
        await #expect(throws: (any Error).self) { try await ChronosBundle.open(folder.url) }
    }

    @Test(
        "Reference and prior verification metadata are checked even when their hashes match",
        arguments: [
            "tolerance", "case-order", "case-count", "output-shape", "verification-count", "verification-source",
            "inspection-type",
        ])
    func referenceMetadata(kind: String) async throws {
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        try folder.writeBundle()
        let name = kind.hasPrefix("verification") || kind == "inspection-type" ? "verification.json" : "reference.json"
        try folder.changeJSON(name) { object in
            switch kind {
            case "tolerance": object["atol"] = 1
            case "case-order":
                let cases = object["cases"] as? [[String: Any]] ?? []
                object["cases"] = Array(cases.reversed())
            case "case-count":
                let cases = object["cases"] as? [[String: Any]] ?? []
                object["cases"] = Array(cases.dropLast())
            case "output-shape":
                var cases = object["cases"] as? [[String: Any]] ?? []
                if !cases.isEmpty { cases[0]["quantiles"] = [[[0.0]]] }
                object["cases"] = cases
            case "verification-count": object["elements"] = 6047
            case "verification-source": object["identity"] = [:]
            case "inspection-type":
                var inspection = object["inspection"] as? [String: Any] ?? [:]
                inspection["compute_types"] = ["Float16"]
                object["inspection"] = inspection
            default: break
            }
        }
        try folder.refreshInventory(name)
        await #expect(throws: (any Error).self) { try await ChronosBundle.open(folder.url) }
    }
}
