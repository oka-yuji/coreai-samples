import CryptoKit
import Foundation
import ModelLibraryFeature
import Testing

@testable import ModelLibraryStorage

struct ChronosImportFixture {
    let files = StoreFixture()
    let artifact: ModelArtifact
    let model: LibraryModel

    init() throws {
        artifact = try ChronosModelArtifact.make()
        model = try #require(MacDemoCatalog.make().model("chronos2"))
    }

    func store() -> ManagedModelStore {
        files.store(availableCapacity: { _ in 1_000_000_000 })
    }

    func manifest() throws -> Data {
        let url = try #require(
            Bundle.module.url(forResource: "chronos2-manifest", withExtension: "json", subdirectory: "Resources"))
        return try Data(contentsOf: url)
    }

    func sparseSource(at source: URL) throws {
        for file in artifact.files {
            let url = source.appending(path: file.path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: url)
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.truncate(atOffset: UInt64(file.bytes))
        }
        try manifest().write(to: source.appending(path: "chronos2.json"))
    }

    func recordSyntheticInstallation() throws -> ValidatedModelResource {
        let revision = try #require(artifact.sourceRevision)
        let resource = ValidatedModelResource(
            modelID: model.id, artifactID: artifact.id, contract: artifact.contract, revision: revision,
            upstreamRevision: artifact.upstream?.revision,
            directory: files.root.appending(path: "artifacts/\(artifact.id)/\(revision)", directoryHint: .isDirectory),
            sizeBytes: try #require(artifact.totalFileBytes), sourceKind: .localManifest)
        try sparseSource(at: resource.directory)
        let ledger = Ledger(entries: [
            Ledger.Entry(
                modelID: model.id, artifactID: artifact.id, contract: artifact.contract, revision: revision,
                upstreamRevision: resource.upstreamRevision, sizeBytes: resource.sizeBytes,
                installedAt: Date(timeIntervalSince1970: 1_000), generation: 1, sourceKind: .localManifest)
        ])
        try JSONEncoder().encode(ledger).write(to: files.root.appending(path: "ledger.json"))
        return resource
    }

    func damage(_ kind: String, at source: URL) throws -> ManagedModelStoreError {
        let manifest = source.appending(path: "chronos2.json")
        switch kind {
        case "manifest":
            var data = try Data(contentsOf: manifest)
            data[0] ^= 1
            try data.write(to: manifest)
            return .checksumMismatch("chronos2.json")
        case "missing":
            try FileManager.default.removeItem(at: source.appending(path: "NOTICE"))
            return .missingFile("NOTICE")
        case "extra":
            try Data().write(to: source.appending(path: ".DS_Store"))
            return .unexpectedFile(".DS_Store")
        case "symlink":
            let path = source.appending(path: "NOTICE")
            try FileManager.default.removeItem(at: path)
            try FileManager.default.createSymbolicLink(at: path, withDestinationURL: manifest)
            return .symbolicLink
        default:
            let path = source.appending(path: "LICENSE")
            var data = try Data(contentsOf: path)
            data[0] ^= 1
            try data.write(to: path)
            return .checksumMismatch("LICENSE")
        }
    }

    func sourceSummary(_ source: URL) throws -> [String: String] {
        let manager = FileManager.default
        var summary: [String: String] = [:]
        for file in artifact.files {
            let url = source.appending(path: file.path)
            let attributes = try? manager.attributesOfItem(atPath: url.path)
            guard let type = attributes?[.type] as? FileAttributeType else {
                summary[file.path] = "missing"
                continue
            }
            if type == .typeSymbolicLink {
                summary[file.path] = try manager.destinationOfSymbolicLink(atPath: url.path)
            } else if file.path == "Model.aimodel/main.mlirb" {
                summary[file.path] = "\(attributes?[.size] as? NSNumber ?? 0)"
            } else {
                summary[file.path] = sha256(try Data(contentsOf: url))
            }
        }
        if manager.fileExists(atPath: source.appending(path: ".DS_Store").path) { summary[".DS_Store"] = "present" }
        return summary
    }

    func hashes(_ source: URL) throws -> [String: String] {
        try Dictionary(
            uniqueKeysWithValues: artifact.files.map { file in
                let handle = try FileHandle(forReadingFrom: source.appending(path: file.path))
                defer { try? handle.close() }
                var hash = SHA256()
                while try autoreleasepool(invoking: {
                    guard let data = try handle.read(upToCount: 1_048_576), !data.isEmpty else { return false }
                    hash.update(data: data)
                    return true
                }) {}
                return (file.path, hash.finalize().map { String(format: "%02x", $0) }.joined())
            })
    }
}
