import CryptoKit
import Foundation
import ModelLibraryFeature
import SpeechAssets
import Testing

@testable import ModelLibraryStorage

struct VoxCPMImportFixture {
    let files = StoreFixture()
    let contents: [String: Data]
    let inventory: Data
    let artifact: ModelArtifact
    let model: LibraryModel
    var source: URL { files.base.appending(path: "source", directoryHint: .isDirectory) }
    var replacement: URL { files.base.appending(path: "replacement", directoryHint: .isDirectory) }

    init() throws {
        contents = try Self.contents(VoxCPMModelArtifact.profile)
        inventory = try Self.inventory(contents)
        artifact = try VoxCPMModelArtifact.make(inventory: inventory)
        model = Self.model(artifact)
    }

    static func model(_ artifact: ModelArtifact) -> LibraryModel {
        LibraryModel(
            id: "voxcpm2", name: "VoxCPM2", category: "Speech Synthesis", artifacts: [artifact],
            defaultArtifactID: artifact.id, platforms: [.macOS], demos: [])
    }

    static func manifest(_ profile: VoxCPMBundleProfile) throws -> Data {
        let url = try #require(
            Bundle.module.url(
                forResource: "voxcpm2-\(profile.rawValue)-manifest", withExtension: "json", subdirectory: "Resources"))
        return try Data(contentsOf: url)
    }

    static func contents(_ profile: VoxCPMBundleProfile) throws -> [String: Data] {
        var contents = [
            VoxCPMBundleInventory.manifestPath: try manifest(profile),
            VoxCPMBundleInventory.tokenizerPath: Data("tokenizer".utf8), "LICENSE": Data("license".utf8),
            "NOTICE": Data("notice".utf8),
        ]
        for (asset, precision) in profile.precisions {
            for file in ["main.mlirb", "main.hash", "metadata.json"] {
                contents["artifacts/\(asset)-\(precision).aimodel/\(file)"] = Data("\(asset)-\(precision)/\(file)".utf8)
            }
        }
        return contents
    }

    static func inventory(_ contents: [String: Data]) throws -> Data {
        let files: [[String: Any]] = contents.keys.sorted().map { path in
            let data = contents[path] ?? Data()
            return ["path": path, "bytes": data.count, "sha256": sha256(data)]
        }
        return try JSONSerialization.data(withJSONObject: [
            "format": 1, "count": files.count, "bytes": contents.values.reduce(0) { $0 + $1.count }, "files": files,
        ])
    }

    func store() -> ManagedModelStore {
        ManagedModelStore(
            root: files.root, removeMigratedSources: false, clock: { Date(timeIntervalSince1970: 1_000) },
            writeLedger: { try $0.write(to: $1, options: .atomic) }, availableCapacity: { _ in .max },
            voxcpmInventory: inventory)
    }

    func install(into store: ManagedModelStore) async throws -> ValidatedModelResource {
        try files.write(contents, to: source)
        let staging = try await store.copyForImport(from: source, model: model, artifact: artifact) { _ in }
        return try await store.promote(staging: staging, model: model, artifact: artifact)
    }

    func ledger() throws -> Data {
        try Data(contentsOf: files.root.appending(path: "ledger.json"))
    }

    func stagingIsEmpty() -> Bool {
        let area = files.root.appending(path: "staging/\(artifact.id)/\(artifact.sourceRevision ?? "")")
        return ((try? FileManager.default.contentsOfDirectory(atPath: area.path)) ?? []).isEmpty
    }

    func damage(_ kind: String, at folder: URL) throws -> ManagedModelStoreError {
        let paths = contents.keys.sorted()
        let manager = FileManager.default
        switch kind {
        case "altered":
            let path = try #require(paths.first { $0.hasSuffix("/main.hash") })
            var data = try Data(contentsOf: folder.appending(path: path))
            data[0] ^= 1
            try data.write(to: folder.appending(path: path))
            return .checksumMismatch(path)
        case "missing":
            let path = try #require(paths.last { $0.hasSuffix("/metadata.json") })
            try manager.removeItem(at: folder.appending(path: path))
            return .missingFile(path)
        case "extra":
            try Data().write(to: folder.appending(path: ".DS_Store"))
            return .unexpectedFile(".DS_Store")
        default:
            let link = folder.appending(path: VoxCPMBundleInventory.tokenizerPath)
            let outside = files.base.appending(path: "outside-\(UUID().uuidString).json")
            try manager.moveItem(at: link, to: outside)
            try manager.createSymbolicLink(at: link, withDestinationURL: outside)
            return .symbolicLink
        }
    }

    func digests(_ folder: URL) throws -> [String: String] {
        var result: [String: String] = [:]
        let entries = try #require(FileManager.default.enumerator(atPath: folder.path))
        while let path = entries.nextObject() as? String {
            switch entries.fileAttributes?[.type] as? FileAttributeType {
            case .typeRegular?: result[path] = try digest(folder.appending(path: path))
            case .typeSymbolicLink?: result[path] = "link"
            default: result[path] = "directory"
            }
        }
        return result
    }

    private func digest(_ file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hash = SHA256()
        while try autoreleasepool(invoking: {
            guard let data = try handle.read(upToCount: 8_388_608), !data.isEmpty else { return false }
            hash.update(data: data)
            return true
        }) {}
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
