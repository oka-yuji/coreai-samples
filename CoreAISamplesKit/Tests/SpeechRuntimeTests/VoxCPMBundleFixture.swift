import CryptoKit
import Foundation
import SpeechCore
import Testing

@testable import SpeechAssets
@testable import SpeechRuntime

struct VoxCPMBundleFixture: Sendable {
    static let assets = ["tslm", "ralm", "proj", "locenc", "dit", "vae"]

    let base = FileManager.default.temporaryDirectory.appending(
        path: "VoxCPMBundleTests-" + UUID().uuidString, directoryHint: .isDirectory)
    let manifest: Data
    let contents: [String: Data]
    let key: String
    var source: URL { base.appending(path: "source", directoryHint: .isDirectory) }
    var copies: URL { base.appending(path: "copies", directoryHint: .isDirectory) }
    var copy: URL { copies.appending(path: key, directoryHint: .isDirectory) }

    init() throws {
        manifest = try Self.manifest(.fp32)
        var contents = [
            "manifest.json": manifest, "tokenizer.json": Data("tokenizer".utf8), "LICENSE": Data("license".utf8),
            "NOTICE": Data("notice".utf8),
        ]
        for asset in Self.assets {
            for file in ["main.mlirb", "main.hash", "metadata.json"] {
                contents["artifacts/\(asset)-fp32.aimodel/\(file)"] = Data("\(asset)/\(file)".utf8)
            }
        }
        self.contents = contents
        key = ValidatedSpeechBundle.verifiedKey(
            manifest: manifest, files: try VoxCPMBundleInventory.files(inventory: Self.inventory(contents)),
            names: .voxcpm2)
        for (path, data) in contents { try write(data, to: path) }
    }

    static func manifest(_ profile: VoxCPMBundleProfile) throws -> Data {
        let url = try #require(
            Bundle.module.url(
                forResource: "voxcpm2-\(profile.rawValue)-manifest", withExtension: "json", subdirectory: "Resources"))
        return try Data(contentsOf: url)
    }

    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    func inventory() throws -> Data {
        try Self.inventory(contents)
    }

    static func inventory(_ contents: [String: Data]) throws -> Data {
        let files = contents.keys.sorted().map { path in
            ["path": path, "bytes": contents[path]?.count ?? 0, "sha256": digest(contents[path] ?? Data())]
        }
        let total = contents.values.reduce(0) { $0 + $1.count }
        return try JSONSerialization.data(withJSONObject: [
            "format": 1, "count": files.count, "bytes": total, "files": files,
        ])
    }

    func layout() throws -> @Sendable (Data) throws -> SpeechBundleLayout {
        let files = try VoxCPMBundleInventory.files(inventory: inventory())
        return { try SpeechBundleLayout.voxcpm2(manifest: $0, files: files) }
    }

    func open() throws -> ValidatedSpeechBundle {
        try ValidatedSpeechBundle.open(source, copyingInto: copies, layout: layout(), names: .voxcpm2)
    }

    func write(_ data: Data, to path: String) throws {
        let url = source.appending(path: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    }

    func apply(_ defect: VoxCPMBundleDefect) throws {
        let files = FileManager.default
        switch defect {
        case .changedTokenizer:
            try write(Data("tokenizex".utf8), to: "tokenizer.json")
        case .changedAsset:
            try write(Data("dit/main.mlirX".utf8), to: "artifacts/dit-fp32.aimodel/main.mlirb")
        case .missingFile:
            try files.removeItem(at: source.appending(path: "artifacts/vae-fp32.aimodel/metadata.json"))
        case .symlinkedFile:
            let outside = base.appending(path: "outside-tokenizer.json")
            try files.moveItem(at: source.appending(path: "tokenizer.json"), to: outside)
            try files.createSymbolicLink(at: source.appending(path: "tokenizer.json"), withDestinationURL: outside)
        case .symlinkedFolder:
            let outside = base.appending(path: "outside.aimodel", directoryHint: .isDirectory)
            try files.moveItem(at: source.appending(path: "artifacts/locenc-fp32.aimodel"), to: outside)
            try files.createSymbolicLink(
                at: source.appending(path: "artifacts/locenc-fp32.aimodel"), withDestinationURL: outside)
        }
    }

    func copyFolders() throws -> Set<String> {
        guard FileManager.default.fileExists(atPath: copies.path) else { return [] }
        return Set(try FileManager.default.contentsOfDirectory(atPath: copies.path)).subtracting([".lock"])
    }

    func copiedFiles() throws -> Set<String> {
        guard FileManager.default.fileExists(atPath: copy.path) else { return [] }
        var result: Set<String> = []
        for path in try FileManager.default.subpathsOfDirectory(atPath: copy.path) {
            var status = stat()
            if lstat(copy.appending(path: path).path, &status) == 0, status.st_mode & S_IFMT == S_IFREG {
                result.insert(path)
            }
        }
        return result
    }

    func remove() { try? FileManager.default.removeItem(at: base) }
}
