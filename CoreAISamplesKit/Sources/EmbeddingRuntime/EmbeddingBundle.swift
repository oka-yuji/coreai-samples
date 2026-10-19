import CryptoKit
import EmbeddingCore
import Foundation

public struct EmbeddingBundle: Sendable {
    struct Manifest: Decodable {
        let format: Int
        let revision: String
        let precision: String
        let files: [String: String]
    }

    public let directory: URL
    public let revision: String
    public let precision: String

    public init(directory: URL) throws {
        let root = directory.resolvingSymlinksInPath().standardizedFileURL
        let manifest = try JSONDecoder().decode(
            Manifest.self, from: Data(contentsOf: root.appending(path: "embedding-manifest.json")))
        let tables: [String]
        switch manifest.precision {
        case "fp16-safe-s128-v70": tables = ["token_embeddings.bf16"]
        case "int4-int8-ptq-s128-v70":
            tables = [
                "token_embeddings.int4", "token_embedding_scales.f32", "quantization-quality.json",
                "quantized-reference.json",
            ]
        default: throw EmbeddingError.invalidBundle("Unsupported precision profile")
        }
        guard manifest.format == 1,
            manifest.revision == "914f7f89142e33e77833254d9c9b90c3cef7303b",
            (["tokenizer.json", "tokenizer_config.json", "MODEL_LICENSE.txt"] + tables)
                .allSatisfy({ manifest.files[$0] != nil }),
            manifest.files.keys.contains(where: { $0.hasPrefix("text.aimodel/") }),
            manifest.files.keys.contains(where: { $0.hasPrefix("vision.aimodel/") })
        else { throw EmbeddingError.invalidBundle("Unsupported bundle contract") }
        for (path, expected) in manifest.files.sorted(by: { $0.key < $1.key }) {
            try Task.checkCancellation()
            let file = try Self.containedFile(path, root: root)
            let actual = try Self.digest(file)
            guard actual == expected else { throw EmbeddingError.invalidBundle("Checksum mismatch: " + path) }
        }
        for name in ["text.aimodel", "vision.aimodel"] {
            let asset = root.appending(path: name)
            guard let enumerator = FileManager.default.enumerator(atPath: asset.path)
            else { throw EmbeddingError.invalidBundle("Missing asset: " + name) }
            for case let child as String in enumerator {
                let file = try Self.containedFile(name + "/" + child, root: root)
                if try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
                    let relative = name + "/" + child
                    guard manifest.files[relative] != nil else {
                        throw EmbeddingError.invalidBundle("Unlisted asset file: " + relative)
                    }
                }
            }
        }
        self.directory = root
        revision = manifest.revision
        precision = manifest.precision
    }

    static func containedFile(_ path: String, root: URL) throws -> URL {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.split(separator: "/").contains("..") else {
            throw EmbeddingError.invalidBundle("Invalid file path")
        }
        var componentURL = root
        for component in path.split(separator: "/") {
            componentURL.append(path: String(component))
            let attributes = try? FileManager.default.attributesOfItem(atPath: componentURL.path)
            if attributes?[.type] as? FileAttributeType == .typeSymbolicLink {
                throw EmbeddingError.invalidBundle("Symbolic links are not supported in bundles")
            }
        }
        let file = root.appending(path: path).resolvingSymlinksInPath().standardizedFileURL
        guard file.path.hasPrefix(root.path + "/") else { throw EmbeddingError.invalidBundle("File escapes bundle") }
        return file
    }

    static func digest(_ file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
            try Task.checkCancellation()
            hash.update(data: data)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
