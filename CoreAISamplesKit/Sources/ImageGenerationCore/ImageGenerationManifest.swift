import Foundation

public struct ImageGenerationManifest: Decodable, Sendable {
    public struct FileRecord: Decodable, Equatable, Sendable {
        public let bytes: Int64
        public let sha256: String
    }

    public struct CompiledAsset: Decodable, Equatable, Sendable {
        public let asset: String
        public let architecture: String
        public let sourceSha256: String

        public init(asset: String, architecture: String, sourceSha256: String) {
            self.asset = asset
            self.architecture = architecture
            self.sourceSha256 = sourceSha256
        }

        public func validateMetadata(_ data: Data) throws {
            guard data.count <= 1_048_576 else {
                throw ImageGenerationError.invalidProfile("compiled metadata size")
            }
            let metadata = try JSONDecoder().decode(CompiledMetadata.self, from: data)
            guard metadata.sourceHash.lowercased() == sourceSha256 else {
                throw ImageGenerationError.invalidProfile("compiled metadata source")
            }
        }
    }

    public let format: Int
    public let profile: ImageGenerationProfile
    public let assets: [String: FileRecord]
    public let compiledAssets: [String: CompiledAsset]?

    public static func decode(_ rawBytes: Data) throws -> Self {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let manifest = try decoder.decode(Self.self, from: rawBytes)
        try manifest.validate()
        return manifest
    }

    public func validate() throws {
        try profile.validate()
        guard format == 2, !assets.isEmpty, assets["bundle.json"] == nil,
            assets.allSatisfy({
                imageRelativePathIsValid($0.key) && $0.value.bytes >= 0 && imageDigestIsValid($0.value.sha256)
            })
        else { throw ImageGenerationError.invalidProfile("bundle inventory") }
        var total: Int64 = 0
        for file in assets.values {
            let (sum, overflow) = total.addingReportingOverflow(file.bytes)
            guard !overflow else { throw ImageGenerationError.invalidProfile("inventory size") }
            total = sum
        }
        for component in profile.components.values {
            guard assets.keys.contains(where: { $0.hasPrefix(component.asset + "/") }) else {
                throw ImageGenerationError.invalidProfile("missing component asset")
            }
        }
        let compiled = compiledAssets ?? [:]
        guard Set(compiled.keys).isSubset(of: ImageGenerationProfile.componentNames),
            Set(compiled.values.map(\.asset)).count == compiled.count
        else { throw ImageGenerationError.invalidProfile("compiled components") }
        for (kind, value) in compiled {
            guard let original = profile.components[kind], imageRelativePathIsValid(value.asset),
                value.asset.hasSuffix(".aimodelc"), !value.architecture.isEmpty, value.architecture.utf8.count <= 32,
                value.architecture.utf8.allSatisfy({ (48...57).contains($0) || (97...122).contains($0) }),
                imageDigestIsValid(value.sourceSha256),
                assets[original.asset + "/main.mlirb"]?.sha256 == value.sourceSha256,
                let metadata = assets[value.asset + "/metadata.json"], metadata.bytes <= 1_048_576
            else { throw ImageGenerationError.invalidProfile("compiled source identity") }
        }
    }
}

private struct CompiledMetadata: Decodable {
    let sourceHash: String
}
