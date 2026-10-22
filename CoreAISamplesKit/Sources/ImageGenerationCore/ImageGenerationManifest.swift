import Foundation

public struct ImageGenerationManifest: Decodable, Sendable {
    public struct FileRecord: Decodable, Equatable, Sendable {
        public let bytes: Int64
        public let sha256: String
    }

    public let format: Int
    public let profile: ImageGenerationProfile
    public let assets: [String: FileRecord]

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
    }
}
