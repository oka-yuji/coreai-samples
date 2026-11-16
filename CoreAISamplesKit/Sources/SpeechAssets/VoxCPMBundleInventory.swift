import Foundation
import SpeechCore

public enum VoxCPMBundleInventory {
    private struct Listing: Decodable {
        struct Entry: Decodable {
            let bytes: Int64
            let path: String
            let sha256: String
        }

        let format: Int
        let count: Int
        let bytes: Int64
        let files: [Entry]
    }

    public static let modelID = "openbmb/VoxCPM2"
    public static let revision = "7936080b41163874f38610f46f438e2c0757f9fd"
    public static let upstreamRevision = "300e4260b509066151d623dc9c0aa922cf82232b"
    public static let manifestPath = "manifest.json"
    public static let tokenizerPath = "tokenizer.json"
    public static let sampleRate = 48_000
    public static let patch = [1, 64, 4]
    public static let assetNames = ["tslm", "ralm", "proj", "locenc", "dit", "vae"]
    public static let functions: [String: Set<String>] = [
        "tslm": ["embed", "prefill", "step"], "ralm": ["prefill", "step"], "proj": ["heads", "fuse", "prefill_fuse"],
        "locenc": ["main"], "dit": ["main"], "vae": ["main"],
    ]

    public static func files(profile: VoxCPMBundleProfile) throws -> [IrodoriBundleInventory.File] {
        guard let url = Bundle.module.url(forResource: "voxcpm2-\(profile.rawValue).inventory", withExtension: "json")
        else { throw SpeechError.invalidAsset("The application's VoxCPM2 inventory is missing.") }
        return try files(inventory: Data(contentsOf: url))
    }

    public static func files(inventory data: Data) throws -> [IrodoriBundleInventory.File] {
        guard let listing = try? JSONDecoder().decode(Listing.self, from: data), listing.format == 1,
            listing.count == listing.files.count, Set(listing.files.map(\.path)).count == listing.files.count,
            listing.files.allSatisfy({ isRelative($0.path) && $0.bytes >= 0 && isDigest($0.sha256) }),
            listing.files.contains(where: { $0.path == manifestPath }), total(listing.files) == listing.bytes
        else { throw SpeechError.invalidAsset("The VoxCPM2 inventory is not valid.") }
        return listing.files.map { IrodoriBundleInventory.File(path: $0.path, bytes: $0.bytes, sha256: $0.sha256) }
            .sorted { $0.path < $1.path }
    }

    public static func manifest(in data: Data) throws -> VoxCPMBundleManifest {
        guard let manifest = try? JSONDecoder().decode(VoxCPMBundleManifest.self, from: data) else {
            throw SpeechError.invalidAsset("The VoxCPM2 manifest could not be read.")
        }
        _ = try manifest.validate()
        return manifest
    }

    public static func profile(in data: Data) throws -> VoxCPMBundleProfile {
        guard let manifest = try? JSONDecoder().decode(VoxCPMBundleManifest.self, from: data) else {
            throw SpeechError.invalidAsset("The VoxCPM2 manifest could not be read.")
        }
        return try manifest.validate()
    }

    private static func isRelative(_ path: String) -> Bool {
        !path.isEmpty
            && !path.split(separator: "/", omittingEmptySubsequences: false).contains {
                $0.isEmpty || $0 == "." || $0 == ".."
            }
    }

    private static func isDigest(_ text: String) -> Bool {
        text.utf8.count == 64 && text.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }

    private static func total(_ entries: [Listing.Entry]) -> Int64? {
        var sum: Int64 = 0
        for entry in entries {
            let (next, overflow) = sum.addingReportingOverflow(entry.bytes)
            guard !overflow else { return nil }
            sum = next
        }
        return sum
    }
}
