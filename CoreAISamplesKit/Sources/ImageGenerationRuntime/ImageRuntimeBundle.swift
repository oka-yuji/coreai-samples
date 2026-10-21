import CryptoKit
import Foundation
import ImageGenerationCore
import ModelLibraryFeature

struct ImageRuntimeBundle: Sendable {
    let directory: URL
    let manifestSHA256: String
    let manifest: ImageGenerationManifest
    let scaling: ImageLatentScaling

    var profile: ImageGenerationProfile { manifest.profile }
    var tokenizerDirectory: URL { directory.appending(path: "config/processor", directoryHint: .isDirectory) }

    static func load(_ resource: ValidatedModelResource, budget: ImageMemoryBudget) throws -> Self {
        guard resource.sourceKind == .localManifest, resource.contract == "qwen-image21-mobile-v2",
            resource.upstreamRevision == ImageGenerationProfile.revision, resource.directory.isFileURL,
            resource.revision == budget.manifestSHA256
        else { throw ImageExecutionError.invalidResource }
        let root = resource.directory
        let rootValues = try root.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard rootValues.isDirectory == true, rootValues.isSymbolicLink != true else {
            throw ImageExecutionError.invalidResource
        }
        let raw = try smallFile(root.appending(path: "bundle.json"))
        let hash = SHA256.hash(data: raw).map { String(format: "%02x", $0) }.joined()
        guard hash == resource.revision else { throw ImageExecutionError.invalidResource }
        let manifest = try ImageGenerationManifest.decode(raw)
        let acceptance = try decoder().decode(Acceptance.self, from: raw)
        guard acceptance.verified, acceptance.numericVerified, acceptance.qualityVerified,
            manifest.profile.id == budget.profileID,
            Set([
                "LICENSE", "NOTICE.txt", "config/processor/tokenizer.json", "config/processor/tokenizer_config.json",
                "config/vae/config.json",
            ]).isSubset(of: Set(manifest.assets.keys))
        else { throw ImageExecutionError.invalidResource }
        try verifyFiles(root, manifest: manifest, manifestBytes: Int64(raw.count), resourceBytes: resource.sizeBytes)
        for (kind, component) in manifest.profile.components {
            let header = try decoder().decode(
                ComponentHeader.self, from: smallFile(root.appending(path: kind + "/component.json")))
            guard header.format == 2, header.kind == kind, header.conversionComplete, header.stateNames.isEmpty,
                header.profile == manifest.profile, header.inputs == component.inputs,
                header.outputs == component.outputs,
                header.asset == URL(filePath: component.asset).lastPathComponent,
                component.outputs.allSatisfy({ $0.dtype == .float32 }),
                component.inputs.allSatisfy({ [.float32, .int32, .bool].contains($0.dtype) })
            else { throw ImageExecutionError.invalidResource }
        }
        let configuration = try decoder().decode(
            VAEConfiguration.self, from: smallFile(root.appending(path: "config/vae/config.json")))
        let scaling = try ImageLatentScaling(
            mean: configuration.latentsMean, standardDeviation: configuration.latentsStd)
        guard manifest.profile.components["denoiser"]?.computeDtype == manifest.profile.workload.schedulerOutputDtype
        else {
            throw ImageExecutionError.invalidResource
        }
        return Self(directory: root, manifestSHA256: hash, manifest: manifest, scaling: scaling)
    }

    private static func verifyFiles(
        _ root: URL, manifest: ImageGenerationManifest, manifestBytes: Int64, resourceBytes: Int64
    ) throws {
        let expected = Set(manifest.assets.keys).union(["bundle.json"])
        let folders = Set(
            expected.flatMap { name in
                let parts = name.split(separator: "/")
                return parts.indices.dropLast().map { parts[...$0].joined(separator: "/") }
            })
        let base = root.standardizedFileURL.resolvingSymlinksInPath().path + "/"
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey]
        guard let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: Array(keys)) else {
            throw ImageExecutionError.invalidResource
        }
        var found: Set<String> = []
        for case let file as URL in enumerator {
            try Task.checkCancellation()
            let values = try file.resourceValues(forKeys: keys)
            let path = file.standardizedFileURL.resolvingSymlinksInPath().path
            guard values.isSymbolicLink != true, path.hasPrefix(base) else { throw ImageExecutionError.invalidResource }
            let relative = String(path.dropFirst(base.count))
            if values.isDirectory == true {
                guard folders.contains(relative) else { throw ImageExecutionError.invalidResource }
            } else {
                guard values.isRegularFile == true, expected.contains(relative) else {
                    throw ImageExecutionError.invalidResource
                }
                found.insert(relative)
            }
        }
        guard found == expected else { throw ImageExecutionError.invalidResource }
        var total = manifestBytes
        for (name, expected) in manifest.assets {
            let (sum, overflow) = total.addingReportingOverflow(expected.bytes)
            guard !overflow else { throw ImageExecutionError.invalidResource }
            total = sum
            let actual = try fileRecord(root.appending(path: name))
            guard actual.0 == expected.bytes, actual.1 == expected.sha256 else {
                throw ImageExecutionError.invalidResource
            }
        }
        guard total == resourceBytes else { throw ImageExecutionError.invalidResource }
    }

    private static func smallFile(_ url: URL) throws -> Data {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true else {
            throw ImageExecutionError.invalidResource
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var result = Data()
        while try autoreleasepool(invoking: {
            try Task.checkCancellation()
            guard let data = try handle.read(upToCount: 1_048_576), !data.isEmpty else { return false }
            guard result.count + data.count <= 33_554_432 else { throw ImageExecutionError.invalidResource }
            result.append(data)
            return true
        }) {}
        return result
    }

    private static func fileRecord(_ url: URL) throws -> (Int64, String) {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        var count: Int64 = 0
        while try autoreleasepool(invoking: {
            try Task.checkCancellation()
            guard let data = try handle.read(upToCount: 1_048_576), !data.isEmpty else { return false }
            let (next, overflow) = count.addingReportingOverflow(Int64(data.count))
            guard !overflow else { throw ImageExecutionError.invalidResource }
            count = next
            hash.update(data: data)
            return true
        }) {}
        return (count, hash.finalize().map { String(format: "%02x", $0) }.joined())
    }

    private static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }
}

private struct Acceptance: Decodable {
    let verified: Bool
    let numericVerified: Bool
    let qualityVerified: Bool
}

private struct ComponentHeader: Decodable {
    let format: Int
    let kind: String
    let asset: String
    let profile: ImageGenerationProfile
    let conversionComplete: Bool
    let inputs: [ImageTensorContract]
    let outputs: [ImageTensorContract]
    let stateNames: [String]
}

private struct VAEConfiguration: Decodable {
    let latentsMean: [Float]
    let latentsStd: [Float]
}
