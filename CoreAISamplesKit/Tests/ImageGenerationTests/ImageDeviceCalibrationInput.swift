import CryptoKit
import Foundation
import ImageGenerationCore
import ImageGenerationRuntime
import ModelLibraryFeature

struct ImageDeviceCalibrationInput: Decodable {
    struct File: Decodable {
        let path: String
        let sha256: String

        func read(from root: URL, maximum: Int) throws -> Data {
            let data = try ImageDeviceCalibrationInput.read(root.appending(path: path), maximum: maximum)
            guard ImageDeviceCalibrationInput.digest(data) == sha256 else {
                throw ImageExecutionError.invalidResource
            }
            return data
        }
    }

    let format: Int
    let bundlePath: String
    let bundleBytes: Int64
    let budget: ImageMemoryBudget
    let hardwareModel: String
    let osBuild: String
    let prompt: String
    let seed: UInt64
    let noise: File
    let reference: File?
    let trace: Bool

    func prepare(at url: URL) throws -> (ValidatedModelResource, ImageGenerationRequest) {
        guard format == 1, !trace, bundleBytes > 0,
            hardwareModel == (try ImageDeviceMemory.systemValue("hw.machine")),
            osBuild == (try ImageDeviceMemory.systemValue("kern.osversion"))
        else { throw ImageExecutionError.invalidResource }
        let root = url.deletingLastPathComponent()
        let resource = ValidatedModelResource(
            modelID: "qwen-image21-turbo", artifactID: budget.profileID, contract: "qwen-image21-mobile-v2",
            revision: budget.manifestSHA256, upstreamRevision: ImageGenerationProfile.revision,
            directory: URL(filePath: bundlePath, relativeTo: root), sizeBytes: bundleBytes, sourceKind: .localManifest)
        let pixels = try reference.map {
            try ImageGenerationImageReader.read($0.read(from: root, maximum: 16_777_216)).squareReference()
        }
        let raw = try noise.read(from: root, maximum: ImageEuler.latentCount * 4)
        guard raw.count == ImageEuler.latentCount * 4 else { throw ImageExecutionError.invalidResource }
        let values = raw.withUnsafeBytes { bytes in
            (0..<ImageEuler.latentCount).map {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self)))
            }
        }
        let request = ImageGenerationRequest(
            id: UUID(), prompt: prompt, seed: seed, reference: pixels, initialNoise: values)
        try request.validate()
        return (resource, request)
    }

    static func read(_ url: URL, maximum: Int) throws -> Data {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: maximum + 1) ?? Data()
        guard !data.isEmpty, data.count <= maximum else { throw ImageExecutionError.invalidResource }
        return data
    }

    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
