import CoreGraphics
import CryptoKit
import Darwin
import Foundation
import ImageDiagnosticFeature
import ImageGenerationCore
import ImageIO
import ModelLibraryFeature
import UniformTypeIdentifiers

struct ImageDiagnosticConfiguration: Codable, Sendable {
    struct Input: Codable, Sendable {
        let path: String
        let sha256: String
    }
    let format: Int
    let runID: String
    let bundlePath: String
    let bundleBytes: Int64
    let budget: ImageMemoryBudget
    let budgetBasis: String
    let budgetEvidence: String
    let hardwareModel: String
    let osBuild: String
    let prompt: String
    let seed: UInt64
    let reference: Input?
    let noise: Input?
    let trace: Bool
    let stop: ImageDiagnosticStop?

    func validate() throws {
        guard format == 1, !runID.isEmpty, runID.utf8.count <= 100,
            runID.utf8.allSatisfy({ (48...57).contains($0) || (97...122).contains($0) || $0 == 45 }),
            !bundlePath.isEmpty, bundleBytes > 0, ["estimate", "measured"].contains(budgetBasis),
            !budgetEvidence.isEmpty, !hardwareModel.isEmpty, !osBuild.isEmpty
        else { throw ImageExecutionError.invalidResource }
        if let stop {
            guard (1...60_000).contains(stop.delayMilliseconds), stop.operation != .release,
                stop.operation != .loading || stop.step == nil,
                stop.component == .denoiser && stop.operation == .inference
                    ? stop.step.map({ (0..<8).contains($0) }) == true : stop.step == nil
            else { throw ImageExecutionError.invalidResource }
        }
        try ImageGenerationRequest(
            id: UUID(uuid: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)), prompt: prompt, seed: seed
        )
        .validate()
    }
}

struct ImageDiagnosticContext: Sendable {
    let configuration: ImageDiagnosticConfiguration
    let resource: ValidatedModelResource
    let request: ImageGenerationRequest
    let directory: URL
}

struct ImageDiagnosticStopRecord: Codable, Sendable {
    let phase: String?
    let secondsSinceDiagnosticStart: Double?
    let secondsSinceRuntimeStart: Double?
}

public actor ImageDiagnosticFiles {
    public init() {}

    func recoveredCleanup(directory: String, attempt: Int) throws {
        let url = URL(filePath: directory)
        let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard !directory.isEmpty, values.isDirectory == true, values.isSymbolicLink != true, attempt > 0 else {
            throw ImageExecutionError.invalidResource
        }
        let data = try JSONSerialization.data(
            withJSONObject: ["format": 1, "released": true, "attempt": attempt], options: .sortedKeys)
        try data.write(to: url.appending(path: "cleanup-retry-\(attempt).json"), options: .withoutOverwriting)
    }

    func prepare(_ configurationURL: URL, requestID: UUID, hardware: String, osBuild: String) throws
        -> ImageDiagnosticContext
    {
        let raw = try read(configurationURL, maximum: 1_048_576)
        let configuration = try JSONDecoder().decode(ImageDiagnosticConfiguration.self, from: raw)
        try configuration.validate()
        guard configuration.hardwareModel == hardware, configuration.osBuild == osBuild else {
            throw ImageExecutionError.invalidResource
        }
        let root = configurationURL.deletingLastPathComponent()
        let resource = ValidatedModelResource(
            modelID: "qwen-image21-turbo", artifactID: configuration.budget.profileID,
            contract: "qwen-image21-mobile-v2",
            revision: configuration.budget.manifestSHA256, upstreamRevision: ImageGenerationProfile.revision,
            directory: URL(filePath: configuration.bundlePath, relativeTo: root), sizeBytes: configuration.bundleBytes,
            sourceKind: .localManifest)
        let reference: ImageGenerationPixels?
        if let input = configuration.reference {
            let data = try inputData(input, root: root, maximum: 16_777_216)
            guard let source = CGImageSourceCreateWithData(data as CFData, nil), CGImageSourceGetCount(source) == 1,
                let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
                let width = properties[kCGImagePropertyPixelWidth] as? Int,
                let height = properties[kCGImagePropertyPixelHeight] as? Int,
                (1...1_024).contains(width), (1...1_024).contains(height),
                let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else { throw ImageExecutionError.invalidResource }
            reference = try ImageGenerationImageReader.read(image)
                .oriented(properties[kCGImagePropertyOrientation] as? Int ?? 1).squareReference()
        } else {
            reference = nil
        }
        let noise: [Float]?
        if let input = configuration.noise {
            let data = try inputData(input, root: root, maximum: ImageEuler.latentCount * 4)
            guard data.count == ImageEuler.latentCount * 4 else { throw ImageExecutionError.invalidResource }
            noise = data.withUnsafeBytes { bytes in
                (0..<ImageEuler.latentCount).map {
                    Float(
                        bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self)))
                }
            }
        } else {
            noise = nil
        }
        let request = ImageGenerationRequest(
            id: requestID, prompt: configuration.prompt, seed: configuration.seed,
            reference: reference, initialNoise: noise)
        try request.validate()
        let directory = root.appending(path: "image-check-" + configuration.runID)
        guard mkdir(directory.path, 0o700) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        try raw.write(to: directory.appending(path: "request.json"), options: .withoutOverwriting)
        return ImageDiagnosticContext(
            configuration: configuration, resource: resource, request: request, directory: directory)
    }

    func finish(
        _ context: ImageDiagnosticContext, result: ImageGenerationResult?, report: ImageExecutionReport?,
        error: String?, cleanupError: String?, cancelled: Bool, stop: ImageDiagnosticStopRecord? = nil
    ) throws -> ImageDiagnosticOutcome {
        if let report {
            guard report.requestID == context.request.id, report.status != .running,
                report.profileID == context.configuration.budget.profileID,
                report.manifestSHA256 == context.configuration.budget.manifestSHA256
            else { throw ImageExecutionError.invalidResource }
        }
        if let result {
            guard result.requestID == context.request.id, result.report.requestID == context.request.id,
                result.report.status == .completed,
                result.report.profileID == context.configuration.budget.profileID,
                result.report.manifestSHA256 == context.configuration.budget.manifestSHA256,
                result.pixels.width == 512, result.pixels.height == 512
            else { throw ImageExecutionError.invalidResource }
            try Data(result.pixels.rgba).write(
                to: context.directory.appending(path: "output.rgba"), options: .withoutOverwriting)
            try png(result.pixels).write(
                to: context.directory.appending(path: "output.png"), options: .withoutOverwriting)
        }
        let status: ImageExecutionReport.Status =
            cleanupError != nil
            ? .releaseUnconfirmed
            : cancelled ? .cancelled : error != nil ? .failed : report?.status ?? .failed
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let report {
            try encoder.encode(report).write(
                to: context.directory.appending(path: "execution.json"), options: .withoutOverwriting)
        }
        struct Completion: Encodable {
            let format = 1
            let status: ImageExecutionReport.Status
            let error: String?
            let cleanupError: String?
            let budgetBasis: String
            let budgetEvidence: String
            let hardwareModel: String
            let osBuild: String
            let outputSHA256: String?
            let stopRequest: ImageDiagnosticStopRecord?
        }
        let configuration = context.configuration
        let completion = Completion(
            status: status, error: error, cleanupError: cleanupError,
            budgetBasis: configuration.budgetBasis, budgetEvidence: configuration.budgetEvidence,
            hardwareModel: configuration.hardwareModel, osBuild: configuration.osBuild,
            outputSHA256: result.map { digest(Data($0.pixels.rgba)) }, stopRequest: stop)
        try encoder.encode(completion).write(
            to: context.directory.appending(path: "completion.json"), options: .withoutOverwriting)
        let failures = [error.map { "Execution: " + $0 }, cleanupError.map { "Cleanup: " + $0 }].compactMap { $0 }
        return ImageDiagnosticOutcome(
            status: status, directory: context.directory.path,
            message: failures.isEmpty ? status.rawValue : failures.joined(separator: "; "))
    }

    private func inputData(_ input: ImageDiagnosticConfiguration.Input, root: URL, maximum: Int) throws -> Data {
        guard !input.path.isEmpty, input.sha256.count == 64 else { throw ImageExecutionError.invalidResource }
        let data = try read(URL(filePath: input.path, relativeTo: root), maximum: maximum)
        guard digest(data) == input.sha256 else { throw ImageExecutionError.invalidResource }
        return data
    }

    private func read(_ url: URL, maximum: Int) throws -> Data {
        guard url.isFileURL else { throw ImageExecutionError.invalidResource }
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true,
            let size = values.fileSize, size <= maximum
        else { throw ImageExecutionError.invalidResource }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: maximum + 1) ?? Data()
        guard data.count == size, data.count <= maximum else { throw ImageExecutionError.invalidResource }
        return data
    }

    private func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

    private func png(_ pixels: ImageGenerationPixels) throws -> Data {
        guard let provider = CGDataProvider(data: Data(pixels.rgba) as CFData),
            let color = CGColorSpace(name: CGColorSpace.sRGB),
            let image = CGImage(
                width: pixels.width, height: pixels.height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: pixels.width * 4, space: color,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { throw ImageExecutionError.invalidResource }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw ImageExecutionError.invalidResource
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw ImageExecutionError.invalidResource }
        return data as Data
    }
}
