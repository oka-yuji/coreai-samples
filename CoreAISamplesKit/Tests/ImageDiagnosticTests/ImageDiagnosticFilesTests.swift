import CryptoKit
import Foundation
import ImageGenerationCore
import Testing

@testable import ImageGenerationRuntime

struct ImageDiagnosticFilesTests {
    @Test("Pinned noise bytes retain their float bits and an existing diagnostic output is never overwritten")
    func inputsAndOutput() async throws {
        let root = try diagnosticTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let values = (0..<ImageEuler.latentCount).map { Float(($0 % 127) - 63) / 127 }
        let raw = values.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) }
        try raw.write(to: root.appending(path: "noise.f32"))
        let specification = try diagnosticSpecification(
            root, noise: ["path": "noise.f32", "sha256": diagnosticDigest(raw)])
        let files = ImageDiagnosticFiles()
        let id = UUID()
        let context = try await files.prepare(
            specification, requestID: id, hardware: "fixture-device", osBuild: "fixture-os")
        #expect(context.request.initialNoise?.map(\.bitPattern) == values.map(\.bitPattern))
        #expect(context.request.seed == UInt64(Int64.max))
        #expect(context.resource.revision == String(repeating: "a", count: 64))
        let copied = try Data(contentsOf: context.directory.appending(path: "request.json"))
        #expect(try copied == Data(contentsOf: specification))
        await #expect(throws: (any Error).self) {
            try await files.prepare(specification, requestID: UUID(), hardware: "fixture-device", osBuild: "fixture-os")
        }
        #expect(try Data(contentsOf: root.appending(path: "noise.f32")) == raw)
        #expect(try Data(contentsOf: context.directory.appending(path: "request.json")) == copied)
    }

    @Test("Wrong input hashes device identities and nonfinite noise are rejected before output creation")
    func inputRejections() async throws {
        let root = try diagnosticTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let files = ImageDiagnosticFiles()
        var raw = Data(repeating: 0, count: ImageEuler.latentCount * 4)
        try raw.write(to: root.appending(path: "noise.f32"))
        let specification = try diagnosticSpecification(
            root, noise: ["path": "noise.f32", "sha256": String(repeating: "b", count: 64)])
        await #expect(throws: ImageExecutionError.invalidResource) {
            try await files.prepare(specification, requestID: UUID(), hardware: "fixture-device", osBuild: "fixture-os")
        }
        raw.replaceSubrange(0..<4, with: [0, 0, 128, 127])
        try raw.write(to: root.appending(path: "noise.f32"))
        _ = try diagnosticSpecification(root, noise: ["path": "noise.f32", "sha256": diagnosticDigest(raw)])
        await #expect(throws: ImageGenerationError.nonFinite) {
            try await files.prepare(specification, requestID: UUID(), hardware: "fixture-device", osBuild: "fixture-os")
        }
        await #expect(throws: ImageExecutionError.invalidResource) {
            try await files.prepare(specification, requestID: UUID(), hardware: "other-device", osBuild: "fixture-os")
        }
        #expect(!FileManager.default.fileExists(atPath: root.appending(path: "image-check-fixture").path))
    }

    @Test("Cancellation before a native report is recorded as cancelled and does not produce an image")
    func earlyCancellation() async throws {
        let root = try diagnosticTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let files = ImageDiagnosticFiles()
        let specification = try diagnosticSpecification(root)
        let context = try await files.prepare(
            specification, requestID: UUID(), hardware: "fixture-device", osBuild: "fixture-os")
        let outcome = try await files.finish(
            context, result: nil, report: nil, error: "cancelled", cleanupError: nil, cancelled: true)
        #expect(outcome.status == .cancelled)
        #expect(!FileManager.default.fileExists(atPath: context.directory.appending(path: "output.png").path))
        let completion = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: context.directory.appending(path: "completion.json")))
                as? [String: Any])
        #expect(completion["status"] as? String == "cancelled")
        #expect(completion["budgetBasis"] as? String == "estimate")
        await #expect(throws: (any Error).self) {
            try await files.finish(context, result: nil, report: nil, error: nil, cleanupError: nil, cancelled: false)
        }
    }

    @Test("Invalid cancellation plans and path traversal in run identifiers are refused")
    func invalidSpecification() throws {
        let root = try diagnosticTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let path = try diagnosticSpecification(root)
        let initial = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
        var value = initial
        value["runID"] = "../escape"
        let invalidID = try JSONDecoder().decode(
            ImageDiagnosticConfiguration.self, from: JSONSerialization.data(withJSONObject: value))
        #expect(throws: ImageExecutionError.invalidResource) { try invalidID.validate() }
        value = initial
        value["stop"] = ["component": "denoiser", "operation": "inference", "step": 8, "delayMilliseconds": 25]
        let invalidStop = try JSONDecoder().decode(
            ImageDiagnosticConfiguration.self, from: JSONSerialization.data(withJSONObject: value))
        #expect(throws: ImageExecutionError.invalidResource) { try invalidStop.validate() }
    }

    @Test("A report from another request cannot be published as this diagnostic completion")
    func wrongReport() async throws {
        let root = try diagnosticTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let files = ImageDiagnosticFiles()
        let path = try diagnosticSpecification(root)
        let context = try await files.prepare(
            path, requestID: UUID(), hardware: "fixture-device", osBuild: "fixture-os")
        var report = ImageExecutionReport(
            requestID: UUID(), profileID: "fixture-profile",
            manifestSHA256: String(repeating: "a", count: 64), noiseSource: "fixture", diagnosticsEnabled: false)
        report.status = .completed
        await #expect(throws: ImageExecutionError.invalidResource) {
            try await files.finish(
                context, result: nil, report: report, error: nil, cleanupError: nil, cancelled: false)
        }
        #expect(!FileManager.default.fileExists(atPath: context.directory.appending(path: "completion.json").path))
    }

    @Test("Successful diagnostics retain exact transparent pixels and reject a result with another model identity")
    func completedPixels() async throws {
        let root = try diagnosticTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let files = ImageDiagnosticFiles()
        let path = try diagnosticSpecification(root)
        let context = try await files.prepare(
            path, requestID: UUID(), hardware: "fixture-device", osBuild: "fixture-os")
        let bytes: [UInt8] = (0..<(512 * 512)).flatMap { index in [UInt8(index % 256), 128, 192, UInt8(index % 4)] }
        let pixels = try ImageGenerationPixels(width: 512, height: 512, rgba: bytes)
        var wrong = ImageExecutionReport(
            requestID: context.request.id, profileID: "other-profile",
            manifestSHA256: String(repeating: "a", count: 64), noiseSource: "fixture", diagnosticsEnabled: false)
        wrong.status = .completed
        let invalid = ImageGenerationResult(requestID: context.request.id, pixels: pixels, report: wrong)
        await #expect(throws: ImageExecutionError.invalidResource) {
            try await files.finish(
                context, result: invalid, report: nil, error: nil, cleanupError: nil, cancelled: false)
        }
        #expect(!FileManager.default.fileExists(atPath: context.directory.appending(path: "output.png").path))
        var report = ImageExecutionReport(
            requestID: context.request.id, profileID: "fixture-profile",
            manifestSHA256: String(repeating: "a", count: 64), noiseSource: "fixture", diagnosticsEnabled: false)
        report.status = .completed
        let valid = ImageGenerationResult(requestID: context.request.id, pixels: pixels, report: report)
        let outcome = try await files.finish(
            context, result: valid, report: report, error: nil, cleanupError: nil, cancelled: false)
        #expect(outcome.status == .completed)
        #expect(try Data(contentsOf: context.directory.appending(path: "output.rgba")) == Data(bytes))
        #expect(try ImageGenerationImageReader.read(context.directory.appending(path: "output.png")) == pixels)
    }
}

func diagnosticTemporaryDirectory() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    return root
}

func diagnosticDigest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func diagnosticSpecification(_ root: URL, noise: [String: String]? = nil) throws -> URL {
    let budgets = Dictionary(
        uniqueKeysWithValues: ImageModelComponent.allCases.map {
            ($0.rawValue, ["loadingAdditionalBytes": 1, "inferenceAdditionalBytes": 1])
        })
    var value: [String: Any] = [
        "format": 1, "runID": "fixture", "bundlePath": "fixture-bundle", "bundleBytes": 1,
        "budget": [
            "profileID": "fixture-profile", "manifestSHA256": String(repeating: "a", count: 64), "components": budgets,
        ],
        "budgetBasis": "estimate", "budgetEvidence": "Synthetic test only", "hardwareModel": "fixture-device",
        "osBuild": "fixture-os", "prompt": "fixture prompt", "seed": Int64.max, "trace": false,
    ]
    if let noise { value["noise"] = noise }
    let path = root.appending(path: "specification.json")
    try JSONSerialization.data(withJSONObject: value).write(to: path)
    return path
}
