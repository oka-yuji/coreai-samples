import Foundation
import ImageGenerationCore
import ImageGenerationRuntime
import Testing

struct ImageDirectoryTraceTests {
    @Test("Diagnostic output retains nonfinite raw values with shape dtype and byte identity")
    func rawTrace() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let writer = ImageDirectoryTrace(directory: root)
        let contract = ImageTensorContract(name: "noise", shape: [3], dtype: .float32, layout: "row-major")
        try await writer.record(
            name: "denoiser-0", values: ["noise": .floats([1, .nan, .infinity])], contracts: [contract])
        let raw = try Data(contentsOf: root.appending(path: "denoiser-0-noise.f32"))
        #expect(raw.count == 12)
        let values = raw.withUnsafeBytes { bytes in
            (0..<3).map {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self)))
            }
        }
        #expect(values[0] == 1 && values[1].isNaN && values[2].isInfinite)
        let metadata = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: root.appending(path: "denoiser-0.json")))
                as? [String: Any])
        let tensors = try #require(metadata["tensors"] as? [[String: Any]])
        #expect(tensors[0]["nonfiniteElements"] as? Int == 2)
        #expect((tensors[0]["sha256"] as? String)?.count == 64)
        await #expect(throws: ImageExecutionError.self) {
            try await writer.record(name: "denoiser-0", values: ["noise": .floats([0, 0, 0])], contracts: [contract])
        }
        #expect(try Data(contentsOf: root.appending(path: "denoiser-0-noise.f32")) == raw)
    }

    @Test("Diagnostics refuse an existing destination and names that could leave the trace directory")
    func protectedDestination() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let writer = ImageDirectoryTrace(directory: root)
        let contract = ImageTensorContract(name: "x", shape: [1], dtype: .int32, layout: "row-major")
        await #expect(throws: ImageExecutionError.self) {
            try await writer.record(name: "../escape", values: ["x": .integers([1])], contracts: [contract])
        }
        await #expect(throws: POSIXError.self) {
            try await writer.record(name: "safe", values: ["x": .integers([1])], contracts: [contract])
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty)
    }
}
