import Darwin
import Foundation
import ModelLibraryFeature
import ModelLibraryTransfer
import Testing

@Suite(.timeLimit(.minutes(1)))
struct ArtifactTransferMemoryTests {
    @Test(
        "QMH-01 Reusing a large artifact preserves its checksum without retaining the whole file in memory",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_HASH_MEMORY_TEST"] == "1")
    )
    func testGivenLargeArtifactWhenReusedThenHashMemoryIsBounded() async throws {
        let size = try #require(Int(ProcessInfo.processInfo.environment["COREAI_HASH_MEMORY_MIB"] ?? "256"))
        let hashes = [
            64: "3b6a07d0d404fab4e23b6d34bc6696a6a312dd92821332385e5af7c01c421351",
            256: "a6d72ac7690f53be6ae46ba88506bd97302a093f7108472bd9efc3cefda06484",
        ]
        let hash = try #require(hashes[size])
        let bytes = Int64(size) * 1_048_576
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appending(path: "weights.bin")
        try #require(FileManager.default.createFile(atPath: file.path, contents: nil))
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: UInt64(bytes))
        try handle.close()
        let artifact = ModelArtifact(
            id: "hash-memory", title: "Hash memory", contract: "fixture", platforms: [.macOS], sizeBytes: bytes,
            download: ModelDownload(repository: "owner/fixture", revision: revision),
            files: [ModelArtifactFile(path: "weights.bin", bytes: bytes, sha256: hash)])
        let model = LibraryModel(
            id: "hash-memory", name: "Hash memory", category: "Test", artifacts: [artifact],
            defaultArtifactID: artifact.id, platforms: [.macOS], demos: [])
        let transfer = ArtifactTransfer(
            fileLoader: { _, _ in throw MemoryCheckError.unexpectedDownload },
            availableCapacity: { _ in .max })
        try await transfer.download(model: model, artifact: artifact, to: directory) { _ in }
        let peak = try kernelPeak()
        print("QMH-01 synthetic_bytes=\(bytes) kernel_lifetime_peak_bytes=\(peak)")
        #expect(peak < UInt64(bytes) / 2)
        let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
        #expect((attributes[.size] as? NSNumber)?.int64Value == bytes)
    }

    private func kernelPeak() throws -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let offset = try #require(MemoryLayout<task_vm_info_data_t>.offset(of: \.ledger_phys_footprint_peak))
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        try #require(status == KERN_SUCCESS)
        try #require(Int(count) * MemoryLayout<integer_t>.size >= offset + MemoryLayout<UInt64>.size)
        try #require(info.ledger_phys_footprint_peak > 0)
        return UInt64(info.ledger_phys_footprint_peak)
    }

    private enum MemoryCheckError: Error {
        case unexpectedDownload
    }
}
