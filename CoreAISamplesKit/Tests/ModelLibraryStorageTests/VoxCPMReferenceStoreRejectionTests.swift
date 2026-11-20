import Foundation
import ModelLibraryFeature
import SpeechAssets
import Synchronization
import Testing

@testable import ModelLibraryStorage

struct VoxCPMReferenceStoreRejectionTests {
    @Test(
        "VCM-03 A rejected replacement preserves the existing Vox registration, source and managed files",
        arguments: ["altered", "missing", "extra", "symlink", "cancel"])
    func referenceRejects(_ damage: String) async throws {
        try await reject(reference: true, damage: damage)
    }
    @Test(
        "The original FP16 storage path rejects altered, incomplete and linked bundles and a cancellation after copying",
        arguments: ["altered", "missing", "extra", "symlink", "cancel"])
    func legacyControls(_ damage: String) async throws {
        try await reject(reference: false, damage: damage)
    }
    private func reject(reference: Bool, damage: String) async throws {
        let fixture = try VoxCPMImportFixture()
        defer { fixture.files.remove() }
        let parts = try VoxCPMImportFixture.contents(reference ? .referenceFP32 : .fp16)
        let inventory = try VoxCPMImportFixture.inventory(parts)
        let artifact =
            try reference
            ? VoxCPMModelArtifact.makeReference(inventory: inventory) : VoxCPMModelArtifact.make(inventory: inventory)
        let model = VoxCPMImportFixture.model(artifact)
        let store = ManagedModelStore(
            root: fixture.files.root, removeMigratedSources: false, clock: { Date(timeIntervalSince1970: 1000) },
            writeLedger: { try $0.write(to: $1, options: .atomic) }, availableCapacity: { _ in .max },
            voxcpmInventory: fixture.inventory, voxcpmReferenceInventory: reference ? inventory : nil)
        try fixture.files.write(parts, to: fixture.source)
        let stage = try await store.copyForImport(from: fixture.source, model: model, artifact: artifact) { _ in }
        let installed = try await store.promote(staging: stage, model: model, artifact: artifact)
        let ledger = try fixture.ledger()
        let previous = try fixture.digests(installed.directory)
        try fixture.files.write(parts, to: fixture.replacement)
        let relative = reference ? "artifacts/refenc-fp32.aimodel/main.hash" : "artifacts/tslm-fp16.aimodel/main.hash"
        let target = fixture.replacement.appending(path: relative)
        let expected: ManagedModelStoreError?
        switch damage {
        case "altered":
            var changed = try Data(contentsOf: target)
            changed[0] ^= 1
            try changed.write(to: target)
            expected = .checksumMismatch(relative)
        case "missing":
            try FileManager.default.removeItem(at: target)
            expected = .missingFile(relative)
        case "extra":
            try Data().write(to: fixture.replacement.appending(path: "extra.bin"))
            expected = .unexpectedFile("extra.bin")
        case "symlink":
            let external = fixture.files.base.appending(path: "outside.bin")
            try FileManager.default.moveItem(at: target, to: external)
            try FileManager.default.createSymbolicLink(at: target, withDestinationURL: external)
            expected = .symbolicLink
        default:
            try #require(damage == "cancel")
            expected = nil
        }
        let sourceBefore = try fixture.digests(fixture.replacement)
        if let expected {
            await #expect(throws: expected) {
                try await store.copyForImport(from: fixture.replacement, model: model, artifact: artifact) { _ in }
            }
        } else {
            let copied = Mutex<Int64>(0)
            let source = fixture.replacement
            await #expect(throws: CancellationError.self) {
                try await withThrowingTaskGroup(of: URL.self) { group in
                    group.addTask {
                        try await store.copyForImport(from: source, model: model, artifact: artifact) { progress in
                            copied.withLock { $0 = progress.completedBytes }
                            withUnsafeCurrentTask { $0?.cancel() }
                        }
                    }
                    _ = try await group.next()
                }
            }
            #expect(copied.withLock { $0 } > 0)
        }
        #expect(try fixture.ledger() == ledger)
        #expect(try fixture.digests(fixture.replacement) == sourceBefore)
        #expect(try fixture.digests(installed.directory) == previous)
        let full = try await store.inspect(model: model, artifact: artifact, verification: .full)
        #expect(full.resource == installed)
        let path = fixture.files.root.appending(path: "staging/" + artifact.id + "/" + (artifact.sourceRevision ?? ""))
        #expect(((try? FileManager.default.contentsOfDirectory(atPath: path.path)) ?? []).isEmpty)
    }
}
