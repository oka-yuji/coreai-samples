import Foundation
import ModelLibraryFeature
import SpeechAssets
import Testing

@testable import ModelLibraryStorage

struct VoxCPMReferenceMacStoreTests {
    @Test("VCM-02 Two local VoxCPM2 registrations keep separate directories and reject cross-profile replacements")
    func independentImports() async throws {
        let data = try VoxCPMImportFixture()
        defer { data.files.remove() }
        let parts = try VoxCPMImportFixture.contents(.referenceFP32)
        let inventory = try VoxCPMImportFixture.inventory(parts)
        let reference = try VoxCPMModelArtifact.makeReference(inventory: inventory)
        let model = LibraryModel(
            id: "voxcpm2", name: "VoxCPM2", category: "Speech Synthesis",
            artifacts: [data.artifact, reference], defaultArtifactID: data.artifact.id, platforms: [.macOS], demos: [])
        let store = ManagedModelStore(
            root: data.files.root, removeMigratedSources: false, clock: { Date(timeIntervalSince1970: 1000) },
            writeLedger: { try $0.write(to: $1, options: .atomic) }, availableCapacity: { _ in .max },
            voxcpmInventory: data.inventory, voxcpmReferenceInventory: inventory)
        try data.files.write(data.contents, to: data.source)
        try data.files.write(parts, to: data.replacement)
        let input = try data.digests(data.replacement)
        func install(_ asset: ModelArtifact, from url: URL) async throws -> ValidatedModelResource {
            let prepared = try await store.copyForImport(from: url, model: model, artifact: asset) { _ in }
            return try await store.promote(staging: prepared, model: model, artifact: asset)
        }
        let legacy = try await install(data.artifact, from: data.source)
        let approved = try await install(reference, from: data.replacement)
        #expect(legacy.identity != approved.identity && legacy.directory != approved.directory)
        #expect(data.artifact.files.count == 22 && reference.files.count == 25)
        #expect(try await store.inspect(model: model, artifact: reference, verification: .full).resource == approved)
        let ledger = try data.ledger()
        await #expect(throws: ManagedModelStoreError.contractMismatch("VoxCPM2 configuration fp16")) {
            try await store.copyForImport(from: data.source, model: model, artifact: reference) { _ in }
        }
        await #expect(throws: ManagedModelStoreError.contractMismatch("VoxCPM2 configuration reference_fp32")) {
            try await store.copyForImport(from: data.replacement, model: model, artifact: data.artifact) { _ in }
        }
        let oldFullPrecision = data.files.base.appending(path: "legacy-fp32", directoryHint: .isDirectory)
        try data.files.write(VoxCPMImportFixture.contents(.fp32), to: oldFullPrecision)
        await #expect(throws: ManagedModelStoreError.contractMismatch("VoxCPM2 configuration fp32")) {
            try await store.copyForImport(from: oldFullPrecision, model: model, artifact: reference) { _ in }
        }
        let forged = ModelArtifact(
            id: reference.id, title: reference.title, contract: reference.contract, platforms: reference.platforms,
            sizeBytes: reference.sizeBytes, download: try VoxCPMModelArtifact.makeDownload().download,
            upstream: reference.upstream, files: reference.files, localBundle: reference.localBundle)
        #expect(forged.isSourceValid && forged != reference)
        await #expect(throws: ManagedModelStoreError.contractMismatch("VoxCPM2 inventory")) {
            try await store.stagingDirectory(for: VoxCPMImportFixture.model(forged), artifact: forged)
        }
        #expect(try data.ledger() == ledger)
        #expect(try data.digests(data.replacement) == input)
        #expect(try await store.inspect(model: model, artifact: data.artifact, verification: .full).resource == legacy)
        let lease = try await store.acquire(
            model: model, artifact: reference, directory: approved.directory, owner: UUID())
        await #expect(throws: ManagedModelStoreError.inUse) { try await store.delete(identity: approved.identity) }
        await lease.release()
        #expect(try await store.delete(identity: approved.identity) == approved.sizeBytes)
        #expect(try await store.inspect(model: model, artifact: data.artifact, verification: .full).resource == legacy)
    }
}
