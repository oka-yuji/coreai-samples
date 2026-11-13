import Foundation
import ModelLibraryFeature
import Testing

@testable import ModelLibraryStorage

struct ChronosDownloadTests {
    @Test("CD-01 The uploaded Chronos locator leaves the original manifest and inventory identity unchanged")
    func pinnedLocator() throws {
        let artifact = try ChronosModelArtifact.make()
        #expect(
            artifact.download
                == ModelDownload(
                    repository: "okayuji/chronos-2-coreai-fp32", revision: "d3ead488046498faa8b3b5103b1923e52a313d0e",
                    root: "bundle"))
        #expect(artifact.id == "chronos2-fp32" && artifact.sourceKind == .localManifest)
        #expect(artifact.sourceRevision == "786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572")
        #expect(artifact.files.count == 9 && artifact.totalFileBytes == 478_864_956)
        #expect(
            artifact.localBundle?.manifestPath == "chronos2.json" && artifact.localBundle?.profileID == "chronos2-fp32")
        #expect(artifact.upstream?.revision == "29ec3766d36d6f73f0696f85560a422f50e8498c")
    }

    @Test("CD-02 A recorded manifest installation keeps its location, ledger and lease when a locator is added")
    func recordedImport() async throws {
        let fixture = try ChronosImportFixture()
        defer { fixture.files.remove() }
        let old = try fixture.recordSyntheticInstallation()
        let ledgerURL = fixture.files.root.appending(path: "ledger.json")
        let ledger = try Data(contentsOf: ledgerURL)
        let store = fixture.store()
        let resource = try #require(try await store.inspect(model: fixture.model, artifact: fixture.artifact).resource)
        #expect(resource == old && resource.identity == old.identity)
        #expect(resource.directory.lastPathComponent == ChronosModelArtifact.manifestSHA256)
        #expect(resource.directory.lastPathComponent != fixture.artifact.download?.revision)
        let lease = try await store.acquire(
            model: fixture.model, artifact: fixture.artifact, directory: old.directory, owner: UUID())
        #expect(await store.activeLeases(identity: resource.identity) == 1)
        await #expect(throws: ManagedModelStoreError.inUse) { try await store.delete(identity: resource.identity) }
        #expect(try Data(contentsOf: ledgerURL) == ledger)
        await lease.release()
        #expect(await store.activeLeases(identity: old.identity) == 0)
        #expect(try await store.inspect(model: fixture.model, artifact: fixture.artifact).resource == old)
    }

    @Test("CD-01 Changing only the Chronos download locator still fails canonical storage validation")
    func forgedLocator() async throws {
        let fixture = try ChronosImportFixture()
        defer { fixture.files.remove() }
        var object = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(fixture.artifact)) as? [String: Any])
        object["download"] = [
            "repository": "other/model", "revision": String(repeating: "a", count: 40), "root": "bundle",
        ]
        let artifact = try JSONDecoder().decode(
            ModelArtifact.self, from: JSONSerialization.data(withJSONObject: object))
        #expect(artifact.isSourceValid && artifact.sourceRevision == fixture.artifact.sourceRevision)
        let model = LibraryModel(
            id: fixture.model.id, name: fixture.model.name, category: fixture.model.category, artifacts: [artifact],
            defaultArtifactID: artifact.id, platforms: [.macOS], demos: [])
        await #expect(throws: ManagedModelStoreError.contractMismatch("Chronos inventory")) {
            try await fixture.store().stagingDirectory(for: model, artifact: artifact)
        }
    }
}
