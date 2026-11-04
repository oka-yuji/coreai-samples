import Foundation
import ModelLibraryFeature
import Testing

@testable import ModelLibraryStorage

struct RecordedInspectionTests {
    @Test(
        "LIGHT-01 Recorded artifacts with matching structure are ready without reading their contents",
        arguments: ["x.aimodel/main.mlirb", "contract.json"])
    func sameSizeChange(_ path: String) async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let sample = try Sample.gliner()
        let store = fixture.store()
        let installed = try await fixture.install(sample, into: store)
        let ledger = try Data(contentsOf: fixture.root.appending(path: "ledger.json"))
        let file = installed.directory.appending(path: path)
        var data = try Data(contentsOf: file)
        data[0] ^= 1
        try data.write(to: file)
        #expect(try await fixture.store().inspect(sample).resource == installed)
        #expect(
            try await store.inspect(model: sample.model, artifact: sample.artifact, verification: .full).reason
                == describe(ManagedModelStoreError.checksumMismatch(path)))
        #expect(try Data(contentsOf: fixture.root.appending(path: "ledger.json")) == ledger)
    }

    @Test(
        "LIGHT-02 A missing or mismatched ledger cannot trust a same-size changed artifact",
        arguments: ["missing", "corrupt", "sizeBytes", "upstreamRevision", "modelID"])
    func untrustedLedger(_ defect: String) async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let sample = try Sample.gliner()
        let installed = try await fixture.install(sample, into: fixture.store())
        let path = "x.aimodel/main.mlirb"
        let file = installed.directory.appending(path: path)
        var data = try Data(contentsOf: file)
        data[0] ^= 1
        try data.write(to: file)
        let ledgerURL = fixture.root.appending(path: "ledger.json")
        switch defect {
        case "missing": try FileManager.default.removeItem(at: ledgerURL)
        case "corrupt": try Data("invalid".utf8).write(to: ledgerURL)
        default:
            let raw = try Data(contentsOf: ledgerURL)
            var ledger = try #require(JSONSerialization.jsonObject(with: raw) as? [String: Any])
            var entries = try #require(ledger["entries"] as? [[String: Any]])
            entries[0][defect] = defect == "sizeBytes" ? 1 : "different"
            ledger["entries"] = entries
            try JSONSerialization.data(withJSONObject: ledger).write(to: ledgerURL)
        }
        let before = try? Data(contentsOf: ledgerURL)
        #expect(
            try await fixture.store().inspect(sample).reason
                == describe(ManagedModelStoreError.checksumMismatch(path)))
        #expect((try? Data(contentsOf: ledgerURL)) == before)
    }

    @Test(
        "LIGHT-02 Recorded artifacts still reject missing, resized, extra and symbolic-link entries",
        arguments: ["missing", "size", "extra", "link"])
    func structureChange(_ defect: String) async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let sample = try Sample.gliner()
        let store = fixture.store()
        let installed = try await fixture.install(sample, into: store)
        let file = installed.directory.appending(path: "x.aimodel/main.mlirb")
        switch defect {
        case "missing": try FileManager.default.removeItem(at: file)
        case "size": try Data("short".utf8).write(to: file)
        case "extra": try Data().write(to: installed.directory.appending(path: "extra"))
        default:
            let outside = fixture.base.appending(path: "outside")
            try FileManager.default.moveItem(at: file, to: outside)
            try FileManager.default.createSymbolicLink(at: file, withDestinationURL: outside)
        }
        #expect(try await store.inspect(sample).resource == nil)
    }

    @Test("LIGHT-03 Promotion rejects a same-size change before recording the artifact")
    func promotionVerifiesContent() async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let sample = try Sample.gliner()
        let store = fixture.store()
        let staging = try await store.staging(sample)
        try fixture.write(sample.files, to: staging)
        let path = "x.aimodel/main.mlirb"
        let file = staging.appending(path: path)
        var data = try Data(contentsOf: file)
        data[0] ^= 1
        try data.write(to: file)
        await #expect(throws: ManagedModelStoreError.checksumMismatch(path)) {
            try await store.promote(staging, sample)
        }
        #expect(try await store.inspect(sample).isMissing)
        #expect(!FileManager.default.fileExists(atPath: fixture.root.appending(path: "ledger.json").path))
    }
}
