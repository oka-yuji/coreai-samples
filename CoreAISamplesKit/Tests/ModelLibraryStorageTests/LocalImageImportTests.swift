import Foundation
import ModelLibraryFeature
import ModelLibraryTestSupport
import Testing

@testable import ModelLibraryStorage

@Suite(.timeLimit(.minutes(1)))
struct LocalImageImportTests {
    @Test(
        "QMI-01 Local manifest identity survives promotion, restart inspection and deletion without changing the selected folder"
    )
    func installRestartDelete() async throws {
        let fixture = try ImageBundleFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let installed = try await fixture.install(into: store)
        #expect(installed.sourceKind == .localManifest)
        #expect(installed.revision == fixture.artifact.localBundle?.manifestSHA256)
        #expect(installed.matches(model: fixture.model, artifact: fixture.artifact))
        let restarted = fixture.store()
        guard case .ready(let restored) = try await restarted.inspect(model: fixture.model, artifact: fixture.artifact)
        else {
            Issue.record("The local resource was not restored")
            return
        }
        #expect(restored == installed)
        #expect(try await restarted.delete(identity: restored.identity) == restored.sizeBytes)
        #expect(!FileManager.default.fileExists(atPath: restored.directory.path))
        for (name, bytes) in fixture.files {
            #expect(try Data(contentsOf: fixture.source.appending(path: name)) == bytes)
        }
    }

    @Test("QMI-02 Unknown manifest bytes never gain trust from their self-reported verified flag")
    func untrustedManifest() async throws {
        let fixture = try ImageBundleFixture()
        defer { fixture.remove() }
        let path = fixture.source.appending(path: "bundle.json")
        let text = try String(contentsOf: path, encoding: .utf8)
        try Data(text.replacingOccurrences(of: "fixture-qmi-profile", with: "changed-qmi-profile").utf8).write(to: path)
        let store = fixture.store()
        await #expect(throws: ManagedModelStoreError.checksumMismatch("bundle.json")) {
            try await store.copyForImport(
                from: fixture.source, model: fixture.model, artifact: fixture.artifact, progress: { _ in })
        }
        guard case .missing = try await store.inspect(model: fixture.model, artifact: fixture.artifact) else {
            Issue.record("An untrusted manifest became ready")
            return
        }
    }

    @Test("QMI-02 Even a pinned manifest must retain reviewed acceptance and every component's matching contract")
    func inconsistentKnownManifest() async throws {
        for defect in ["unreviewed", "component", "profile"] {
            let fixture = try ImageBundleFixture { manifest, files in
                if defect == "unreviewed" {
                    manifest["verified"] = false
                } else {
                    let name = "text/component.json"
                    let bytes = try #require(files[name])
                    var header = try #require(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
                    if defect == "component" {
                        header["kind"] = "vision"
                    } else {
                        var profile = try #require(header["profile"] as? [String: Any])
                        profile["id"] = "another-profile"
                        header["profile"] = profile
                    }
                    files[name] = try JSONSerialization.data(withJSONObject: header, options: .sortedKeys)
                }
            }
            defer { fixture.remove() }
            let store = fixture.store()
            await #expect(throws: (any Error).self) { try await fixture.install(into: store) }
            guard case .missing = try await store.inspect(model: fixture.model, artifact: fixture.artifact) else {
                Issue.record("An inconsistent bundle became ready")
                return
            }
        }
    }

    @Test("QMI-03 Extra files, missing files and symbolic links are rejected before promotion")
    func sourceStructure() async throws {
        for defect in ["extra", "missing", "link"] {
            let fixture = try ImageBundleFixture()
            defer { fixture.remove() }
            let weights = fixture.source.appending(path: "text/text.aimodel/weights.bin")
            if defect == "extra" {
                try Data("extra".utf8).write(to: fixture.source.appending(path: "extra"))
            } else if defect == "missing" {
                try FileManager.default.removeItem(at: weights)
            } else {
                try FileManager.default.removeItem(at: weights)
                try FileManager.default.createSymbolicLink(
                    at: weights, withDestinationURL: fixture.source.appending(path: "LICENSE"))
            }
            await #expect(throws: (any Error).self) {
                try await fixture.store().copyForImport(
                    from: fixture.source, model: fixture.model, artifact: fixture.artifact, progress: { _ in })
            }
            #expect(!FileManager.default.fileExists(atPath: fixture.root.appending(path: "artifacts").path))
        }
    }

    @Test("QMI-03 Cancellation and insufficient capacity preserve an installed local bundle")
    func cancelAndCapacity() async throws {
        let fixture = try ImageBundleFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let installed = try await fixture.install(into: store)
        let operation = Task {
            try await store.copyForImport(from: fixture.source, model: fixture.model, artifact: fixture.artifact) { _ in
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        await #expect(throws: CancellationError.self) { try await operation.value }
        await #expect(throws: ManagedModelStoreError.insufficientSpace) {
            try await fixture.store(available: 0).copyForImport(
                from: fixture.source, model: fixture.model, artifact: fixture.artifact, progress: { _ in })
        }
        guard case .ready(let resource) = try await store.inspect(model: fixture.model, artifact: fixture.artifact)
        else {
            Issue.record("Cancellation lost the previous resource")
            return
        }
        #expect(resource == installed)
    }

    @Test("QMI-03 Import cannot copy the managed installation or its parent into itself")
    func overlappingSource() async throws {
        let fixture = try ImageBundleFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let installed = try await fixture.install(into: store)
        for source in [installed.directory, fixture.root, fixture.base] {
            await #expect(throws: ManagedModelStoreError.pathEscape) {
                try await store.copyForImport(
                    from: source, model: fixture.model, artifact: fixture.artifact, progress: { _ in })
            }
        }
        guard case .ready(let resource) = try await store.inspect(model: fixture.model, artifact: fixture.artifact)
        else {
            Issue.record("Rejecting an overlapping source lost the installation")
            return
        }
        #expect(resource == installed)
        for (name, bytes) in fixture.files {
            #expect(try Data(contentsOf: fixture.source.appending(path: name)) == bytes)
        }
    }

    @Test("QMI-04 A live lease rejects both import and deletion and a staged bundle waits for lease release")
    func leaseProtection() async throws {
        let fixture = try ImageBundleFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let installed = try await fixture.install(into: store)
        let staged = try await store.copyForImport(
            from: fixture.source, model: fixture.model, artifact: fixture.artifact, progress: { _ in })
        let lease = await store.lease(identity: installed.identity, owner: UUID())
        await #expect(throws: ManagedModelStoreError.inUse) {
            try await store.copyForImport(
                from: fixture.source, model: fixture.model, artifact: fixture.artifact, progress: { _ in })
        }
        await #expect(throws: ManagedModelStoreError.inUse) { try await store.delete(identity: installed.identity) }
        await #expect(throws: ManagedModelStoreError.inUse) {
            try await store.promote(staging: staged, model: fixture.model, artifact: fixture.artifact)
        }
        await lease.release()
        #expect(try await store.promote(staging: staged, model: fixture.model, artifact: fixture.artifact) == installed)
    }
}
