import Foundation
import ModelLibraryFeature
import Testing

@testable import ModelLibraryStorage

struct ManagedModelStoreBoundaryTests {
    @Test(
        "STORE-05 Parent folders replaced by symbolic links are rejected, and the folders they point to are never moved or deleted"
    )
    func parentSymbolicLinks() async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let gliner = try Sample.gliner()
        let installed = try await fixture.install(gliner, into: store)
        let parent = fixture.root.appending(path: "artifacts/\(gliner.artifact.id)")
        let moved = fixture.documents.appending(path: "legacy-artifact")
        try FileManager.default.createDirectory(at: fixture.documents, withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: parent, to: moved)
        try FileManager.default.createSymbolicLink(at: parent, withDestinationURL: moved)
        let kept = fixture.contents(moved)
        #expect(try await store.inspect(gliner).reason == describe(ManagedModelStoreError.symbolicLink))
        await #expect(throws: ManagedModelStoreError.symbolicLink) {
            try await store.delete(identity: installed.identity)
        }
        #expect(await store.deletionSize(identity: installed.identity) == 0)
        #expect(!kept.isEmpty && fixture.contents(moved) == kept)

        let embedding = try Sample.embedding()
        let distributionFolder = fixture.documents.appending(path: "legacy-distribution")
        let external = distributionFolder.appending(path: distribution)
        try fixture.write(embedding.files, to: external)
        let stagingParent = fixture.root.appending(path: "staging/\(embedding.artifact.id)")
        try FileManager.default.createDirectory(
            at: stagingParent.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: stagingParent, withDestinationURL: distributionFolder)
        let outside = fixture.contents(distributionFolder)
        for staging in [external, fixture.stagingArea(embedding)] {
            await #expect(throws: ManagedModelStoreError.symbolicLink) { try await store.promote(staging, embedding) }
        }
        await #expect(throws: ManagedModelStoreError.symbolicLink) { try await store.staging(embedding) }
        #expect(fixture.contents(distributionFolder) == outside)
        #expect(!FileManager.default.fileExists(atPath: fixture.artifactDirectory(embedding).path))

        let library = fixture.base.appending(path: "Library")
        let linkedRoot = fixture.base.appending(path: "LinkedLibrary")
        try FileManager.default.createDirectory(at: library, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: linkedRoot, withDestinationURL: library)
        let linkedStore = ManagedModelStore(root: linkedRoot)
        let linked = try await fixture.install(gliner, into: linkedStore)
        #expect(try await linkedStore.inspect(gliner).resource == linked)
        #expect(try await linkedStore.delete(identity: linked.identity) == linked.sizeBytes)
    }

    @Test("STORE-01 An unreadable folder inside a bundle is rejected instead of hiding unlisted files and links")
    func unreadableFolders() async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let sample = try Sample.gliner()
        let installed = try await fixture.install(sample, into: store)
        let asset = installed.directory.appending(path: "x.aimodel")
        try Data("unlisted".utf8).write(to: asset.appending(path: "extra.bin"))
        try FileManager.default.createSymbolicLink(
            at: asset.appending(path: "extra-link"), withDestinationURL: asset.appending(path: "extra.bin"))
        try FileManager.default.setAttributes([.posixPermissions: 0o111], ofItemAtPath: asset.path)
        let unreadable = describe(ManagedModelStoreError.unreadableDirectory("x.aimodel"))
        #expect(try await store.inspect(sample).reason == unreadable)

        let staging = try await store.staging(sample)
        try fixture.write(sample.files, to: staging)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o111], ofItemAtPath: staging.appending(path: "x.aimodel").path)
        let refused = await #expect(throws: (any Error).self) { try await store.promote(staging, sample) }
        #expect(describe(refused) == unreadable)

        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: asset.path)
        let readable = try await store.inspect(sample)
        #expect(readable.resource == nil && readable.reason != nil)
    }

    @Test("STORE-01 Unlisted files with AppleDouble names are rejected in bundles and staging")
    func appleDoubleNames() async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let sample = try Sample.gliner()
        let installed = try await fixture.install(sample, into: store)
        for name in ["._unexpected", "x.aimodel/._unexpected"] {
            let extra = installed.directory.appending(path: name)
            try Data("unlisted".utf8).write(to: extra)
            #expect(try await store.inspect(sample).reason == describe(ManagedModelStoreError.unexpectedFile(name)))
            try FileManager.default.removeItem(at: extra)
        }
        #expect(try await store.inspect(sample).resource == installed)

        let staging = try await store.staging(sample)
        try fixture.write(sample.files.merging(["._unexpected": Data("unlisted".utf8)]) { $1 }, to: staging)
        await #expect(throws: ManagedModelStoreError.unexpectedFile("._unexpected")) {
            try await store.promote(staging, sample)
        }
        #expect(try await store.inspect(sample).resource == installed)
    }

    @Test("STORE-03 A candidate inside the managed area is reported as undetermined and left unchanged")
    func candidatesInsideTheStore() async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let sample = try Sample.gliner()
        let staging = try await store.staging(sample)
        try fixture.write(sample.files, to: staging)
        let staged = fixture.contents(staging)
        let alias = fixture.documents.appending(path: "gliner-alias")
        try FileManager.default.createDirectory(at: fixture.documents, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: staging)

        let report = await store.migrate([staging, fixture.root, alias], sample)
        for candidate in [staging, fixture.root, alias] {
            #expect(report.outcomes[candidate]?.undetermined == describe(ManagedModelStoreError.pathEscape))
        }
        #expect(fixture.contents(staging) == staged)
        #expect(try await store.inspect(sample).isMissing)
    }
}
