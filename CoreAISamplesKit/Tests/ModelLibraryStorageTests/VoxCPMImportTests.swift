import Darwin
import Foundation
import ModelLibraryFeature
import SpeechAssets
import Synchronization
import Testing

@testable import ModelLibraryStorage

struct VoxCPMImportTests {
    private static let pinned: [VoxCPMBundleProfile: (count: Int, bytes: Int64, manifest: String)] = [
        .fp32: (22, 9_349_325_520, "eda3c0098d35c8b305884e84ffd79005a7562cb027510ee2fa3b7f55ef2493da"),
        .fp16: (22, 4_677_413_664, "fbe90a07bcdb4736b2b8b9dcd0b1f11b71857c28e4d506aad216bcda6c76754c"),
        .fp16Proj32: (22, 4_715_188_166, "f7eb7f49fad8af5b58f1ed14c4ae53be508e885bbc1cbd93a0291bda30d536c6"),
    ]

    @Test("VML-01 The artifact pins the file count, total size and manifest SHA-256 of the bundled inventory")
    func definition() async throws {
        let profile = VoxCPMModelArtifact.profile
        let artifact = try VoxCPMModelArtifact.make()
        let inventory = try VoxCPMBundleInventory.files(profile: profile)
        let expected = try #require(Self.pinned[profile])
        #expect(
            artifact.files == inventory.map { ModelArtifactFile(path: $0.path, bytes: $0.bytes, sha256: $0.sha256) })
        #expect(artifact.files.count == expected.count && artifact.totalFileBytes == expected.bytes)
        #expect(artifact.sizeBytes == artifact.totalFileBytes)
        #expect(inventory.first { $0.path == VoxCPMBundleInventory.manifestPath }?.sha256 == expected.manifest)
        #expect(
            artifact.sourceRevision == expected.manifest && artifact.localBundle?.manifestSHA256 == expected.manifest)
        for configuration in VoxCPMBundleProfile.allCases {
            let listed = try #require(
                VoxCPMBundleInventory.files(profile: configuration).first {
                    $0.path == VoxCPMBundleInventory.manifestPath
                })
            let manifest = try VoxCPMImportFixture.manifest(configuration)
            #expect(sha256(manifest) == listed.sha256 && Int64(manifest.count) == listed.bytes, "\(configuration)")
        }
        #expect(artifact.id == "voxcpm2-" + profile.rawValue && artifact.contract == VoxCPMModelArtifact.contract)
        #expect(artifact.sourceKind == .localManifest && artifact.download == nil && artifact.isSourceValid)
        #expect(artifact.platforms == [.macOS] && artifact.localBundle?.manifestPath == "manifest.json")
        #expect(
            artifact.upstream == ModelUpstream(repository: "openbmb/VoxCPM2", revision: VoxCPMBundleInventory.revision))
        #expect(try JSONDecoder().decode(ModelArtifact.self, from: JSONEncoder().encode(artifact)) == artifact)
        let policy = ModelAcquisitionPolicy(artifact: artifact)
        #expect(policy.preferredMethod == .importBundle && policy.allowsImport && policy.readyTitle == "Imported")
        #expect(try VoxCPMImportFixture.contents(profile).keys.sorted() == artifact.files.map(\.path))
        let files = StoreFixture()
        defer { files.remove() }
        let store = files.store()
        let model = VoxCPMImportFixture.model(artifact)
        #expect(try await store.inspect(model: model, artifact: artifact).isMissing)
        #expect(try await store.stagingDirectory(for: model, artifact: artifact).lastPathComponent == expected.manifest)
    }

    @Test(
        "VML-01 A definition that differs from the pinned inventory is never trusted",
        arguments: ["hash", "bytes", "missing", "extra", "manifest", "profile", "platforms", "upstream"])
    func forgedDefinition(_ field: String) async throws {
        let known = try VoxCPMModelArtifact.make()
        var files = known.files
        var source = known.localBundle
        var platforms = known.platforms
        var upstream = known.upstream
        switch field {
        case "hash":
            files[0] = ModelArtifactFile(path: files[0].path, bytes: files[0].bytes, sha256: sha256(Data("x".utf8)))
        case "bytes":
            files[0] = ModelArtifactFile(path: files[0].path, bytes: files[0].bytes + 1, sha256: files[0].sha256)
        case "missing": files.removeAll { $0.path == VoxCPMBundleInventory.tokenizerPath }
        case "extra": files.append(ModelArtifactFile(path: "extra.bin", bytes: 0, sha256: sha256(Data())))
        case "manifest":
            let changed = String(repeating: "b", count: 64)
            files = files.map {
                $0.path == VoxCPMBundleInventory.manifestPath
                    ? ModelArtifactFile(path: $0.path, bytes: $0.bytes, sha256: changed) : $0
            }
            source = ModelLocalBundle(manifestSHA256: changed, profileID: known.id, manifestPath: "manifest.json")
        case "profile":
            source = ModelLocalBundle(
                manifestSHA256: known.sourceRevision ?? "", profileID: "voxcpm2-other", manifestPath: "manifest.json")
        case "platforms": platforms = [.macOS, .iOS]
        default: upstream = ModelUpstream(repository: "other/model", revision: VoxCPMBundleInventory.revision)
        }
        let artifact = ModelArtifact(
            id: known.id, title: known.title, contract: known.contract, platforms: platforms,
            sizeBytes: known.sizeBytes, upstream: upstream, files: files, localBundle: source)
        #expect(artifact.isSourceValid && artifact != known)
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let model = VoxCPMImportFixture.model(artifact)
        let rejection = ManagedModelStoreError.contractMismatch("VoxCPM2 inventory")
        #expect(try await store.inspect(model: model, artifact: artifact).reason == describe(rejection))
        await #expect(throws: rejection) { try await store.stagingDirectory(for: model, artifact: artifact) }
    }

    @Test("VML-02 A synthetic bundle with an injected inventory imports with a full check and leaves its folder intact")
    func syntheticImport() async throws {
        let fixture = try VoxCPMImportFixture()
        defer { fixture.files.remove() }
        let store = fixture.store()
        try fixture.files.write(fixture.contents, to: fixture.source)
        let selected = fixture.files.contents(fixture.source)
        let copied = Mutex<Int64>(0)
        let staging = try await store.copyForImport(
            from: fixture.source, model: fixture.model, artifact: fixture.artifact
        ) { progress in copied.withLock { $0 = progress.completedBytes } }
        #expect(copied.withLock { $0 } == fixture.artifact.totalFileBytes)
        let installed = try await store.promote(staging: staging, model: fixture.model, artifact: fixture.artifact)
        #expect(installed.sourceKind == .localManifest && installed.revision == fixture.artifact.sourceRevision)
        #expect(installed.sizeBytes == fixture.artifact.totalFileBytes && installed.directory != fixture.source)
        #expect(
            try await store.inspect(model: fixture.model, artifact: fixture.artifact, verification: .full).resource
                == installed)
        #expect(fixture.files.contents(installed.directory) == selected)
        #expect(fixture.files.contents(fixture.source) == selected)
        #expect(fixture.stagingIsEmpty())
    }

    @Test(
        "VML-02 Changed, missing, extra or linked files are rejected and the registration and folders stay unchanged",
        arguments: ["altered", "missing", "extra", "symlink"])
    func rejectedImport(_ kind: String) async throws {
        let fixture = try VoxCPMImportFixture()
        defer { fixture.files.remove() }
        let store = fixture.store()
        let installed = try await fixture.install(into: store)
        try fixture.files.write(fixture.contents, to: fixture.replacement)
        let expected = try fixture.damage(kind, at: fixture.replacement)
        try await expectRejection(
            fixture, store: store, installed: installed, expected: expected, copiesFirst: kind == "altered")
    }

    @Test(
        "VML-02 A bundle of another configuration is rejected by name and the registration and folders stay unchanged",
        arguments: VoxCPMBundleProfile.allCases.filter { $0 != VoxCPMModelArtifact.profile })
    func otherConfiguration(_ profile: VoxCPMBundleProfile) async throws {
        let fixture = try VoxCPMImportFixture()
        defer { fixture.files.remove() }
        let store = fixture.store()
        let installed = try await fixture.install(into: store)
        try fixture.files.write(VoxCPMImportFixture.contents(profile), to: fixture.replacement)
        try await expectRejection(
            fixture, store: store, installed: installed,
            expected: .contractMismatch("VoxCPM2 configuration " + profile.rawValue), copiesFirst: false)
    }

    @Test("VML-02 A staged copy that changes before promotion is rejected and the previous registration stays ready")
    func changedStaging() async throws {
        let fixture = try VoxCPMImportFixture()
        defer { fixture.files.remove() }
        let store = fixture.store()
        let installed = try await fixture.install(into: store)
        let ledger = try fixture.ledger()
        let previous = fixture.files.contents(installed.directory)
        let staging = try await store.copyForImport(
            from: fixture.source, model: fixture.model, artifact: fixture.artifact
        ) { _ in }
        let expected = try fixture.damage("altered", at: staging)
        await #expect(throws: expected) {
            try await store.promote(staging: staging, model: fixture.model, artifact: fixture.artifact)
        }
        try await store.discardImport(staging: staging, model: fixture.model, artifact: fixture.artifact)
        #expect(
            try await store.inspect(model: fixture.model, artifact: fixture.artifact, verification: .full).resource
                == installed)
        #expect(try fixture.ledger() == ledger)
        #expect(fixture.files.contents(installed.directory) == previous)
        #expect(fixture.stagingIsEmpty())
    }

    @Test("VML-02 With the pinned inventory the real manifest is accepted and a sparse payload is rejected")
    func pinnedManifestReachesPayload() async throws {
        let artifact = try VoxCPMModelArtifact.make()
        let model = VoxCPMImportFixture.model(artifact)
        let files = StoreFixture()
        defer { files.remove() }
        for file in artifact.files {
            let url = files.documents.appending(path: file.path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: url)
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.truncate(atOffset: UInt64(file.bytes))
        }
        let manifest = try VoxCPMImportFixture.manifest(VoxCPMModelArtifact.profile)
        try manifest.write(to: files.documents.appending(path: VoxCPMBundleInventory.manifestPath))
        let store = files.store()
        await #expect(throws: ManagedModelStoreError.checksumMismatch("LICENSE")) {
            try await store.copyForImport(from: files.documents, model: model, artifact: artifact) { _ in }
        }
        #expect(try await store.inspect(model: model, artifact: artifact).isMissing)
        #expect(try Data(contentsOf: files.documents.appending(path: VoxCPMBundleInventory.manifestPath)) == manifest)
    }

    @Test("VML-03 An import in use refuses deletion and replacement until every lease is released, then deletes")
    func leaseGating() async throws {
        let fixture = try VoxCPMImportFixture()
        defer { fixture.files.remove() }
        let store = fixture.store()
        let installed = try await fixture.install(into: store)
        let selected = fixture.files.contents(fixture.source)
        let first = try await store.acquire(
            model: fixture.model, artifact: fixture.artifact, directory: installed.directory, owner: UUID())
        let second = try await store.acquire(
            model: fixture.model, artifact: fixture.artifact, directory: installed.directory, owner: UUID())
        await #expect(throws: ManagedModelStoreError.inUse) { try await store.delete(identity: installed.identity) }
        await #expect(throws: ManagedModelStoreError.inUse) {
            try await store.copyForImport(from: fixture.source, model: fixture.model, artifact: fixture.artifact) { _ in
            }
        }
        await first.release()
        #expect(await store.activeLeases(identity: installed.identity) == 1)
        await #expect(throws: ManagedModelStoreError.inUse) { try await store.delete(identity: installed.identity) }
        #expect(
            try await store.inspect(model: fixture.model, artifact: fixture.artifact, verification: .full).resource
                == installed)
        await second.release()
        #expect(await store.activeLeases(identity: installed.identity) == 0)
        #expect(try await store.delete(identity: installed.identity) == installed.sizeBytes)
        #expect(try await store.inspect(model: fixture.model, artifact: fixture.artifact).isMissing)
        #expect(!FileManager.default.fileExists(atPath: installed.directory.path))
        #expect(fixture.files.contents(fixture.source) == selected)
    }

    @Test(
        "VML-05 The real bundle imports with a full check, rejects bad replacements and keeps its folder unchanged",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_VOXCPM2_IMPORT_BUNDLES"] != nil))
    func knownBundle() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_VOXCPM2_IMPORT_BUNDLES"])
        let bundles = URL(filePath: path, directoryHint: .isDirectory)
        let source = bundles.appending(path: VoxCPMModelArtifact.profile.rawValue, directoryHint: .isDirectory)
        let artifact = try VoxCPMModelArtifact.make()
        let model = VoxCPMImportFixture.model(artifact)
        let fixture = try VoxCPMImportFixture()
        defer { fixture.files.remove() }
        try FileManager.default.createDirectory(at: fixture.files.base, withIntermediateDirectories: true)
        let clock = ContinuousClock()
        let started = clock.now
        let before = try fixture.digests(source)
        #expect(
            before.filter { $0.value != "directory" }
                == Dictionary(uniqueKeysWithValues: artifact.files.map { ($0.path, $0.sha256) }))
        let hashed = clock.now
        let store = fixture.files.store()
        let staging = try await store.copyForImport(from: source, model: model, artifact: artifact) { _ in }
        let copied = clock.now
        let installed = try await store.promote(staging: staging, model: model, artifact: artifact)
        let promoted = clock.now
        #expect(installed.sourceKind == .localManifest && installed.revision == artifact.sourceRevision)
        #expect(installed.sizeBytes == artifact.totalFileBytes && installed.directory != source)
        #expect(try await store.inspect(model: model, artifact: artifact, verification: .full).resource == installed)
        let inspected = clock.now
        let lease = try await store.acquire(
            model: model, artifact: artifact, directory: installed.directory, owner: UUID())
        await #expect(throws: ManagedModelStoreError.inUse) { try await store.delete(identity: installed.identity) }
        await lease.release()
        let ledger = try fixture.ledger()
        for profile in VoxCPMBundleProfile.allCases where profile != VoxCPMModelArtifact.profile {
            let other = bundles.appending(path: profile.rawValue, directoryHint: .isDirectory)
            try #require(FileManager.default.fileExists(atPath: other.path))
            await #expect(throws: ManagedModelStoreError.contractMismatch("VoxCPM2 configuration " + profile.rawValue))
            {
                try await store.copyForImport(from: other, model: model, artifact: artifact) { _ in }
            }
            #expect(try await store.inspect(model: model, artifact: artifact).resource == installed)
        }
        for kind in ["altered", "missing", "extra", "symlink"] {
            try #require(clonefile(source.path, fixture.replacement.path, UInt32(CLONE_NOFOLLOW)) == 0)
            let expected = try fixture.damage(kind, at: fixture.replacement)
            await #expect(throws: expected) {
                try await store.copyForImport(from: fixture.replacement, model: model, artifact: artifact) { _ in }
            }
            #expect(try await store.inspect(model: model, artifact: artifact).resource == installed)
            try FileManager.default.removeItem(at: fixture.replacement)
        }
        #expect(try fixture.ledger() == ledger)
        #expect(try await store.inspect(model: model, artifact: artifact, verification: .full).resource == installed)
        let footprint = await store.deletionSize(identity: installed.identity)
        #expect(try await store.delete(identity: installed.identity) == installed.sizeBytes)
        #expect(try await store.inspect(model: model, artifact: artifact).isMissing)
        let rehashing = clock.now
        #expect(try fixture.digests(source) == before)
        let finished = clock.now
        print(
            "VML-05 files \(artifact.files.count) bytes \(installed.sizeBytes) managed \(footprint)",
            "hash \(hashed - started) copy \(copied - hashed) promote \(promoted - copied)",
            "full-inspection \(inspected - promoted) rehash \(finished - rehashing) total \(finished - started)")
    }

    private func expectRejection(
        _ fixture: VoxCPMImportFixture, store: ManagedModelStore, installed: ValidatedModelResource,
        expected: ManagedModelStoreError, copiesFirst: Bool
    ) async throws {
        let ledger = try fixture.ledger()
        let previous = fixture.files.contents(installed.directory)
        let selected = fixture.files.contents(fixture.replacement)
        let copied = Mutex<Int64>(0)
        await #expect(throws: expected) {
            try await store.copyForImport(
                from: fixture.replacement, model: fixture.model, artifact: fixture.artifact
            ) { progress in copied.withLock { $0 = progress.completedBytes } }
        }
        #expect(copied.withLock { $0 > 0 } == copiesFirst)
        #expect(
            try await store.inspect(model: fixture.model, artifact: fixture.artifact, verification: .full).resource
                == installed)
        #expect(try fixture.ledger() == ledger)
        #expect(fixture.files.contents(installed.directory) == previous)
        #expect(fixture.files.contents(fixture.replacement) == selected)
        #expect(fixture.stagingIsEmpty())
    }
}
