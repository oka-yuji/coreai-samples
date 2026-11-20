#if os(macOS)
    import Darwin
    import Foundation
    import ModelLibraryFeature
    import ModelLibraryStorage
    import Testing

    struct VoxCPMReferenceMacImportNativeTests {
        private static let env = ProcessInfo.processInfo.environment
        private static let inputKey = "COREAI_VOXCPM2_REFERENCE_IMPORT_BUNDLE"
        private static let outputKey = "COREAI_VOXCPM2_REFERENCE_IMPORT_OUTPUT"

        @Test(
            "VCM-04 An empty Mac Models store imports and fully inspects the sealed reference voice bundle",
            .enabled(if: env[inputKey] != nil))
        func realLocalImport() async throws {
            let inPath = try #require(Self.env[Self.inputKey])
            let outPath = try #require(Self.env[Self.outputKey])
            try #require(!inPath.isEmpty && !outPath.isEmpty)
            let source = URL(filePath: inPath, directoryHint: .isDirectory)
            let output = URL(filePath: outPath, directoryHint: .isDirectory)
            var relation = FileManager.URLRelationship.other
            try FileManager.default.getRelationship(
                &relation, ofDirectoryAt: source.resolvingSymlinksInPath(),
                toItemAt: output.deletingLastPathComponent().resolvingSymlinksInPath())
            try #require(relation == .other)
            guard mkdir(output.path, S_IRWXU) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
            let fixture = try VoxCPMImportFixture()
            defer { fixture.files.remove() }
            let catalog = try MacDemoCatalog.make()
            let model = try #require(catalog.model("voxcpm2"))
            let artifact = try #require(model.artifact("voxcpm2-reference_fp32"))
            let seal = "5beadfe31f8b37667f3e4ef42a8451b29dca1814a4cb53a253e96d9a2fdd1b51"
            try #require(artifact.files.count == 25 && artifact.sizeBytes == 9_541_520_509)
            try #require(
                artifact.sourceRevision == seal && artifact.sourceKind == .localManifest && artifact.download == nil)
            let before = try fixture.digests(source)
            let entries = Dictionary(uniqueKeysWithValues: artifact.files.map { ($0.path, $0.sha256) })
            let sourceEntries = before.filter { $0.value != "directory" }
            try #require(sourceEntries == entries)
            let store = ManagedModelStore(root: output.appending(path: "Models", directoryHint: .isDirectory))
            let prior = try await store.inspect(model: model, artifact: artifact)
            try #require(prior.isMissing)
            let staged = try await store.copyForImport(from: source, model: model, artifact: artifact) { _ in }
            let installed = try await store.promote(staging: staged, model: model, artifact: artifact)
            let full = try await store.inspect(model: model, artifact: artifact, verification: .full)
            let checked = try #require(full.resource)
            try #require(
                checked == installed && checked.directory != source && checked.matches(model: model, artifact: artifact)
            )
            let registration = try #require(await store.generation(of: checked.identity))
            let active = await store.activeLeases(identity: checked.identity)
            try #require(active == 0 && checked.sizeBytes == 9_541_520_509)
            let managed = try fixture.digests(checked.directory)
            let after = try fixture.digests(source)
            try #require(managed == before && after == before)
            let record: [String: Any] = [
                "version": 1, "complete": true, "manifestSHA256": seal, "modelID": checked.modelID,
                "artifactID": checked.artifactID, "fileCount": entries.count, "bytes": checked.sizeBytes,
                "registration": registration, "activeLeases": active, "sourceUnchanged": true,
                "managedHashesMatched": true,
            ]
            try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys, .prettyPrinted])
                .write(to: output.appending(path: "import.json"), options: .withoutOverwriting)
        }
    }
#endif
