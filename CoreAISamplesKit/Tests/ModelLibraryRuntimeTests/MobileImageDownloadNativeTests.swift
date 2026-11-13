import Darwin
import Foundation
import ImageGenerationRuntime
import ModelLibraryFeature
import ModelLibraryRuntime
import ModelLibraryStorage
import ModelLibraryTransfer
import Synchronization
import Testing

@MainActor
struct MobileImageDownloadNativeTests {
    @Test(
        "QD-05 The public image bundle downloads through Models with its pinned identity and without generation",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_MOBILE_IMAGE_DOWNLOAD_OUTPUT"] != nil),
        .timeLimit(.minutes(60)))
    func freshDownload() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_MOBILE_IMAGE_DOWNLOAD_OUTPUT"])
        let requested = URL(filePath: path, directoryHint: .isDirectory)
        let resolved = try #require(realpath(requested.deletingLastPathComponent().path, nil))
        defer { free(resolved) }
        let output = URL(filePath: String(cString: resolved)).appending(path: requested.lastPathComponent)
        try #require(mkdir(output.path, 0o700) == 0)
        let root = output.appending(path: "Models")
        try #require(!FileManager.default.fileExists(atPath: root.path))
        let catalog = try IOSDemoCatalog.make()
        let model = try #require(catalog.model("qwen-image21-turbo"))
        let artifact = try #require(model.artifact("mobile-image-lab-w4-w8-v1"))
        let source = try #require(artifact.download)
        try #require(
            source.repository == "okayuji/mobile-image-lab-coreai"
                && source.revision == "fdd2b8cb17f48976207039f1f780360ba60275dc" && source.root == "bundle")
        try #require(artifact.files.count == 51 && artifact.totalFileBytes == 13_783_495_057)
        try #require(
            artifact.sourceRevision == "9d61c180a8c72b67ba34e8ece6e842fec489dc98fc68b2b875fa07a1900dd3e5")
        let prefix = "https://huggingface.co/\(source.repository)/resolve/\(source.revision)/bundle/"
        let expected = Set(artifact.files.map { prefix + $0.path })
        let observed = Mutex<[String: Int]>([:])
        let transfer = ArtifactTransfer(fileLoader: { request, destination in
            let address = try #require(request.url?.absoluteString)
            try #require(expected.contains(address) && (request.allHTTPHeaderFields ?? [:]).isEmpty)
            let response = try await ArtifactTransfer.load(request, to: destination)
            let status = try #require((response as? HTTPURLResponse)?.statusCode)
            observed.withLock { $0[address] = status }
            return response
        })
        let storage = ManagedModelStore(root: root)
        let assembly = ModelLibraryAssembly(
            adapters: [], store: storage, transfer: transfer, migrationCandidates: { [] })
        let status = await ModelLibrarySelfTest.run(
            output: output.appending(path: "acquisition.json"), root: root, acquire: artifact.id,
            assembly: assembly,
            afterAcquisition: ModelLibrarySelfTest.AcquisitionCheck { context in
                guard context.store.state.session == nil else { return ["Acquisition created an image session"] }
                return []
            }, catalog: { catalog })
        try #require(status == 0)
        let responses = observed.withLock { $0 }
        try #require(Set(responses.keys) == expected && responses.values.allSatisfy { $0 == 200 })
        let inspection = try await storage.inspect(model: model, artifact: artifact, verification: .full)
        guard case .ready(let resource) = inspection else {
            Issue.record("The newly downloaded bundle did not pass full canonical inspection")
            return
        }
        try #require(resource.matches(model: model, artifact: artifact))
        try #require(resource.sourceKind == .localManifest && resource.sizeBytes == 13_783_495_057)
        try #require(resource.directory.path.hasPrefix(root.path + "/"))
        try await bindWithoutGeneration(resource)
        let lease = await storage.lease(identity: resource.identity, owner: UUID())
        do {
            try #require(await storage.activeLeases(identity: resource.identity) == 1)
            await #expect(throws: ManagedModelStoreError.inUse) {
                try await storage.delete(identity: resource.identity)
            }
        } catch {
            await lease.release()
            throw error
        }
        await lease.release()
        try #require(await storage.activeLeases(identity: resource.identity) == 0)
        let evidence: [String: Any] = [
            "format": "mobile-image-lab-download/1", "revision": source.revision,
            "manifestSHA256": resource.revision, "sourceKind": resource.sourceKind.rawValue,
            "artifactID": resource.artifactID, "bytes": resource.sizeBytes, "files": artifact.files.count,
            "fileRequests": responses, "freshManagedStore": true, "migrationCandidates": 0,
            "leasesAfterRelease": 0, "modelInferenceExecuted": false, "liveServiceBinding": true,
            "calibrationHardware": "iPhone18,1", "calibrationOSBuild": "24A446",
            "scope":
                "Production Models acquisition, canonical inspection and live-service binding on the host with explicit known phone calibration; no phone or image generation",
        ]
        try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys]).write(
            to: output.appending(path: "download.json"), options: .withoutOverwriting)
    }

    private func bindWithoutGeneration(_ resource: ValidatedModelResource) async throws {
        let calibration = try ImageDeviceCalibration.makeMobileImageLab()
        let budget = try #require(
            calibration.budget(for: resource, hardwareModel: "iPhone18,1", osBuild: "24A446"))
        let session = ModelDemoSession(
            id: LibraryTicket(owner: UUID(), number: 1),
            selection: DemoSelection(modelID: resource.modelID, demoID: "image-generation"), resource: resource)
        var service: LiveImageGenerationService?
        var permission: ManagedImageGeneration.Permission?
        var initialPermission: Bool?
        let managed = ManagedImageGeneration { bound, allowed in
            initialPermission = allowed()
            permission = allowed
            let created = LiveImageGenerationService(session: bound, budget: budget, isPermitted: allowed)
            service = created
            return created
        }
        let allowed: ManagedImageGeneration.Permission = { true }
        let bound = managed.store(for: session, isPermitted: allowed)
        let store = try #require(bound)
        let created = try #require(service)
        do {
            try #require(initialPermission == false && permission?() == true)
            try #require(created.session.resource == resource && !store.state.presented)
            let before = await created.latestReport()
            try #require(before == nil)
            try await managed.release(session)
            try #require(permission?() == false)
            let after = await created.latestReport()
            try #require(after == nil)
        } catch {
            try await managed.release(session)
            throw error
        }
    }
}
