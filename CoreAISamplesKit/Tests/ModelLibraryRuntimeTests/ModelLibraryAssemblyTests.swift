import FeatureStore
import Foundation
import ModelLibraryStorage
import ModelLibraryTransfer
import Synchronization
import Testing

@testable import ModelLibraryFeature
@testable import ModelLibraryRuntime

private final class Requests: Sendable {
    private let value = Mutex<[String]>([])

    func record(_ request: URLRequest) { value.withLock { $0.append(request.url?.absoluteString ?? "") } }
    var urls: [String] { value.withLock { $0 } }
}

private func addresses(_ artifact: ModelArtifact) -> Set<String> {
    guard let download = artifact.download else { return [] }
    let base = "https://huggingface.co/\(download.repository)/resolve/\(download.revision)/"
    return Set(artifact.files.map { base + (download.root.isEmpty ? "" : download.root + "/") + $0.path })
}

@MainActor
private final class AppAssembly {
    let base: URL
    let root: URL
    let legacy: [URL]
    let labels: OwnerLabels
    let requests: Requests
    let catalog: ModelCatalog
    let events: AsyncStream<String>
    let adapters: [String: FakeAdapter]
    let store: ManagedModelStore
    let assembly: ModelLibraryAssembly
    let library: ModelLibraryAssembly.Library

    init() throws {
        let base = FileManager.default.temporaryDirectory.appending(path: "model-assembly-\(UUID().uuidString)")
        self.base = base
        root = base.appending(path: "ModelLibrary")
        let legacy = ["gliner-fp32", "embeddinggemma2"].map { base.appending(path: "Documents/" + $0) }
        self.legacy = legacy
        for (folder, name) in zip(legacy, ["contract.json", "embedding-manifest.json"]) {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try Data("legacy-\(name)".utf8).write(to: folder.appending(path: name))
            try Data("legacy-weights".utf8).write(to: folder.appending(path: "model.bin"))
        }
        catalog = try IOSDemoCatalog.make()
        let (events, sink) = AsyncStream.makeStream(of: String.self)
        self.events = events
        let labels = OwnerLabels()
        self.labels = labels
        let fakes = catalog.demos.map { FakeAdapter($0.id, label: labels, sink: sink) }
        adapters = Dictionary(uniqueKeysWithValues: fakes.map { ($0.demoID, $0) })
        store = ManagedModelStore(root: root, availableCapacity: { _ in .max })
        let requests = Requests()
        self.requests = requests
        let transfer = ArtifactTransfer(
            fileLoader: { request, _ in
                requests.record(request)
                throw URLError(.notConnectedToInternet)
            }, availableCapacity: { _ in .max })
        assembly = ModelLibraryAssembly(
            adapters: fakes, store: store, transfer: transfer, migrationCandidates: { legacy })
        library = assembly.makeLibrary(catalog: catalog, capabilities: [:], owner: UUID())
    }

    func adapter(_ demoID: String) throws -> FakeAdapter { try #require(adapters[demoID]) }

    func legacyFiles() throws -> [String: Data] {
        var files: [String: Data] = [:]
        for folder in legacy {
            for path in try FileManager.default.subpathsOfDirectory(atPath: folder.path(percentEncoded: false)) {
                files[folder.lastPathComponent + "/" + path] = try Data(contentsOf: folder.appending(path: path))
            }
        }
        return files
    }

    func close() {
        for adapter in adapters.values {
            adapter.releaseStops()
            adapter.completeSaves()
        }
        try? FileManager.default.removeItem(at: base)
    }
}

@MainActor
private func wait(_ library: Store<ModelLibraryFeature>, until condition: (ModelLibraryState) -> Bool) async {
    for await state in Observations({ library.state }) where condition(state) { return }
}

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ModelLibraryAssemblyTests {
    @Test(
        "E2E-01 ADOPT-02 The app assembly connects one provider to each demo of the iOS catalog, passes the download request of every published artifact to the transfer with its catalog definition, and creates no session or provider work before the acquisition completes"
    )
    func publishedArtifacts() async throws {
        let app = try AppAssembly()
        defer { app.close() }
        let library = app.library.store
        let demoIDs = Set(app.catalog.demos.map(\.id))
        #expect(demoIDs == Set(registeredDemos) && app.adapters.count == demoIDs.count)
        #expect(app.assembly.registry.connectedDemos == demoIDs)
        #expect(library.state.environment.connectedDemos == demoIDs)
        let legacy = try app.legacyFiles()
        library.send(.refresh)
        await wait(library) { !$0.installations.values.contains(where: \.phase.isBusy) }
        #expect(library.state.installations.count == 3)
        #expect(library.state.installations.values.allSatisfy { $0 == ModelInstallation() })

        var opened: [DemoSelection] = []
        for section in library.state.sections {
            for demo in section.demos {
                let selection = DemoSelection(modelID: section.id, demoID: demo.id)
                library.send(.openDemo(selection))
                let shown = library.state
                _ = app.library.rendering.preview(for: selection)
                #expect(library.state == shown && shown.selection == selection && shown.session == nil)
                #expect(shown.access(to: selection) == .preview(.missing))
                library.send(.closeDemo)
                opened.append(selection)
            }
        }
        #expect(opened.map(\.demoID) == registeredDemos)

        var requested = 0
        for model in app.catalog.models {
            let selection = DemoSelection(modelID: model.id, demoID: try #require(model.demos.first).demoID)
            for artifact in model.artifacts {
                library.send(.selectArtifact(modelID: model.id, artifactID: artifact.id))
                library.send(.openDemo(selection))
                library.send(.downloadRequested(artifact.id))
                #expect(library.state.installations[artifact.id]?.phase == .downloading(nil))
                #expect(
                    library.state.session == nil && library.state.access(to: selection) == .preview(.downloading(nil)))
                await wait(library) { $0.installations[artifact.id]?.phase == .failed }
                #expect(library.state.session == nil && library.state.access(to: selection) == .preview(.failed))
                guard case .operation(let message)? = library.state.installations[artifact.id]?.failure else {
                    Issue.record("A failed transfer must be reported as an operation failure")
                    continue
                }
                #expect(message.hasPrefix("The connection was interrupted while downloading "))
                let urls = app.requests.urls.dropFirst(requested)
                requested += urls.count
                #expect(!urls.isEmpty && Set(urls).isSubset(of: addresses(artifact)))
                library.send(.closeDemo)
            }
        }
        for selection in opened {
            let adapter = try app.adapter(selection.demoID)
            #expect(adapter.previews == [selection] && adapter.contents.isEmpty && adapter.stops.isEmpty)
            #expect(adapter.executions.isEmpty && adapter.loads == 0 && adapter.rejections == 0)
        }
        #expect(try app.legacyFiles() == legacy)
        for name in ["artifacts", "ledger.json"] {
            #expect(!FileManager.default.fileExists(atPath: app.root.appending(path: name).path(percentEncoded: false)))
        }
    }

    @Test(
        "E2E-01 Direct operations of the app assembly with another owner's ticket, an artifact that was never acquired, or a session that is stopping or stopped take no lease and start no provider work"
    )
    func forgedContexts() async throws {
        let app = try AppAssembly()
        defer { app.close() }
        let library = app.library.store
        let runner = app.assembly.runner
        let adapter = try app.adapter("comparison")
        let model = try #require(app.catalog.model("gliner2.5-multi"))
        let artifact = try #require(model.artifact(model.defaultArtifactID))
        let selection = DemoSelection(modelID: model.id, demoID: "comparison")
        let resource = ValidatedModelResource(
            modelID: model.id, artifactID: artifact.id, contract: artifact.contract, revision: "",
            upstreamRevision: nil, directory: app.base.appending(path: "forged"), sizeBytes: 0)
        let owner = library.state.owner
        app.labels.names[owner] = "A"
        let foreign = ModelDemoSession(
            id: LibraryTicket(owner: UUID(), number: 1), selection: selection, resource: resource)
        let missing = ModelDemoSession(
            id: LibraryTicket(owner: owner, number: 1), selection: selection, resource: resource)
        library.send(.openDemo(selection))
        let shown = library.state
        #expect(shown.session == nil && shown.access(to: selection) == .preview(.missing))

        func attempt(_ session: ModelDemoSession, _ input: String) async {
            _ = app.library.rendering.content(for: session)
            runner.provide(session)
            await runner.prepare(session)
            #expect(!runner.gate(for: session, in: library).isPermitted())
            #expect(!adapter.start(selection, input: input))
            #expect(await app.store.activeLeases(identity: session.identity) == 0)
        }
        await attempt(foreign, "another owner")
        await attempt(missing, "never acquired")

        var events = app.events.makeAsyncIterator()
        let outcomes = Outcomes()
        adapter.holdsStop = true
        let ticket = LibraryTicket(owner: owner, number: 2)
        let stop = Task { await runner.run(.stopDemo(missing, ticket)) { outcomes.record($0) } }
        try await next(&events, "A stopping comparison")
        await attempt(missing, "while stopping")
        adapter.releaseStops()
        await stop.value
        #expect(outcomes.actions == ["stopped 2"])
        await attempt(missing, "after the stop")

        #expect(library.state == shown)
        #expect(adapter.executions.isEmpty && adapter.loads == 0 && adapter.rejections == 4)
        #expect(adapter.stops == [missing])
        #expect(
            app.adapters.values.filter { $0 !== adapter }.allSatisfy {
                $0.previews.isEmpty && $0.contents.isEmpty && $0.stops.isEmpty && $0.executions.isEmpty
            })
    }

    @Test(
        "E2E-01 With an acquired artifact the app assembly rejects another owner's ticket and operations during or after a stop while a valid session still runs"
    )
    func acquiredArtifact() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let assembly = ModelLibraryAssembly(
            adapters: Array(harness.adapters.values), store: harness.store, transfer: harness.transfer)
        let owner = UUID()
        harness.labels.names[owner] = "A"
        let library = assembly.makeLibrary(catalog: harness.catalog, capabilities: [:], owner: owner)
        let store = library.store
        let runner = assembly.runner
        let compare = try #require(harness.adapters["comparison"])
        let extract = try #require(harness.adapters["extraction"])
        var events = harness.events.makeAsyncIterator()
        func show(_ session: ModelDemoSession) async {
            _ = library.rendering.content(for: session)
            runner.provide(session)
            await runner.prepare(session)
        }

        store.send(.downloadRequested("text-fp16"))
        await wait(store) { $0.installations["text-fp16"]?.phase == .ready }
        store.send(.openDemo(DemoSelection.comparison))
        let first = try #require(store.state.session)
        await show(first)
        #expect(compare.start(DemoSelection.comparison, input: "valid"))
        #expect(await harness.leases(first) == 1)

        let foreign = ModelDemoSession(
            id: LibraryTicket(owner: UUID(), number: first.id.number), selection: first.selection,
            resource: first.resource)
        #expect(!store.state.permits(foreign.id, selection: foreign.selection))
        _ = compare.content(for: foreign, gate: runner.gate(for: foreign, in: store))
        #expect(!compare.start(DemoSelection.comparison, input: "another owner"))
        await show(first)
        #expect(compare.start(DemoSelection.comparison, input: "still valid"))

        let firstGate = try #require(compare.gate(for: DemoSelection.comparison))
        compare.holdsStop = true
        store.send(.openDemo(DemoSelection.extraction))
        try await next(&events, "A stopping comparison")
        #expect(store.state.session == first && store.state.access(to: DemoSelection.comparison) == .stopping)
        #expect(!firstGate.isPermitted())
        await show(first)
        #expect(!compare.start(DemoSelection.comparison, input: "while stopping"))
        #expect(await harness.leases(first) == 1)

        compare.releaseStops()
        await wait(store) { $0.session?.selection == DemoSelection.extraction }
        let second = try #require(store.state.session)
        await show(second)
        #expect(!firstGate.isPermitted() && !store.state.permits(first.id, selection: first.selection))
        await show(first)
        #expect(!compare.start(DemoSelection.comparison, input: "after the stop"))
        #expect(extract.start(DemoSelection.extraction, input: "next session"))
        #expect(await harness.leases(second) == 1)
        #expect(compare.executions == [first, first] && compare.stops == [first] && compare.rejections == 3)
        #expect(extract.executions == [second] && extract.rejections == 0)
    }

    @Test(
        "The model library self-test reports the iOS catalog, its five providers, the capability states and missing published artifacts, fails when a catalog demo has no provider, and leaves the iOS-only checks out without a registry"
    )
    func selfTest() async throws {
        let app = try AppAssembly()
        defer { app.close() }
        let output = app.base.appending(path: "selftest.json")
        func report() throws -> [String: Any] {
            try #require(JSONSerialization.jsonObject(with: Data(contentsOf: output)) as? [String: Any])
        }

        let status = await ModelLibrarySelfTest.run(
            output: output, root: app.root, registry: app.assembly.registry,
            capabilities: [.camera: .denied, .network: .available])
        #expect(status == 0)
        let passed = try report()
        let registry = passed["registry"] as? [String: Any]
        #expect(passed["failures"] as? [String] == [] && passed["catalogValidated"] as? Bool == true)
        #expect(registry?["connectedDemos"] as? [String] == registeredDemos.sorted())
        #expect(registry?["matchesCatalog"] as? Bool == true)
        #expect(passed["capabilities"] as? [String: String] == ["camera": "denied", "network": "available"])
        #expect(passed["visibleModels"] as? Int == 2 && passed["visibleDemos"] as? Int == 5)
        let artifacts = try #require(passed["artifacts"] as? [[String: Any]])
        let defined = app.catalog.models.flatMap(\.artifacts)
        #expect(artifacts.map { $0["artifactID"] as? String } == defined.map(\.id))
        #expect(
            artifacts.map { $0["distribution"] as? String }
                == defined.map { $0.download.map { "\($0.repository)@\($0.revision)" } })
        #expect(artifacts.map { $0["files"] as? Int } == defined.map(\.files.count))
        #expect(artifacts.allSatisfy { $0["inspection"] as? String == "missing" })
        #expect(passed["acquisition"] == nil)
        #expect((passed["store"] as? [String: Any])?["exists"] as? Bool == false)
        #expect(!FileManager.default.fileExists(atPath: app.root.path(percentEncoded: false)))

        let partial = ModelDemoAdapterRegistry()
        for adapter in app.adapters.values where adapter.demoID != "object-finder" { partial.register(adapter) }
        #expect(await ModelLibrarySelfTest.run(output: output, root: app.root, registry: partial) == 1)
        let failed = try report()
        #expect((failed["failures"] as? [String])?.count == 1)
        #expect((failed["registry"] as? [String: Any])?["matchesCatalog"] as? Bool == false)

        #expect(await ModelLibrarySelfTest.run(output: output, root: app.root) == 0)
        let catalogOnly = try report()
        #expect(catalogOnly["registry"] == nil && catalogOnly["capabilities"] == nil)
        #expect((catalogOnly["scope"] as? String)?.hasPrefix("Catalog and store checks only") == true)
        #expect(app.adapters.values.allSatisfy { $0.previews.isEmpty && $0.contents.isEmpty && $0.stops.isEmpty })
    }

    @Test(
        "ADOPT-02 The self-test with an acquisition target requests each catalog artifact in turn through the app assembly and fails with the route, defined bytes and inspection of every failed transfer"
    )
    func selfTestAcquisitionFailures() async throws {
        let app = try AppAssembly()
        defer { app.close() }
        let output = app.base.appending(path: "selftest.json")
        let status = await ModelLibrarySelfTest.run(
            output: output, root: app.root, registry: app.assembly.registry, capabilities: [:], acquire: "all",
            assembly: app.assembly)
        #expect(status == 1)
        let report = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: output)) as? [String: Any])
        let acquisition = try #require(report["acquisition"] as? [String: Any])
        #expect(acquisition["target"] as? String == "all")
        let results = try #require(acquisition["artifacts"] as? [[String: Any]])
        let artifacts = app.catalog.models.flatMap(\.artifacts)
        #expect(results.map { $0["artifactID"] as? String } == artifacts.map(\.id))
        for (result, artifact) in zip(results, artifacts) {
            #expect(result["route"] as? String == "download" && result["state"] as? String == "failed")
            #expect(result["revision"] as? String == artifact.download?.revision)
            #expect(result["inspection"] as? String == "missing")
            #expect(result["definedBytes"] as? Int64 == artifact.sizeBytes)
            #expect(result["measuredBytes"] == nil && result["identity"] == nil)
            #expect(
                (result["failure"] as? String)?.contains("The connection was interrupted while downloading") == true)
        }
        #expect((report["failures"] as? [String])?.count == artifacts.count)
        let owners = app.requests.urls.compactMap { url in artifacts.firstIndex { addresses($0).contains(url) } }
        #expect(owners.count == app.requests.urls.count)
        #expect(owners == owners.sorted() && Set(owners) == Set(artifacts.indices))
        #expect(app.adapters.values.allSatisfy { $0.previews.isEmpty && $0.contents.isEmpty && $0.executions.isEmpty })
    }

    @Test(
        "The self-test acquisition downloads a missing artifact, keeps an installed one and migrates a matching folder through the library store, with sizes equal to the definitions, the validated identity and a ready inspection"
    )
    func selfTestAcquisitionRoutes() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let root = harness.base.appending(path: "ModelLibrary")
        let assembly = ModelLibraryAssembly(adapters: [], store: harness.store, transfer: harness.transfer)
        let first = try await ModelLibrarySelfTest.check(
            catalog: harness.catalog, root: root, registry: nil, capabilities: nil, acquire: "text-fp16",
            assembly: assembly)
        #expect(first.failures.isEmpty && first.acquisition?.artifacts.count == 1)
        let downloaded = try #require(first.acquisition?.artifacts.first)
        #expect(downloaded.route == "download" && downloaded.state == "ready" && downloaded.inspection == "ready")
        #expect(downloaded.definedBytes > 0 && downloaded.measuredBytes == downloaded.definedBytes)
        #expect(downloaded.identity?.artifactID == "text-fp16" && downloaded.identity?.revision == distributionRevision)
        #expect(downloaded.identity?.upstreamRevision == upstreamRevision)
        #expect(downloaded.seconds != nil && downloaded.failure == nil)

        let legacy = harness.base.appending(path: "Documents/legacy")
        try FileManager.default.createDirectory(
            at: legacy.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(
            at: root.appending(path: "artifacts/text-fp16/" + distributionRevision), to: legacy)
        let migrating = ModelLibraryAssembly(
            adapters: [], store: harness.store, transfer: harness.transfer, migrationCandidates: { [legacy] })
        let second = try await ModelLibrarySelfTest.check(
            catalog: harness.catalog, root: root, registry: nil, capabilities: nil, acquire: "all",
            assembly: migrating)
        let results = try #require(second.acquisition?.artifacts)
        #expect(second.failures.isEmpty)
        #expect(results.map(\.artifactID) == ["text-fp16", "text-int4", "image-fp16"])
        #expect(results.map(\.route) == ["installed", "migration", "migration"])
        #expect(
            results.allSatisfy {
                $0.state == "ready" && $0.inspection == "ready" && $0.measuredBytes == $0.definedBytes
                    && $0.seconds == nil
            })

        let unknown = try await ModelLibrarySelfTest.check(
            catalog: harness.catalog, root: root, registry: nil, capabilities: nil, acquire: "absent",
            assembly: assembly)
        #expect(unknown.failures == ["acquire: absent names no catalog artifact"])
        #expect(unknown.acquisition?.artifacts.isEmpty == true)
    }
}
