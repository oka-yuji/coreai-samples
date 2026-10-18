import FeatureStore
import Foundation
import ModelLibraryStorage
import ModelLibraryTransfer
import Synchronization
import Testing

@testable import ModelLibraryFeature
@testable import ModelLibraryRuntime

private final class RequestCount: Sendable {
    private let value = Mutex(0)

    func increment() { value.withLock { $0 += 1 } }
    var count: Int { value.withLock { $0 } }
}

@MainActor
private final class AppAssembly {
    let base: URL
    let root: URL
    let legacy: [URL]
    let labels: OwnerLabels
    let requests: RequestCount
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
        let requests = RequestCount()
        self.requests = requests
        let transfer = ArtifactTransfer(
            fileLoader: { _, _ in
                requests.increment()
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
private func settle(_ library: Store<ModelLibraryFeature>) async {
    for await state in Observations({ library.state })
    where !state.installations.values.contains(where: \.phase.isBusy) {
        return
    }
}

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ModelLibraryAssemblyTests {
    @Test(
        "E2E-01 The app assembly connects one provider to each demo of the iOS catalog, and browsing, refreshing and download requests for its unconfigured artifacts create no session, call no provider and leave legacy folders unchanged"
    )
    func unconfiguredArtifacts() async throws {
        let app = try AppAssembly()
        defer { app.close() }
        let library = app.library.store
        let demoIDs = Set(app.catalog.demos.map(\.id))
        #expect(demoIDs == Set(registeredDemos) && app.adapters.count == demoIDs.count)
        #expect(app.assembly.registry.connectedDemos == demoIDs)
        #expect(library.state.environment.connectedDemos == demoIDs)
        let legacy = try app.legacyFiles()
        library.send(.refresh)
        await settle(library)
        #expect(library.state.installations.count == 2)
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
                let artifactID = section.model.defaultArtifactID
                library.send(.downloadRequested(artifactID))
                #expect(library.state.installations[artifactID]?.failure == .downloadNotConfigured)
                #expect(library.state.installations[artifactID]?.phase == .missing && library.state.session == nil)
                library.send(.closeDemo)
                opened.append(selection)
            }
        }
        #expect(opened.map(\.demoID) == registeredDemos)
        for selection in opened {
            let adapter = try app.adapter(selection.demoID)
            #expect(adapter.previews == [selection] && adapter.contents.isEmpty && adapter.stops.isEmpty)
            #expect(adapter.executions.isEmpty && adapter.loads == 0 && adapter.rejections == 0)
        }
        #expect(app.requests.count == 0)
        #expect(try app.legacyFiles() == legacy)
        #expect(!FileManager.default.fileExists(atPath: app.root.path(percentEncoded: false)))
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
        "The model library self-test reports the iOS catalog, its five providers, the capability states and missing unconfigured artifacts, fails when a catalog demo has no provider, and leaves the iOS-only checks out without a registry"
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
        #expect(
            artifacts.map { $0["artifactID"] as? String } == [
                "gliner2.5-multi-unconfigured", "embeddinggemma2-unconfigured",
            ])
        #expect(artifacts.allSatisfy { $0["inspection"] as? String == "missing" && $0["distribution"] == nil })
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
}
