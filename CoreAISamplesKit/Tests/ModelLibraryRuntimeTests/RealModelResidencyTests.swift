import EmbeddingCore
import EmbeddingRuntime
import FeatureStore
import Foundation
import ModelLibraryStorage
import ModelLibraryTransfer
import SwiftUI
import Testing

@testable import ModelLibraryFeature
@testable import ModelLibraryRuntime

@MainActor
@Suite(.serialized, .timeLimit(.minutes(10)))
struct RealModelResidencyTests {
    @Test(
        "A real retained embedding model blocks another owner until native unload completes",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_RESIDENCY_BUNDLE"] != nil))
    func retainsUnloadsAndHandsOff() async throws {
        let source = URL(
            filePath: try #require(ProcessInfo.processInfo.environment["COREAI_RESIDENCY_BUNDLE"]))
        let catalog = try IOSDemoCatalog.make()
        let model = try #require(catalog.model("embeddinggemma2"))
        let artifact = try #require(model.artifact("embeddinggemma2-fp16-safe-s128-v70"))
        let base = FileManager.default.temporaryDirectory.appending(path: "real-residency-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }
        let storage = ManagedModelStore(root: base)
        let staging = try await storage.stagingDirectory(for: model, artifact: artifact)
        for file in artifact.files {
            let destination = staging.appending(path: file.path)
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.linkItem(at: source.appending(path: file.path), to: destination)
        }
        let resource = try await storage.promote(staging: staging, model: model, artifact: artifact)
        let adapter = RetainedEmbeddingAdapter()
        let registry = ModelDemoAdapterRegistry()
        registry.register(adapter)
        let runner = ModelLibraryRunner(
            resources: ModelLibraryResourceRunner(store: storage, transfer: ArtifactTransfer()),
            registry: registry, store: storage)
        func scene() -> Store<ModelLibraryFeature> {
            var state = ModelLibraryState(
                catalog: catalog, environment: LibraryEnvironment(platform: .iOS, connectedDemos: ["image-search"]),
                owner: UUID())
            var installation = ModelInstallation()
            installation.phase = .ready
            installation.resource = resource
            state.installations[artifact.id] = installation
            return Store(initialState: state) { effect, send in await runner.run(effect, send: send) }
        }
        func show(_ library: Store<ModelLibraryFeature>) async throws -> ModelDemoSession {
            let session = try #require(library.state.session)
            _ = runner.gate(for: session, in: library)
            library.send(.requestDemoPreparation(session.id))
            await awaitDemoState(library) {
                $0.demoPreparation == .available || $0.demoPreparation == .occupied
            }
            if let gate = runner.bind(session, in: library) {
                _ = adapter.content(for: session, gate: gate)
            }
            return session
        }
        let first = scene()
        let second = scene()
        let selection = DemoSelection(modelID: model.id, demoID: "image-search")
        first.send(.openDemo(selection))
        let owner = try await show(first)
        var samples = [ResidencyMemory.capture("before-load")]
        let firstVector = try await adapter.encode("a red apple")
        weak let firstModel = adapter.model
        #expect(firstModel != nil)
        samples.append(.capture("request-completed-model-retained"))
        #expect(await storage.activeLeases(identity: resource.identity) == 1)
        second.send(.openDemo(selection))
        let nextOwner = try await show(second)
        #expect(second.state.demoPreparation == .occupied)
        #expect(adapter.bindings == [owner.id])
        #expect(adapter.loads == 1)
        #expect(try await adapter.encode("a red apple") == firstVector)
        first.send(.closeDemo)
        await awaitDemoState(first) { $0.session == nil }
        #expect(adapter.unloaded == [owner.id])
        #expect(adapter.model == nil && firstModel == nil)
        #expect(await storage.activeLeases(identity: resource.identity) == 0)
        samples.append(.capture("after-native-unload-and-stop-ack"))
        #expect(try await show(second) == nextOwner)
        #expect(try await adapter.encode("a red apple") == firstVector)
        #expect(adapter.loads == 2 && adapter.bindings == [owner.id, nextOwner.id])
        samples.append(.capture("second-owner-reloaded"))
        await runner.run(.stopDemo(owner, LibraryTicket(owner: owner.id.owner, number: 999))) { _ in }
        #expect(try await adapter.encode("a red apple") == firstVector)
        #expect(await storage.activeLeases(identity: resource.identity) == 1)
        second.send(.closeDemo)
        await awaitDemoState(second) { $0.session == nil }
        #expect(adapter.unloaded == [owner.id, nextOwner.id])
        #expect(await storage.activeLeases(identity: resource.identity) == 0)
        samples.append(.capture("second-owner-unloaded"))
        if let destination = ProcessInfo.processInfo.environment["COREAI_RESIDENCY_METRICS"] {
            let report = ResidencyReport(
                artifactID: artifact.id, revision: resource.revision, loads: adapter.loads,
                nativeUnloads: adapter.unloaded.count, samples: samples)
            try JSONEncoder().encode(report).write(to: URL(filePath: destination), options: .atomic)
        }
    }
}

@MainActor
private final class RetainedEmbeddingAdapter: ModelDemoAdapter {
    let demoID = "image-search"
    private(set) var model: EmbeddingModel?
    private(set) var gate: DemoExecutionGate?
    private(set) var bindings: [LibraryTicket] = []
    private(set) var unloaded: [LibraryTicket] = []
    private(set) var loads = 0

    func preview(for selection: DemoSelection) -> AnyView { AnyView(EmptyView()) }

    func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
        guard gate.session == session, gate.isPermitted() else { return AnyView(EmptyView()) }
        self.gate = gate
        bindings.append(session.id)
        return AnyView(EmptyView())
    }

    func encode(_ text: String) async throws -> EmbeddingVector {
        guard let gate, gate.isPermitted() else { throw EmbeddingError.notLoaded }
        if model == nil {
            let created = EmbeddingModel()
            model = created
            loads += 1
            try await created.load(directory: gate.session.resource.directory)
        }
        return try await #require(model).encode(text)
    }

    func stop(_ session: ModelDemoSession) async throws {
        guard gate?.session == session, let model else { return }
        try await model.unload()
        await #expect(throws: EmbeddingError.notLoaded) { try await model.encode("unloaded") }
        unloaded.append(session.id)
        self.model = nil
        gate = nil
    }
}
