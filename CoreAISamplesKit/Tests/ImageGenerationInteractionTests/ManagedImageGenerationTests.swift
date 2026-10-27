import Foundation
import ImageGenerationCore
import ImageGenerationRuntime
import ImageGenerationUI
import ModelLibraryRuntime
import Testing

@testable import ModelLibraryFeature

@MainActor
struct ManagedImageGenerationTests {
    private let permitted: ManagedImageGeneration.Permission = { true }
    private let denied: ManagedImageGeneration.Permission = { false }
    private func session(owner: UUID, number: Int) -> ModelDemoSession {
        ModelDemoSession(
            id: LibraryTicket(owner: owner, number: number),
            selection: DemoSelection(modelID: "image", demoID: "image-generation"),
            resource: ValidatedModelResource(
                modelID: "image", artifactID: "fixture", contract: "qwen-image21-mobile-v2",
                revision: String(repeating: "a", count: 64), upstreamRevision: ImageGenerationProfile.revision,
                directory: URL(filePath: "/fixture"), sizeBytes: 1, sourceKind: .localManifest))
    }

    @Test("A denied scene cannot create or release another scene's bound service")
    func deniedScene() async throws {
        var services: [ImageInteractionService] = []
        let managed = ManagedImageGeneration { _, _ in
            let service = ImageInteractionService()
            services.append(service)
            return service
        }
        let first = session(owner: UUID(), number: 1)
        let other = session(owner: UUID(), number: 1)
        #expect(managed.store(for: other, isPermitted: denied) == nil)
        #expect(services.isEmpty)
        let store = try #require(managed.store(for: first, isPermitted: permitted))
        #expect(!store.state.presented && services[0].requests.isEmpty)
        #expect(managed.store(for: other, isPermitted: denied) == nil)
        try await managed.release(other)
        #expect(services.count == 1 && services[0].releases == 0)
        try await managed.release(first)
        #expect(services[0].releases == 1)
    }

    @Test("Returning to a previous owner creates a fresh binding and delayed release cannot clear its replacement")
    func replacingBindings() async throws {
        var services: [ImageInteractionService] = []
        var permissions: [ManagedImageGeneration.Permission] = []
        let managed = ManagedImageGeneration { _, permission in
            let service = ImageInteractionService()
            services.append(service)
            permissions.append(permission)
            return service
        }
        let owner = UUID()
        let first = session(owner: owner, number: 1)
        let second = session(owner: UUID(), number: 1)
        let returning = session(owner: owner, number: 2)
        let firstStore = try #require(managed.store(for: first, isPermitted: permitted))
        firstStore.send(.appeared)
        firstStore.send(.setPrompt("Keep this prompt"))
        firstStore.send(.setSeed("47"))
        let secondStore = try #require(managed.store(for: second, isPermitted: permitted))
        #expect(!permissions[0]() && permissions[1]())
        #expect(firstStore !== secondStore && secondStore.state.result == nil)
        try await managed.release(first)
        #expect(services[0].releases == 1 && services[1].releases == 0 && permissions[1]())
        let returningStore = try #require(managed.store(for: returning, isPermitted: permitted))
        #expect(returningStore !== firstStore && returningStore !== secondStore)
        #expect(returningStore.state.owner != firstStore.state.owner)
        #expect(returningStore.state.prompt == "Keep this prompt" && returningStore.state.seed == "47")
        #expect(returningStore.state.result == nil && !permissions[1]() && permissions[2]())
        try await managed.release(second)
        #expect(services[2].releases == 0 && permissions[2]())
        try await managed.release(returning)
    }

    @Test("Unconfirmed cleanup retains its binding for a later explicit retry")
    func failedCleanup() async throws {
        let service = ImageInteractionService()
        service.stopFailure = true
        let managed = ManagedImageGeneration { _, _ in service }
        let session = session(owner: UUID(), number: 1)
        let store = try #require(managed.store(for: session, isPermitted: permitted))
        store.send(.appeared)
        store.send(.setPrompt("Retained input"))
        await #expect(throws: ImageExecutionError.self) { try await managed.release(session) }
        #expect(managed.store(for: session, isPermitted: permitted) === store)
        #expect(managed.state?.prompt == "Retained input")
        service.stopFailure = false
        try await managed.release(session)
        #expect(service.releases == 2)
    }

    @Test("The adapter rejects mismatched and denied gates before binding and tolerates missing device calibration")
    func adapterGates() async throws {
        var created = 0
        let service = ImageInteractionService()
        let adapter = ImageGenerationAdapter { _, _ in
            created += 1
            return service
        }
        let first = session(owner: UUID(), number: 1)
        let second = session(owner: UUID(), number: 1)
        _ = adapter.preview(for: first.selection)
        _ = adapter.content(for: first, gate: DemoExecutionGate(session: second, isPermitted: permitted))
        _ = adapter.content(for: first, gate: DemoExecutionGate(session: first, isPermitted: denied))
        #expect(created == 0)
        _ = adapter.content(for: first, gate: DemoExecutionGate(session: first, isPermitted: permitted))
        #expect(created == 1 && service.requests.isEmpty)
        try await adapter.stop(second)
        #expect(service.releases == 0)
        try await adapter.stop(first)
        #expect(service.releases == 1)
        let unsupported = ImageGenerationAdapter { _, _ in nil }
        _ = unsupported.content(for: first, gate: DemoExecutionGate(session: first, isPermitted: permitted))
        try await unsupported.stop(first)
    }
}
