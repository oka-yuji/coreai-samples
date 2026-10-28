import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer
import OCRFeature
import SwiftUI
import Testing

@testable import ModelLibraryRuntime
@testable import OCRDemoRuntime

@MainActor
struct TeleOCRMacIntegrationTests {
    @Test(
        "The Mac Models import, preparation, OCR session and release read the fixed page with the real model",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_OCR_MODELS_BUNDLE"] != nil),
        .timeLimit(.minutes(5)))
    func importedPage() async throws {
        let environment = ProcessInfo.processInfo.environment
        let source = URL(filePath: try #require(environment["COREAI_OCR_MODELS_BUNDLE"]))
        let image = URL(filePath: try #require(environment["COREAI_OCR_MODELS_IMAGE"]))
        let reference = URL(filePath: try #require(environment["COREAI_OCR_MODELS_MARKDOWN"]))
        let root = FileManager.default.temporaryDirectory.appending(path: "ocr-models-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let storage = ManagedModelStore(root: root)
        let artifact = try TeleOCRModelArtifact.make()
        let actions = AsyncStream<(LibraryTicket, OCRAction)>.makeStream()
        let adapter = IntegratedOCRAdapter(actions: actions.continuation)
        let assembly = ModelLibraryAssembly(adapters: [adapter], store: storage, transfer: ArtifactTransfer())
        let catalog = try TeleOCRDemoCatalog.make(artifact: artifact)
        let effects = AsyncStream<ModelLibraryEffect>.makeStream()
        let library = Store<ModelLibraryFeature>(
            initialState: ModelLibraryState(
                catalog: catalog, environment: LibraryEnvironment(platform: .macOS, connectedDemos: ["ocr"]),
                owner: UUID())
        ) { effect, send in
            await assembly.runner.run(effect, send: send)
            effects.continuation.yield(effect)
        }
        let rendering = ModelLibraryRendering(registry: assembly.registry, runner: assembly.runner, library: library)
        var completions = effects.stream.makeAsyncIterator()
        library.send(.importRequested(artifact.id, source))
        while let effect = await completions.next() {
            if case .importBundle = effect { break }
        }
        let resource = try #require(library.state.installations[artifact.id]?.resource)
        #expect(library.state.installations[artifact.id]?.phase == .ready)
        library.send(.openDemo(TeleOCRDemoCatalog.selection))
        let session = try #require(library.state.session)
        do {
            _ = rendering.content(for: session)
            library.send(.requestDemoPreparation(session.id))
            while let effect = await completions.next() {
                if case .prepareDemo = effect { break }
            }
            #expect(library.state.demoPreparation == .available)
            _ = rendering.content(for: session)
            let reader = try #require(adapter.store)
            #expect(await storage.activeLeases(identity: resource.identity) == 1)
            reader.send(.selectImage(image))
            var received = actions.stream.makeAsyncIterator()
            while let event = await received.next() {
                if event.0 == session.id {
                    switch event.1 {
                    case .selected, .failed, .cancelled: break
                    default: continue
                    }
                    break
                }
            }
            #expect(reader.state.canRead)
            reader.send(.read)
            while let event = await received.next() {
                if event.0 == session.id {
                    switch event.1 {
                    case .completed, .failed, .cancelled: break
                    default: continue
                    }
                    break
                }
            }
            let document = try #require(reader.state.document)
            #expect(document.isComplete && document.blocks.count == 3)
            #expect(document.image == image.lastPathComponent)
            #expect(try String(contentsOf: reference, encoding: .utf8) == document.markdown + "\n")
        } catch {
            await withTaskCancellationShield {
                await assembly.runner.run(.stopDemo(session, session.id), send: { _ in })
            }
            throw error
        }
        library.send(.closeDemo)
        while let effect = await completions.next() {
            if case .stopDemo = effect { break }
        }
        await withTaskCancellationShield {
            await assembly.runner.run(.stopDemo(session, session.id), send: { _ in })
        }
        #expect(library.state.session == nil && library.state.pendingDestination == nil)
        #expect(await storage.activeLeases(identity: resource.identity) == 0)
        #expect(try await storage.delete(identity: resource.identity) == resource.sizeBytes)
        actions.continuation.finish()
        effects.continuation.finish()
    }
}

@MainActor
private final class IntegratedOCRAdapter: ModelDemoAdapter {
    let demoID = "ocr"
    private let sessions: OCRDemoSessions
    var store: Store<OCRFeature>?

    init(actions: AsyncStream<(LibraryTicket, OCRAction)>.Continuation) {
        sessions = OCRDemoSessions(makeService: { LiveOCRService() }, observe: { actions.yield(($0, $1)) })
    }
    func preview(for selection: DemoSelection) -> AnyView { AnyView(EmptyView()) }
    func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
        guard gate.session == session, gate.isPermitted() else { return preview(for: session.selection) }
        store = sessions.bind(session, permits: gate.isPermitted)
        return AnyView(EmptyView())
    }
    func stop(_ session: ModelDemoSession) async throws { await sessions.stop(session) }
}
