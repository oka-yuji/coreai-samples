import FeatureStore
import ModelLibraryFeature
import ModelLibraryUI
import SwiftUI

@main
struct ModelLibraryPreviewApp: App {
    @State private var driver: PreviewDriver
    @State private var store: Store<ModelLibraryFeature>
    @State private var renderer: PreviewRenderer

    init() {
        do {
            let catalog = try PreviewCatalog.make()
            let driver = PreviewDriver(catalog: catalog)
            _driver = State(initialValue: driver)
            _renderer = State(initialValue: PreviewRenderer(catalog: catalog))
            _store = State(
                initialValue: Store(initialState: ModelLibraryState(catalog: catalog, platform: .iOS), run: driver.run))
        } catch { preconditionFailure(String(describing: error)) }
    }

    var body: some Scene {
        WindowGroup {
            VStack(spacing: 0) {
                HStack {
                    Text("UI preview · no real models").font(.caption)
                    Spacer()
                    Menu("Transfer controls") {
                        Button("Complete pending downloads") { driver.completeDownloads() }
                        Button("Fail pending downloads") { driver.failDownloads() }
                    }
                    .font(.caption)
                }
                .padding(10)
                .background(.yellow.opacity(0.15))
                ModelLibraryView(store: store, renderer: renderer)
            }
        }
    }
}

@MainActor
final class PreviewDriver {
    let catalog: ModelCatalog
    private var installed: Set<String> = ["sample-text"]
    private var pending: [String: CheckedContinuation<URL, any Error>] = [:]

    init(catalog: ModelCatalog) { self.catalog = catalog }

    func run(_ effect: ModelLibraryEffect, send: (ModelLibraryAction) -> Void) async {
        switch effect {
        case .inspect(let model, let request):
            send(.inspected(model.id, request, installed.contains(model.id) ? resource(model) : nil))
        case .download(let model, let request):
            do {
                let folder = try await withCheckedThrowingContinuation { pending[model.id] = $0 }
                try Task.checkCancellation()
                send(.downloaded(model.id, request, folder))
            } catch {
                if !Task.isCancelled { send(.failed(model.id, request, .operation("Preview transfer failure"))) }
            }
        case .validate(let model, let request, _):
            send(.validated(model.id, request, resource(model)))
        case .stopTransfer(let modelID, let request):
            pending.removeValue(forKey: modelID)?.resume(throwing: CancellationError())
            send(.transferStopped(modelID, request))
        case .delete(let resource, let request):
            installed.remove(resource.modelID)
            send(.deleted(resource.modelID, request))
        case .stopDemo(let session, let request):
            send(.demoStopped(session.id, request))
        }
    }

    func completeDownloads() {
        let pending = pending
        self.pending.removeAll()
        for (id, continuation) in pending {
            installed.insert(id)
            continuation.resume(returning: URL(filePath: "/preview/" + id))
        }
    }

    func failDownloads() {
        let pending = pending
        self.pending.removeAll()
        for continuation in pending.values { continuation.resume(throwing: CancellationError()) }
    }

    private func resource(_ model: LibraryModel) -> ValidatedModelResource {
        ValidatedModelResource(
            modelID: model.id, artifactID: model.artifact.id, contract: model.artifact.contract,
            directory: URL(filePath: "/preview/" + model.id), sizeBytes: 1_000)
    }
}

private enum PreviewCatalog {
    static func make() throws -> ModelCatalog {
        let demos = (1...10).map { index in
            DemoDescriptor(
                id: "demo-\(index)", title: "Demo \(index)", systemImage: "doc.text",
                summary: "Inspect the native library layout and its shared availability controls.",
                explanation: [DemoExplanation(title: "Preview", text: "This sample never loads real model files.")],
                platforms: [.iOS])
        }
        let models = ["sample-text", "sample-image"].map { id in
            LibraryModel(
                id: id, name: id == "sample-text" ? "Sample text model" : "Sample image model",
                category: id == "sample-text" ? "Information Extraction" : "Embedding",
                artifact: ModelArtifact(
                    id: id + "-artifact", contract: "preview-v1", sizeBytes: 1_000,
                    download: ModelDownload(repository: "preview/model", revision: String(repeating: "a", count: 40))),
                platforms: [.iOS],
                demos: demos.map { ModelDemoLink(demoID: $0.id, platforms: [.iOS]) })
        }
        return try ModelCatalog(models: models, demos: demos)
    }
}
