import FeatureStore
import ModelLibraryFeature
import ModelLibraryUI
import SwiftUI

@main
struct ModelLibraryPreviewApp: App {
    var body: some Scene {
        WindowGroup { PreviewScreen() }
    }
}

private struct PreviewScreen: View {
    @State private var driver: PreviewDriver
    @State private var store: Store<ModelLibraryFeature>
    @State private var renderer: PreviewRenderer
    @State private var holdsStops = false

    init() {
        do {
            let catalog = try PreviewCatalog.make()
            let driver = PreviewDriver(catalog: catalog)
            _driver = State(initialValue: driver)
            _renderer = State(initialValue: PreviewRenderer(catalog: catalog))
            let environment = LibraryEnvironment(platform: .iOS, connectedDemos: Set(catalog.demos.map(\.id)))
            _store = State(
                initialValue: Store(
                    initialState: ModelLibraryState(catalog: catalog, environment: environment, owner: UUID()),
                    run: driver.run))
        } catch { preconditionFailure(String(describing: error)) }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("UI preview · no real models").font(.caption)
                Spacer()
                Menu("Preview controls") {
                    Button("Complete pending downloads") { driver.completeDownloads() }
                    Button("Fail pending downloads") { driver.failDownloads() }
                    Toggle("Hold demo stops", isOn: $holdsStops)
                    Button("Complete pending stops") { driver.resolveStops(success: true) }
                    Button("Fail pending stops") { driver.resolveStops(success: false) }
                }
                .font(.caption)
            }
            .padding(10)
            .background(.yellow.opacity(0.15))
            ModelLibraryView(store: store, renderer: renderer)
        }
        .onChange(of: holdsStops) { _, value in driver.holdsStops = value }
    }
}

@MainActor
final class PreviewDriver {
    let catalog: ModelCatalog
    private var installed: Set<String> = ["sample-text-fp32"]
    private var pending: [String: CheckedContinuation<URL, any Error>] = [:]
    var holdsStops = false
    private var pendingStops: [LibraryTicket: AsyncStream<Bool>.Continuation] = [:]

    init(catalog: ModelCatalog) { self.catalog = catalog }

    func run(_ effect: ModelLibraryEffect, send: (ModelLibraryAction) -> Void) async {
        switch effect {
        case .inspect(let model, let artifact, let request):
            send(.inspected(artifact.id, request, installed.contains(artifact.id) ? resource(model, artifact) : nil))
        case .download(_, let artifact, let request):
            do {
                let folder = try await withCheckedThrowingContinuation { pending[artifact.id] = $0 }
                try Task.checkCancellation()
                send(.downloaded(artifact.id, request, folder))
            } catch {
                if !Task.isCancelled { send(.failed(artifact.id, request, .operation("Preview transfer failure"))) }
            }
        case .validate(let model, let artifact, let request, _):
            send(.validated(artifact.id, request, resource(model, artifact)))
        case .importBundle(_, let artifact, let request, _):
            send(.failed(artifact.id, request, .operation("This fixture does not import real model files.")))
        case .stopImport(let model, let artifact, let request):
            send(
                .importStopped(artifact.id, request, installed.contains(artifact.id) ? resource(model, artifact) : nil))
        case .stopTransfer(let artifactID, let request):
            pending.removeValue(forKey: artifactID)?.resume(throwing: CancellationError())
            send(.transferStopped(artifactID, request))
        case .delete(let resource, let request):
            installed.remove(resource.artifactID)
            send(.deleted(resource.artifactID, request))
        case .stopDemo(let session, let request):
            if holdsStops {
                let channel = AsyncStream<Bool>.makeStream(bufferingPolicy: .bufferingNewest(1))
                pendingStops[request] = channel.continuation
                defer {
                    pendingStops.removeValue(forKey: request)
                    channel.continuation.finish()
                }
                var iterator = channel.stream.makeAsyncIterator()
                guard let success = await iterator.next(), !Task.isCancelled else { return }
                if success {
                    send(.demoStopped(session.id, request))
                } else {
                    send(.demoStopFailed(session.id, request, "Preview stop failure"))
                }
            } else {
                send(.demoStopped(session.id, request))
            }
        case .prepareDemo(let session):
            send(.demoPreparationChanged(session.id, .available))
        case .acquireUsage(let identity, let request):
            let model = catalog.model(identity.modelID)
            let artifact = model?.artifacts.first { $0.id == identity.artifactID }
            let value = model.flatMap { model in
                artifact.flatMap { installed.contains($0.id) ? resource(model, $0) : nil }
            }
            send(.usageAcquired(identity.modelID, request, value))
        case .releaseUsage:
            break
        }
    }

    func resolveStops(success: Bool) {
        let pending = pendingStops
        pendingStops.removeAll()
        for continuation in pending.values {
            continuation.yield(success)
            continuation.finish()
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

    private func resource(_ model: LibraryModel, _ artifact: ModelArtifact) -> ValidatedModelResource {
        ValidatedModelResource(
            modelID: model.id, artifactID: artifact.id, contract: artifact.contract,
            revision: artifact.download?.revision ?? "", upstreamRevision: artifact.upstream?.revision,
            directory: URL(filePath: "/preview/" + artifact.id), sizeBytes: 1_000)
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
        let models = [("sample-text", ["FP32"]), ("sample-image", ["FP16", "INT8"])].map { id, variants in
            let artifacts = variants.map { title in
                ModelArtifact(
                    id: id + "-" + title.lowercased(), title: title, contract: "preview-v1", platforms: [.iOS],
                    sizeBytes: 1_000,
                    download: ModelDownload(repository: "preview/model", revision: String(repeating: "a", count: 40)),
                    files: [
                        ModelArtifactFile(path: "model.bin", bytes: 1_000, sha256: String(repeating: "0", count: 64))
                    ])
            }
            return LibraryModel(
                id: id, name: id == "sample-text" ? "Sample text model" : "Sample image model",
                category: id == "sample-text" ? "Information Extraction" : "Embedding",
                artifacts: artifacts, defaultArtifactID: id + "-" + variants[0].lowercased(), platforms: [.iOS],
                demos: demos.map { ModelDemoLink(demoID: $0.id, platforms: [.iOS]) })
        }
        return try ModelCatalog(models: models, demos: demos)
    }
}
