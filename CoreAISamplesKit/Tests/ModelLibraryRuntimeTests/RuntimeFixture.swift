import CryptoKit
import FeatureStore
import Foundation
import ModelLibraryStorage
import ModelLibraryTransfer
import SwiftUI
import Synchronization
import Testing

@testable import ModelLibraryFeature
@testable import ModelLibraryRuntime

let distributionRevision = "0123456789abcdef0123456789abcdef01234567"
let upstreamRevision = "914f7f89142e33e77833254d9c9b90c3cef7303b"
let registeredDemos = ["comparison", "extraction", "inspection", "image-search", "object-finder"]

extension DemoSelection {
    static let comparison = DemoSelection(modelID: "text", demoID: "comparison")
    static let extraction = DemoSelection(modelID: "text", demoID: "extraction")
    static let inspection = DemoSelection(modelID: "text", demoID: "inspection")
    static let notes = DemoSelection(modelID: "text", demoID: "notes")
    static let imageSearch = DemoSelection(modelID: "image", demoID: "image-search")
    static let objectFinder = DemoSelection(modelID: "image", demoID: "object-finder")
    static let registered = [comparison, extraction, inspection, imageSearch, objectFinder]
}

struct MissingEvents: Error {
    let events: [String]
}

struct StopFailure: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func next(_ events: inout AsyncStream<String>.Iterator, _ wanted: String...) async throws {
    var remaining = wanted
    while !remaining.isEmpty, let event = await events.next(isolation: #isolation) {
        if let index = remaining.firstIndex(of: event) { remaining.remove(at: index) }
    }
    guard remaining.isEmpty else { throw MissingEvents(events: remaining) }
}

final class LedgerGate: Sendable {
    private let holding = Mutex(false)
    private let gate = DispatchSemaphore(value: 0)
    private let sink: AsyncStream<String>.Continuation

    init(sink: AsyncStream<String>.Continuation) {
        self.sink = sink
    }

    func hold() {
        holding.withLock { $0 = true }
    }

    func release() {
        gate.signal()
    }

    func write(_ data: Data, to url: URL) throws {
        let held = holding.withLock { value in
            defer { value = false }
            return value
        }
        if held {
            sink.yield("ledger")
            gate.wait()
        }
        try data.write(to: url, options: .atomic)
    }
}

@MainActor
final class OwnerLabels {
    var names: [UUID: String] = [:]

    func callAsFunction(_ owner: UUID?) -> String { owner.flatMap { names[$0] } ?? "?" }
}

@MainActor
final class FakeAdapter: ModelDemoAdapter {
    struct Entry {
        var input = ""
        var results: [String] = []
        var runtime: LibraryTicket?
        var gate: DemoExecutionGate?
    }

    let demoID: String
    private let label: OwnerLabels
    private let sink: AsyncStream<String>.Continuation
    private(set) var entries: [DemoSelection: Entry] = [:]
    private(set) var previews: [DemoSelection] = []
    private(set) var contents: [ModelDemoSession] = []
    private(set) var stops: [ModelDemoSession] = []
    private(set) var executions: [ModelDemoSession] = []
    private(set) var rejections = 0
    private(set) var loads = 0
    private(set) var saved: [String] = []
    var holdsStop = false
    var stopFailures: [String] = []
    private var heldStops: [CheckedContinuation<Void, Never>] = []
    private var heldSaves: [CheckedContinuation<Void, Never>] = []

    init(_ demoID: String, label: OwnerLabels, sink: AsyncStream<String>.Continuation) {
        self.demoID = demoID
        self.label = label
        self.sink = sink
    }

    func preview(for selection: DemoSelection) -> AnyView {
        previews.append(selection)
        return AnyView(EmptyView())
    }

    func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
        contents.append(session)
        entries[session.selection, default: Entry()].gate = gate
        return AnyView(EmptyView())
    }

    func gate(for selection: DemoSelection) -> DemoExecutionGate? { entries[selection]?.gate }

    @discardableResult
    func start(_ selection: DemoSelection, input: String) -> Bool {
        guard let gate = entries[selection]?.gate, gate.isPermitted() else {
            rejections += 1
            return false
        }
        if entries[selection]?.runtime != gate.session.id {
            loads += 1
            entries[selection]?.runtime = gate.session.id
        }
        executions.append(gate.session)
        entries[selection]?.input = input
        entries[selection]?.results.append(input + "@" + gate.session.identity.artifactID)
        return true
    }

    func save(_ selection: DemoSelection) async {
        let result = entries[selection]?.results.last ?? ""
        sink.yield(label(entries[selection]?.gate?.session.id.owner) + " saving " + demoID)
        await withCheckedContinuation { heldSaves.append($0) }
        saved.append(result)
    }

    func stop(_ session: ModelDemoSession) async throws {
        stops.append(session)
        sink.yield(label(session.id.owner) + " stopping " + demoID)
        if holdsStop { await withCheckedContinuation { heldStops.append($0) } }
        if !stopFailures.isEmpty { throw StopFailure(message: stopFailures.removeFirst()) }
        if entries[session.selection]?.runtime == session.id { entries[session.selection]?.runtime = nil }
    }

    func releaseStops() {
        let held = heldStops
        heldStops = []
        for continuation in held { continuation.resume() }
    }

    func completeSaves() {
        let held = heldSaves
        heldSaves = []
        for continuation in held { continuation.resume() }
    }
}

@MainActor
final class LibraryScene {
    let owner = UUID()
    let library: Store<ModelLibraryFeature>
    let rendering: ModelLibraryRendering
    private let label: String
    private let harness: RuntimeHarness

    init(_ label: String, harness: RuntimeHarness, capabilities: [DemoCapability: CapabilityState]) {
        self.label = label
        self.harness = harness
        let runner = harness.runner
        let environment = LibraryEnvironment(
            platform: .iOS, connectedDemos: harness.registry.connectedDemos, capabilities: capabilities)
        let sink = harness.sink
        library = Store(
            initialState: ModelLibraryState(catalog: harness.catalog, environment: environment, owner: owner)
        ) { effect, send in
            await runner.run(effect, send: send)
            sink.yield(label + " " + String("\(effect)".prefix { $0 != "(" }))
        }
        rendering = ModelLibraryRendering(registry: harness.registry, runner: runner, library: library)
        harness.labels.names[owner] = label
    }

    var runner: ModelLibraryRunner { harness.runner }
    var registry: ModelDemoAdapterRegistry { harness.registry }
    var adapters: [String: FakeAdapter] { harness.adapters }

    func adapter(_ demoID: String) throws -> FakeAdapter { try #require(adapters[demoID]) }

    func download(_ artifactID: String, _ events: inout AsyncStream<String>.Iterator) async throws {
        library.send(.downloadRequested(artifactID))
        try await next(&events, label + " download", label + " validate")
        #expect(library.state.installations[artifactID]?.phase == .ready)
    }

    @discardableResult
    func show() async throws -> ModelDemoSession {
        let session = try #require(library.state.session)
        _ = rendering.content(for: session)
        runner.provide(session)
        await runner.prepare(session)
        return session
    }
}

@MainActor
final class RuntimeHarness {
    let base = FileManager.default.temporaryDirectory.appending(path: "model-runtime-\(UUID().uuidString)")
    let store: ManagedModelStore
    let resources: ModelLibraryResourceRunner
    let transfer: ArtifactTransfer
    let catalog: ModelCatalog
    let events: AsyncStream<String>
    let sink: AsyncStream<String>.Continuation
    let ledger: LedgerGate
    let labels = OwnerLabels()
    let registry = ModelDemoAdapterRegistry()
    let adapters: [String: FakeAdapter]
    let runner: ModelLibraryRunner

    init() throws {
        var files: [String: Data] = [:]
        for name in [
            "text.aimodel/main.mlirb", "vision.aimodel/main.mlirb", "tokenizer.json", "tokenizer_config.json",
            "MODEL_LICENSE.txt", "token_embeddings.bf16",
        ] {
            files[name] = Data("fixture-\(name)".utf8)
        }
        let manifest: [String: Any] = [
            "format": 1, "revision": upstreamRevision, "precision": "fp16-safe-s128-v70",
            "files": files.mapValues(sha256),
        ]
        files["embedding-manifest.json"] = try JSONSerialization.data(withJSONObject: manifest, options: .sortedKeys)
        let served = files
        let transfer = ArtifactTransfer(
            fileLoader: { request, destination in
                let path = request.url?.absoluteString.components(separatedBy: "/resolve/\(distributionRevision)/").last
                guard let url = request.url, let data = served[path ?? ""],
                    let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)
                else { throw URLError(.fileDoesNotExist) }
                try data.write(to: destination)
                return response
            }, availableCapacity: { _ in .max })
        (events, sink) = AsyncStream.makeStream(of: String.self)
        let ledger = LedgerGate(sink: sink)
        self.ledger = ledger
        store = ManagedModelStore(
            root: base.appending(path: "ModelLibrary"), writeLedger: ledger.write, availableCapacity: { _ in .max })
        self.transfer = transfer
        resources = ModelLibraryResourceRunner(store: store, transfer: transfer)
        catalog = try Self.catalog(files)
        var adapters: [String: FakeAdapter] = [:]
        for demoID in registeredDemos {
            let adapter = FakeAdapter(demoID, label: labels, sink: sink)
            adapters[demoID] = adapter
            registry.register(adapter)
        }
        self.adapters = adapters
        runner = ModelLibraryRunner(
            resources: resources, registry: registry, store: store, retryDelays: [.zero, .zero, .zero])
    }

    func scene(_ label: String, capabilities: [DemoCapability: CapabilityState] = [:]) -> LibraryScene {
        LibraryScene(label, harness: self, capabilities: capabilities)
    }

    func leases(_ session: ModelDemoSession) async -> Int {
        await store.activeLeases(identity: session.identity)
    }

    func close() {
        ledger.release()
        for adapter in adapters.values {
            adapter.releaseStops()
            adapter.completeSaves()
        }
        try? FileManager.default.removeItem(at: base)
    }

    private static func catalog(_ files: [String: Data]) throws -> ModelCatalog {
        let definitions = files.map {
            ModelArtifactFile(path: $0.key, bytes: Int64($0.value.count), sha256: sha256($0.value))
        }
        .sorted { $0.path < $1.path }
        func artifact(_ id: String) -> ModelArtifact {
            ModelArtifact(
                id: id, title: id.uppercased(), contract: "embeddinggemma2-fp16-safe-s128-v70", platforms: [.iOS],
                download: ModelDownload(repository: "owner/fixture", revision: distributionRevision),
                upstream: ModelUpstream(repository: "fixture/source", revision: upstreamRevision), files: definitions)
        }
        let demos = (registeredDemos + ["notes"]).map {
            DemoDescriptor(
                id: $0, title: $0 + " demo", systemImage: "doc.text", summary: "Fixture " + $0, platforms: [.iOS])
        }
        let text = LibraryModel(
            id: "text", name: "Text model", category: "Fixture",
            artifacts: [artifact("text-fp16"), artifact("text-int4")],
            defaultArtifactID: "text-fp16", platforms: [.iOS],
            demos: ["comparison", "extraction", "inspection", "notes"].map {
                ModelDemoLink(demoID: $0, platforms: [.iOS])
            })
        let image = LibraryModel(
            id: "image", name: "Image model", category: "Fixture", artifacts: [artifact("image-fp16")],
            defaultArtifactID: "image-fp16", platforms: [.iOS],
            demos: [
                ModelDemoLink(demoID: "image-search", platforms: [.iOS]),
                ModelDemoLink(demoID: "object-finder", platforms: [.iOS], requiredCapabilities: [.camera]),
            ])
        return try ModelCatalog(models: [text, image], demos: demos)
    }
}
