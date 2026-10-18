import CryptoKit
import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer
import Synchronization

let revision = "0123456789abcdef0123456789abcdef01234567"
let upstream = ModelUpstream(
    repository: "fixture/embedding-source", revision: "914f7f89142e33e77833254d9c9b90c3cef7303b")

enum Event: Equatable, Sendable {
    case began(String)
    case ended(String)
    case sent(String)
    case started(String)
    case cancelled(String)
    case closed(String)
    case ledger
}

struct Missing: Error {
    let events: [Event]
}

func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func name(_ value: some Any) -> String {
    String("\(value)".prefix { $0 != "(" })
}

func temporaryDirectory() -> URL {
    FileManager.default.temporaryDirectory.appending(path: "model-transfer-\(UUID().uuidString)")
}

func next(_ events: inout AsyncStream<Event>.Iterator, _ wanted: Event...) async throws -> [Event] {
    try await next(&events, all: wanted)
}

func next(_ events: inout AsyncStream<Event>.Iterator, all wanted: [Event]) async throws -> [Event] {
    var remaining = wanted
    var seen: [Event] = []
    while !remaining.isEmpty, let event = await events.next(isolation: #isolation) {
        seen.append(event)
        if let index = remaining.firstIndex(of: event) { remaining.remove(at: index) }
    }
    guard remaining.isEmpty else { throw Missing(events: remaining) }
    return seen
}

func precedes(_ first: Event, _ second: Event, in events: [Event]) -> Bool {
    guard let earlier = events.firstIndex(of: first), let later = events.firstIndex(of: second) else { return false }
    return earlier < later
}

func files(in directory: URL) -> Set<String> {
    var paths: Set<String> = []
    guard let entries = FileManager.default.enumerator(atPath: directory.path) else { return paths }
    while let path = entries.nextObject() as? String {
        if entries.fileAttributes?[.type] as? FileAttributeType != .typeDirectory { paths.insert(path) }
    }
    return paths
}

func write(_ files: [String: Data], to directory: URL) throws {
    for (path, data) in files {
        let file = directory.appending(path: path)
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: file)
    }
}

struct Sample {
    let files: [String: Data]
    let artifact: ModelArtifact
    let model: LibraryModel
    let demo = DemoSelection(modelID: "embedding", demoID: "image-search")

    init(
        root: String = "", files: [String: Data]? = nil, contract: String = "embeddinggemma2-fp16-safe-s128-v70",
        pinned: String = revision
    ) throws {
        let files = try files ?? Self.embeddingFiles()
        self.files = files
        artifact = ModelArtifact(
            id: "embedding-fp16", title: "FP16", contract: contract, platforms: [.iOS],
            download: ModelDownload(repository: "owner/fixture", revision: pinned, root: root), upstream: upstream,
            files: files.map { ModelArtifactFile(path: $0.key, bytes: Int64($0.value.count), sha256: sha256($0.value)) }
                .sorted { $0.path < $1.path })
        model = LibraryModel(
            id: "embedding", name: "EmbeddingGemma 2", category: "Fixture", artifacts: [artifact],
            defaultArtifactID: artifact.id, platforms: [.iOS],
            demos: [ModelDemoLink(demoID: "image-search", platforms: [.iOS])])
    }

    var id: String { artifact.id }
    var total: Int64 { artifact.files.reduce(0) { $0 + $1.bytes } }

    func catalog() throws -> ModelCatalog {
        try ModelCatalog(
            models: [model],
            demos: [
                DemoDescriptor(
                    id: "image-search", title: "Image search", systemImage: "photo", summary: "Fixture",
                    platforms: [.iOS])
            ])
    }

    func url(_ path: String) -> String {
        let unreserved = CharacterSet(
            charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
        let parts = ((artifact.download?.root ?? "").split(separator: "/") + path.split(separator: "/"))
            .map { $0.addingPercentEncoding(withAllowedCharacters: unreserved) ?? "" }
        return "https://huggingface.co/owner/fixture/resolve/\(revision)/" + parts.joined(separator: "/")
    }

    static func embeddingFiles() throws -> [String: Data] {
        var files: [String: Data] = [:]
        for name in [
            "text.aimodel/main.hash", "text.aimodel/main.mlirb", "text.aimodel/metadata.json",
            "vision.aimodel/main.hash", "vision.aimodel/main.mlirb", "vision.aimodel/metadata.json",
            "tokenizer.json", "tokenizer_config.json", "MODEL_LICENSE.txt", "token_embeddings.bf16",
        ] {
            files[name] = Data("fixture-fp16-\(name)".utf8)
        }
        let manifest: [String: Any] = [
            "format": 1, "revision": upstream.revision, "precision": "fp16-safe-s128-v70",
            "files": files.mapValues(sha256),
        ]
        files["embedding-manifest.json"] = try JSONSerialization.data(withJSONObject: manifest, options: .sortedKeys)
        return files
    }
}

final class FakeIO: Sendable {
    enum Reply: Sendable {
        case content(Data)
        case status(Int)
        case disconnect
        case held(Data)
    }

    enum Ledger: Sendable {
        case write, fail, hold
    }

    struct State: Sendable {
        var replies: [String: Reply] = [:]
        var requests: [URLRequest] = []
        var running = 0
        var peak = 0
        var capacity = Int64.max
        var ledger = Ledger.write
        fileprivate var gates: [String: CheckedContinuation<Void, Never>] = [:]
        fileprivate var closed = false
    }

    private let state = Mutex(State())
    private let ledgerGate = DispatchSemaphore(value: 0)
    private let sink: AsyncStream<Event>.Continuation

    init(sink: AsyncStream<Event>.Continuation) {
        self.sink = sink
    }

    var requests: [String] { state.withLock { $0.requests.compactMap(\.url?.absoluteString) } }
    var sentRequests: [URLRequest] { state.withLock { $0.requests } }
    var peak: Int { state.withLock { $0.peak } }

    func update(_ change: (inout State) -> Void) {
        state.withLock { change(&$0) }
    }

    func release(_ url: String) {
        state.withLock { $0.gates.removeValue(forKey: url) }?.resume()
    }

    func releaseLedger() {
        ledgerGate.signal()
    }

    func releaseAll() {
        let gates = state.withLock { state in
            state.closed = true
            defer { state.gates = [:] }
            return Array(state.gates.values)
        }
        for gate in gates { gate.resume() }
        ledgerGate.signal()
    }

    func load(_ request: URLRequest, to destination: URL) async throws -> URLResponse {
        let url = request.url?.absoluteString ?? ""
        let reply = state.withLock { state in
            state.requests.append(request)
            state.running += 1
            state.peak = max(state.peak, state.running)
            return state.replies[url] ?? .status(404)
        }
        defer { state.withLock { $0.running -= 1 } }
        var status = 200
        switch reply {
        case .content(let data):
            sink.yield(.started(url))
            try data.write(to: destination)
        case .status(let code):
            sink.yield(.started(url))
            try Data("unavailable".utf8).write(to: destination)
            status = code
        case .disconnect:
            sink.yield(.started(url))
            let signed = request.url?.appending(queryItems: [URLQueryItem(name: "token", value: "secret")])
            throw URLError(.networkConnectionLost, userInfo: [NSURLErrorFailingURLErrorKey: signed as Any])
        case .held(let data):
            try data.write(to: destination)
            await withTaskCancellationHandler {
                await withCheckedContinuation { gate in
                    let waiting = state.withLock { state in
                        guard !state.closed else { return false }
                        state.gates[url] = gate
                        return true
                    }
                    if waiting { sink.yield(.started(url)) } else { gate.resume() }
                }
            } onCancel: {
                sink.yield(.cancelled(url))
            }
            sink.yield(.closed(url))
        }
        guard let address = request.url,
            let response = HTTPURLResponse(url: address, statusCode: status, httpVersion: nil, headerFields: nil)
        else { throw URLError(.badURL) }
        return response
    }

    func capacity(_ url: URL) -> Int64 {
        state.withLock { $0.capacity }
    }

    func writeLedger(_ data: Data, to url: URL) throws {
        let mode = state.withLock { state in
            defer { if state.ledger == .hold { state.ledger = .write } }
            return state.ledger
        }
        switch mode {
        case .write: break
        case .fail: throw CocoaError(.fileWriteOutOfSpace)
        case .hold:
            sink.yield(.ledger)
            ledgerGate.wait()
        }
        try data.write(to: url, options: .atomic)
    }
}

@MainActor
final class Harness {
    private final class Reference {
        weak var store: Store<ModelLibraryFeature>?
    }

    let sample: Sample
    let catalog: ModelCatalog
    let base: URL
    let io: FakeIO
    let store: ManagedModelStore
    let runner: ModelLibraryResourceRunner
    let events: AsyncStream<Event>
    private let sink: AsyncStream<Event>.Continuation
    private(set) var issued: [(action: ModelLibraryAction, commands: [ModelLibraryFeature.Operation])] = []

    init(_ sample: Sample, base: URL = temporaryDirectory(), candidates: [URL] = []) throws {
        let (events, sink) = AsyncStream.makeStream(of: Event.self)
        let io = FakeIO(sink: sink)
        io.update { state in
            for (path, data) in sample.files { state.replies[sample.url(path)] = .content(data) }
        }
        let store = ManagedModelStore(
            root: base.appending(path: "ModelLibrary"), writeLedger: io.writeLedger, availableCapacity: { _ in .max })
        self.sample = sample
        catalog = try sample.catalog()
        self.base = base
        self.io = io
        self.store = store
        runner = ModelLibraryResourceRunner(
            store: store, transfer: ArtifactTransfer(fileLoader: io.load, availableCapacity: io.capacity),
            migrationCandidates: { candidates })
        self.events = events
        self.sink = sink
    }

    var staging: URL { base.appending(path: "ModelLibrary/staging/\(sample.id)/\(revision)") }

    func inspect() async throws -> ManagedInspection {
        try await store.inspect(model: sample.model, artifact: sample.artifact)
    }

    func library(connected: Bool = false) -> Store<ModelLibraryFeature> {
        let environment = LibraryEnvironment(platform: .iOS, connectedDemos: connected ? [sample.demo.demoID] : [])
        let reference = Reference()
        let library = Store<ModelLibraryFeature>(
            initialState: ModelLibraryState(catalog: catalog, environment: environment, owner: UUID())
        ) { [runner, sink, weak self] effect, send in
            sink.yield(.began(name(effect)))
            await runner.run(effect) { action in
                if !Task.isCancelled, let state = reference.store?.state {
                    var next = state
                    self?.issued.append((action, ModelLibraryFeature.reduce(&next, action)))
                    sink.yield(.sent(name(action)))
                }
                send(action)
            }
            sink.yield(.ended(name(effect)))
        }
        reference.store = library
        return library
    }

    func close() {
        io.releaseAll()
        try? FileManager.default.removeItem(at: base)
    }
}

extension ManagedInspection {
    var resource: ValidatedModelResource? { if case .ready(let resource) = self { resource } else { nil } }
    var isMissing: Bool { if case .missing = self { true } else { false } }
}

extension ModelInstallationPhase {
    var isDownloading: Bool { if case .downloading = self { true } else { false } }
}
