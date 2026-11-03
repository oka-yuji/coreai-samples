import Foundation

protocol TypedDecisionBackbone: Sendable {
    func forward(_ ids: [Int], capacity: Int) async throws -> [Float]
}

public struct TypedDecisionLoadStats: Sendable, Equatable {
    public var bundleSeconds = 0.0
    public var tokenizerSeconds = 0.0
    public var headSeconds = 0.0
    public var backboneSeconds = 0.0
    public var cachedSpecialization = false
}

public actor TypedDecisionModel {
    public nonisolated let bundle: TypedDecisionBundle
    public nonisolated let loadStats: TypedDecisionLoadStats
    private let encode: @Sendable (String) -> [Int]
    private let head: TypedDecisionHead
    private let backbone: any TypedDecisionBackbone
    private var busy = false
    private var forwardCheckpoint: (@Sendable () async -> Void)?

    init(
        bundle: TypedDecisionBundle, encode: @escaping @Sendable (String) -> [Int], head: TypedDecisionHead,
        backbone: any TypedDecisionBackbone, loadStats: TypedDecisionLoadStats = TypedDecisionLoadStats()
    ) {
        self.bundle = bundle
        self.encode = encode
        self.head = head
        self.backbone = backbone
        self.loadStats = loadStats
    }

    public static func load(bundle url: URL) async throws -> TypedDecisionModel {
        let clock = ContinuousClock()
        var stats = TypedDecisionLoadStats()
        var start = clock.now
        let bundle = try TypedDecisionBundle(url: url)
        stats.bundleSeconds = (clock.now - start) / .seconds(1)
        try Task.checkCancellation()
        start = clock.now
        let tokenizer = try await DecisionModel.loadTokenizer(bundle.tokenizerURL)
        stats.tokenizerSeconds = (clock.now - start) / .seconds(1)
        start = clock.now
        let head = try TypedDecisionHead(url: bundle.headURL, width: bundle.hiddenSize)
        stats.headSeconds = (clock.now - start) / .seconds(1)
        try Task.checkCancellation()
        start = clock.now
        let (runtime, cached) = try await TypedDecisionRuntime.load(bundle: bundle)
        stats.backboneSeconds = (clock.now - start) / .seconds(1)
        stats.cachedSpecialization = cached
        return TypedDecisionModel(
            bundle: bundle, encode: { tokenizer.encode(text: $0, addSpecialTokens: false) }, head: head,
            backbone: runtime, loadStats: stats)
    }

    public nonisolated func capacity(forTokens count: Int) -> Int {
        min((count + 255) / 256 * 256, bundle.limits.inputTokens)
    }

    public func tokenIDs(for request: TypedDecisionRequest) throws -> [Int] {
        try TypedDecisionRenderer.tokenIDs(for: request, limit: bundle.limits.inputTokens, encode: encode)
    }

    public func tokenCount(for request: TypedDecisionRequest) throws -> Int {
        try tokenIDs(for: request).count
    }

    public func decide(_ request: TypedDecisionRequest) async throws -> TypedDecisionAnswer {
        try await decide(request, capacity: nil)
    }

    func decide(_ request: TypedDecisionRequest, capacity: Int?) async throws -> TypedDecisionAnswer {
        let ids = try tokenIDs(for: request)
        guard let temperature = bundle.temperatures[request.kind] else {
            throw TypedDecisionError.invalidBundle("The bundle has no temperature for \(request.kind.rawValue).")
        }
        guard !busy else { throw TypedDecisionError.busy }
        busy = true
        defer { busy = false }
        try Task.checkCancellation()
        if let forwardCheckpoint { await forwardCheckpoint() }
        let hidden = try await backbone.forward(ids, capacity: capacity ?? self.capacity(forTokens: ids.count))
        try Task.checkCancellation()
        let logits = try head.logits(hidden: hidden)
        let probabilities = try head.probabilities(
            logits: logits, kind: request.kind, count: request.options.count, temperature: temperature)
        return TypedDecisionAnswer(
            kind: request.kind, options: request.options, logits: logits, probabilities: probabilities.shipping,
            probabilitiesT1: probabilities.plain, top: TypedDecisionHead.top(of: probabilities.shipping),
            tokenCount: ids.count)
    }

    func setForwardCheckpoint(_ checkpoint: (@Sendable () async -> Void)?) {
        forwardCheckpoint = checkpoint
    }
}
