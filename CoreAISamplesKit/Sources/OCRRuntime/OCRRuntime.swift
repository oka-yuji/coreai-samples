import Foundation
import OCRCore

public enum OCRRuntimeError: Error, Equatable, Sendable {
    case busy
    case invalidBundle(String)
    case invalidRequest(String)
    case invalidOutput(String)
}

public struct OCRLoadStats: Sendable {
    public struct Asset: Sendable {
        public let name: String
        public let cached: Bool
        public let seconds: Double
    }

    public let assets: [Asset]
    public let tokenizerSeconds: Double
    public let totalSeconds: Double
}

public struct OCRRecognition: Sendable {
    public let text: String
    public let rawText: String
    public let tokens: [Int32]
    public let finish: OCRGeneration.Finish
    public let gridHeight: Int
    public let gridWidth: Int
    public let windowCount: Int
    public let imageTokens: Int
    public let promptTokens: Int
    public let visionSeconds: Double
    public let firstTokenSeconds: Double
    public let totalSeconds: Double
}

public actor OCRRuntime {
    public nonisolated let loadStats: OCRLoadStats
    private let model: any OCRModel
    private var busy = false

    init(model: any OCRModel, loadStats: OCRLoadStats) {
        self.model = model
        self.loadStats = loadStats
    }

    public static func load(bundle: URL) async throws -> OCRRuntime {
        let (model, stats) = try await CoreAIOCRModel.load(bundle: bundle)
        return OCRRuntime(model: model, loadStats: stats)
    }

    public static func deleteSpecializationCache(bundle: URL) throws {
        try CoreAIOCRModel.deleteSpecializationCache(bundle: bundle)
    }

    public func recognize(
        _ image: OCRImage, instruction: String, maxNewTokens: Int = OCRGeneration.defaultMaxNewTokens
    ) async throws -> OCRRecognition {
        guard !busy else { throw OCRRuntimeError.busy }
        busy = true
        defer { busy = false }
        return try await run(model, image: image, instruction: instruction, maxNewTokens: maxNewTokens)
    }

    private func run<Model: OCRModel>(_ model: Model, image: OCRImage, instruction: String, maxNewTokens: Int)
        async throws -> OCRRecognition
    {
        let clock = ContinuousClock()
        let start = clock.now
        try Task.checkCancellation()
        let vision = try OCRVisionInput(image)
        let prompt = try OCRPrompt(
            instruction: instruction, gridHeight: vision.gridHeight, gridWidth: vision.gridWidth, encode: model.encode)
        guard maxNewTokens > 0, vision.windowCount <= model.windowLimit,
            maxNewTokens <= model.tokenLimit - prompt.ids.count
        else {
            throw OCRRuntimeError.invalidRequest(
                "\(vision.windowCount) windows and \(prompt.ids.count) + \(maxNewTokens) tokens exceed the limits.")
        }
        try Task.checkCancellation()
        let features = try vision.tokenFeatures(from: await model.vision(vision))
        let visionEnd = clock.now
        try Task.checkCancellation()
        var embeddings = try await model.embed(prompt.ids)
        let width = embeddings.count / prompt.ids.count
        guard width > 0, embeddings.count == prompt.ids.count * width,
            features.count == prompt.imageRange.count * width
        else { throw OCRRuntimeError.invalidOutput("The image features do not fit the prompt embeddings.") }
        embeddings.replaceSubrange(
            prompt.imageRange.lowerBound * width..<prompt.imageRange.upperBound * width, with: features)
        try Task.checkCancellation()
        var cache = try model.makeCache(capacity: prompt.ids.count + maxNewTokens)
        var generation = OCRGeneration(prompt: prompt.ids, maxNewTokens: maxNewTokens)
        let rotary = OCRTextRotary.cosSin(prompt.positions)
        try Task.checkCancellation()
        var token = try generation.select(await model.prefill(embeddings, rotary: rotary, cache: &cache))
        let firstToken = clock.now
        var position = prompt.ids.count
        while generation.finish == nil {
            try Task.checkCancellation()
            let next = OCRTextRotary.cosSin(
                OCRRopePositions(writePosition: position, count: 1, delta: prompt.ropeDelta))
            token = try generation.select(await model.step(token, position: position, rotary: next, cache: &cache))
            position += 1
        }
        guard let finish = generation.finish else { throw OCRRuntimeError.invalidOutput("Generation did not finish.") }
        return OCRRecognition(
            text: OCRTable.pythonStripped(model.decode(generation.tokens, skipSpecialTokens: true)),
            rawText: model.decode(generation.tokens, skipSpecialTokens: false), tokens: generation.tokens,
            finish: finish, gridHeight: vision.gridHeight, gridWidth: vision.gridWidth,
            windowCount: vision.windowCount, imageTokens: prompt.imageRange.count, promptTokens: prompt.ids.count,
            visionSeconds: (visionEnd - start) / .seconds(1), firstTokenSeconds: (firstToken - start) / .seconds(1),
            totalSeconds: (clock.now - start) / .seconds(1))
    }
}
