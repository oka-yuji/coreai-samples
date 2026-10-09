import GLiNERCore

public struct GLiNERExtractor: Sendable {
    public let preprocessor: GLiNERPreprocessor
    public let model: GLiNERModel

    public init(preprocessor: GLiNERPreprocessor, model: GLiNERModel) {
        self.preprocessor = preprocessor
        self.model = model
    }

    @concurrent
    public func extract(
        text: String, labels: [EntityLabel], tokenCapacities: [Int] = PreparedInput.tokenCapacities
    ) async throws -> ExtractionResult {
        try await timedExtract(text: text, labels: labels, tokenCapacities: tokenCapacities).result
    }

    @concurrent
    public func timedExtract(
        text: String, labels: [EntityLabel], tokenCapacities: [Int] = PreparedInput.tokenCapacities
    ) async throws -> (result: ExtractionResult, times: GLiNERStageTimes) {
        let clock = ContinuousClock()
        let started = clock.now
        let input = try preprocessor.prepare(text: text, schema: .entities(labels), tokenCapacities: tokenCapacities)
        let preprocess = clock.now - started
        let (result, encode, pair) = try await model.extract(input, text: text)
        return (result, GLiNERStageTimes(preprocess: preprocess, encode: encode, pair: pair))
    }

    @concurrent
    public func classify(
        text: String, task: String, labels: [String], tokenCapacities: [Int] = PreparedInput.tokenCapacities
    ) async throws -> ClassificationResult {
        let input = try preprocessor.prepare(
            text: text, schema: .classification(name: task, labels: labels), tokenCapacities: tokenCapacities)
        return try await model.classify(input)
    }
}

public struct GLiNERStageTimes: Equatable, Sendable {
    public var preprocess: Duration

    public var encode: Duration

    public var pair: Duration
}
