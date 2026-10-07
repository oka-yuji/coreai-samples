import GLiNERCore

/// Text in, results out: preprocessing, then `GLiNERModel` (encode, candidates, pair) and decoding.
public struct GLiNERExtractor: Sendable {
    public let preprocessor: GLiNERPreprocessor
    public let model: GLiNERModel

    public init(preprocessor: GLiNERPreprocessor, model: GLiNERModel) {
        self.preprocessor = preprocessor
        self.model = model
    }

    /// The spans of each label in `text`, with scalar ranges of `text`. Up to 16 labels; each description goes into
    /// the prompt.
    ///
    /// - Parameter tokenCapacities: The capacities (T) to choose from; the smallest that fits is used.
    @concurrent
    public func extract(
        text: String, labels: [EntityLabel], tokenCapacities: [Int] = PreparedInput.tokenCapacities
    ) async throws -> ExtractionResult {
        try await timedExtract(text: text, labels: labels, tokenCapacities: tokenCapacities).result
    }

    /// `extract(text:labels:tokenCapacities:)` with the wall-clock time of each stage.
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

    /// The probability of each label for `text` under the classification `task`. Up to 16 labels.
    ///
    /// - Parameter tokenCapacities: The capacities (T) to choose from; the smallest that fits is used.
    @concurrent
    public func classify(
        text: String, task: String, labels: [String], tokenCapacities: [Int] = PreparedInput.tokenCapacities
    ) async throws -> ClassificationResult {
        let input = try preprocessor.prepare(
            text: text, schema: .classification(name: task, labels: labels), tokenCapacities: tokenCapacities)
        return try await model.classify(input)
    }
}

/// The wall-clock time of the stages of one extraction.
public struct GLiNERStageTimes: Equatable, Sendable {
    public var preprocess: Duration
    /// `encode_t<T>`, without the wait for earlier requests.
    public var encode: Duration
    /// The candidate pool, `pair_t<T>` and decoding.
    public var pair: Duration
}
