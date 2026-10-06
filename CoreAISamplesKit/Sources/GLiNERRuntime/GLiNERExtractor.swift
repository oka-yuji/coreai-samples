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
        let input = try preprocessor.prepare(text: text, schema: .entities(labels), tokenCapacities: tokenCapacities)
        return try await model.extract(input, text: text)
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
