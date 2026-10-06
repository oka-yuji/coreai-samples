import GLiNERCore

public struct GLiNERPreprocessor: Sendable {
    public let tokenizer: GLiNERTokenizer

    public init(tokenizer: GLiNERTokenizer) {
        self.tokenizer = tokenizer
    }

    public func prepare(
        text: String,
        schema: GLiNERSchema,
        tokenCapacities: [Int] = PreparedInput.tokenCapacities
    ) throws -> PreparedInput {
        try PreparedInput(
            text: text,
            schema: schema,
            tokenCapacities: tokenCapacities,
            splitWords: JapaneseWordSplitter.words(in:),
            tokenize: tokenizer.tokenIDs(for:)
        )
    }
}
