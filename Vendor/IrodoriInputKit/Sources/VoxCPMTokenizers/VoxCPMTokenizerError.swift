public enum VoxCPMTokenizerError: Error, Equatable, Sendable {
    case unexpectedSHA256
    case invalidVocabulary
    case invalidTokenID
    case tokenizer(String)
}
