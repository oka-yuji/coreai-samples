public enum IrodoriInputError: Error, Equatable {
    case normalization(String)
    case emptyText
    case invalidFrameCount
    case tokenizer(String)
    case invalidTokenID
}
