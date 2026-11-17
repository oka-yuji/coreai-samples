#if os(macOS) || os(iOS)
    protocol VoxCPMTextTokenizer {
        var vocabulary: [Int32: String] { get }
        func tokenIDs(for text: String) throws -> [Int32]
    }
#endif
