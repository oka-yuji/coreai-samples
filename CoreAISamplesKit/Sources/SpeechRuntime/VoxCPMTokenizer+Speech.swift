#if os(macOS) || os(iOS)
    import VoxCPMTokenizers

    extension VoxCPMTokenizer: VoxCPMTextTokenizer {}
#endif
