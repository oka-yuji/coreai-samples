#if os(macOS) || os(iOS)
    enum VoxCPMFunction: String, CaseIterable, Sendable {
        case embed = "tslm.embed"
        case prefill = "tslm.prefill"
        case step = "tslm.step"
        case residualPrefill = "ralm.prefill"
        case residualStep = "ralm.step"
        case heads = "proj.heads"
        case fuse = "proj.fuse"
        case prefillFuse = "proj.prefill_fuse"
        case encode = "locenc.main"
        case estimate = "dit.main"
        case decode = "vae.main"

        var asset: String { String(rawValue.prefix { $0 != "." }) }
        var name: String { String(rawValue.drop { $0 != "." }.dropFirst()) }
    }
#endif
