#if os(macOS) || os(iOS)
    enum VoxCPMInput: Equatable, Sendable {
        case floats([Float])
        case integers([Int32])

        var count: Int {
            switch self {
            case .floats(let values): values.count
            case .integers(let values): values.count
            }
        }
    }
#endif
