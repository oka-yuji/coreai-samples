struct PreparedCopyProcess: Equatable, Sendable {
    enum Probe: Equatable, Sendable {
        case unavailable
        case absent
        case present(start: Int64, zombie: Bool)
    }

    let pid: Int32
    let start: Int64

    static func probe(_ pid: Int32) -> Probe {
        .unavailable
    }

    static func current() -> PreparedCopyProcess? {
        nil
    }

    static func liveness(of mark: PreparedCopyMark, probe: Probe) -> PreparedCopyLiveness {
        .alive
    }

    static func liveness(of mark: PreparedCopyMark) -> PreparedCopyLiveness {
        .alive
    }
}
