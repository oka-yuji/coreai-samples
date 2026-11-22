import Darwin

struct PreparedCopyProcess: Equatable, Sendable {
    enum Probe: Equatable, Sendable {
        case unavailable
        case absent
        case present(start: Int64, zombie: Bool)
    }

    let pid: Int32
    let start: Int64

    static func probe(_ pid: Int32) -> Probe {
        var name: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, pid]
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.size
        guard sysctl(&name, UInt32(name.count), &info, &size, nil, 0) == 0 else { return .unavailable }
        guard size > 0 else { return .absent }
        let time = info.kp_proc.p_un.__p_starttime
        let start = Int64(time.tv_sec) * 1_000_000 + Int64(time.tv_usec)
        return .present(start: start, zombie: Int32(info.kp_proc.p_stat) == SZOMB)
    }

    static func current() -> PreparedCopyProcess? {
        let pid = getpid()
        guard case .present(let start, false) = probe(pid) else { return nil }
        return PreparedCopyProcess(pid: pid, start: start)
    }

    static func liveness(of mark: PreparedCopyMark, probe: Probe) -> PreparedCopyLiveness {
        switch probe {
        case .unavailable:
            return .unknown
        case .absent:
            return .dead
        case .present(let start, let zombie):
            return zombie || start != mark.start ? .dead : .alive
        }
    }

    static func liveness(of mark: PreparedCopyMark) -> PreparedCopyLiveness {
        liveness(of: mark, probe: probe(mark.pid))
    }
}
