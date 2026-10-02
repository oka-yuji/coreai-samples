import Foundation

enum CheckpointRole: Hashable {
    case divergence
    case afterUser
    case generationStart
}

struct ReusePlan: Equatable {
    struct Stop: Equatable {
        let position: Int
        let role: CheckpointRole
    }

    static let divergenceStopTokens = 256

    let shared: Int
    let keep: Int
    let stops: [Stop]

    init(processed: [Int32], saved: [Int], prompt: [Int32], hybrid: Bool, afterUser: Int) {
        let shared = zip(processed, prompt).prefix { $0 == $1 }.count
        func latest(atMost limit: Int) -> Int { saved.filter { $0 <= limit }.max() ?? 0 }
        let keep: Int
        if shared == prompt.count {
            keep = saved.contains(shared) ? shared : hybrid ? latest(atMost: shared - 1) : shared - 1
        } else if !hybrid || shared == processed.count {
            keep = shared
        } else {
            keep = latest(atMost: shared)
        }
        var stops: [Stop] = []
        if shared < prompt.count, shared - keep >= Self.divergenceStopTokens {
            stops.append(Stop(position: shared, role: .divergence))
        }
        if hybrid, afterUser > keep { stops.append(Stop(position: afterUser, role: .afterUser)) }
        if prompt.count > keep { stops.append(Stop(position: prompt.count, role: .generationStart)) }
        self.shared = shared
        self.keep = keep
        self.stops = stops.sorted { $0.position < $1.position }
    }
}

struct SplitMix64: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

struct Sampler {
    enum Mode: Equatable {
        case greedy
        case topK(Int)
        case topP(Double)
        case full
    }

    let mode: Mode
    let temperature: Float
    private var random: SplitMix64
    private var scratch: [Float] = []

    init(mode: Mode, temperature: Float = 1, seed: UInt64? = nil) {
        self.mode = temperature > 0 ? mode : .greedy
        self.temperature = temperature
        random = SplitMix64(state: seed ?? UInt64.random(in: .min ... .max))
    }

    mutating func next(_ logits: UnsafeBufferPointer<Float16>) -> Int32 {
        if scratch.count < logits.count { scratch = [Float](repeating: 0, count: logits.count) }
        if mode == .greedy { return Engine.argmax(logits, scratch: &scratch) }
        var top = -Float.infinity
        for i in logits.indices {
            scratch[i] = Float(logits[i]) / temperature
            top = max(top, scratch[i])
        }
        var total = 0.0
        for i in logits.indices {
            scratch[i] = exp(scratch[i] - top)
            total += Double(scratch[i])
        }
        let pool = candidates(count: logits.count, total: total)
        var target = Double.random(in: 0..<1, using: &random) * pool.reduce(0) { $0 + Double(scratch[$1]) }
        for index in pool {
            target -= Double(scratch[index])
            if target < 0 { return Int32(index) }
        }
        return Int32(pool.last!)
    }

    private func candidates(count: Int, total: Double) -> [Int] {
        if mode == .full { return Array(0..<count) }
        var floor: Float = 1e-4
        var pool: [Int]
        while true {
            pool = (0..<count).filter { scratch[$0] >= floor }
            let enough =
                switch mode {
                case .topK(let k): pool.count >= k
                case .topP(let p): pool.reduce(0) { $0 + Double(scratch[$1]) } >= p * total
                default: true
                }
            if enough || floor == 0 { break }
            floor = floor > 1e-30 ? floor / 1e4 : 0
        }
        pool.sort { scratch[$0] != scratch[$1] ? scratch[$0] > scratch[$1] : $0 < $1 }
        switch mode {
        case .topK(let k):
            return Array(pool.prefix(max(k, 1)))
        case .topP(let p):
            var sum = 0.0
            let last = pool.firstIndex { sum += Double(scratch[$0]); return sum >= p * total } ?? pool.count - 1
            return Array(pool[...last])
        default:
            return pool
        }
    }
}

struct ThinkParser {
    enum Part: Equatable {
        case reasoning(String)
        case response(String)
    }

    static let open = "<think>"
    static let close = "</think>"

    private(set) var reasoning: Bool
    private var pending = ""
    private var fresh = true

    init(reasoning: Bool) {
        self.reasoning = reasoning
    }

    static func opensReasoning(_ text: String) -> Bool {
        guard let open = text.range(of: Self.open, options: .backwards) else { return false }
        guard let close = text.range(of: Self.close, options: .backwards) else { return true }
        return close.lowerBound < open.lowerBound
    }

    mutating func feed(_ text: String) -> [Part] {
        pending += text
        var parts: [Part] = []
        while true {
            let tag = reasoning ? Self.close : Self.open
            if let range = pending.range(of: tag) {
                emit(pending[..<range.lowerBound], into: &parts)
                pending = String(pending[range.upperBound...])
                reasoning.toggle()
                fresh = true
                continue
            }
            let held = (1..<tag.count).last { pending.hasSuffix(tag.prefix($0)) } ?? 0
            emit(pending.dropLast(held), into: &parts)
            pending = String(pending.suffix(held))
            return parts
        }
    }

    mutating func finish() -> [Part] {
        var parts: [Part] = []
        emit(pending[...], into: &parts)
        pending = ""
        return parts
    }

    private mutating func emit(_ text: Substring, into parts: inout [Part]) {
        var text = text
        if fresh {
            text = text.drop { $0 == "\n" }
            fresh = text.isEmpty
        }
        guard !text.isEmpty else { return }
        parts.append(reasoning ? .reasoning(String(text)) : .response(String(text)))
    }
}
