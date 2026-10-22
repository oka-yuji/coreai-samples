import Foundation

enum CheckpointRole: Hashable {
    case divergence
    case afterUser
    case previousAfterUser
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

    init(
        processed: [Int32], saved: [Int], prompt: [Int32], hybrid: Bool, afterUser: Int, questionStart: () throws -> Int
    ) rethrows {
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
        if shared < prompt.count, min(shared, afterUser) - keep >= Self.divergenceStopTokens {
            let position = min(shared, try questionStart())
            if position - keep >= Self.divergenceStopTokens {
                stops.append(Stop(position: position, role: .divergence))
            }
        }
        if hybrid, afterUser > keep { stops.append(Stop(position: afterUser, role: .afterUser)) }
        if prompt.count > keep { stops.append(Stop(position: prompt.count, role: .generationStart)) }
        self.shared = shared
        self.keep = keep
        self.stops = stops.sorted { $0.position < $1.position }
    }

    static let spliceWindow = 8

    static func splice(
        _ prompt: [Int32], processed: [Int32], generations: [(start: Int, tokens: [Int32])], decode: ([Int]) -> String
    ) -> (prompt: [Int32], spliced: Int) {
        var prompt = prompt
        var spliced = 0
        for (start, tokens) in generations.sorted(by: { $0.start < $1.start }) {
            guard start <= min(prompt.count, processed.count), prompt[..<start] == processed[..<start],
                !prompt[start...].starts(with: tokens)
            else { continue }
            let text = decode(tokens.map(Int.init))
            let counts = [tokens.count] + (1...spliceWindow).flatMap { [tokens.count + $0, tokens.count - $0] }
            guard
                let count = counts.first(where: {
                    $0 > 0 && start + $0 <= prompt.count
                        && decode(prompt[start..<start + $0].map(Int.init)).utf8.elementsEqual(text.utf8)
                })
            else { continue }
            prompt.replaceSubrange(start..<start + count, with: tokens)
            spliced += 1
        }
        return (prompt, spliced)
    }
}

enum ToolOutputCut: Equatable {
    case dropped
    case prefix(Int)

    static let marker = "[truncated]"
    static let floor = 500
    static let protectedGroups = 2

    func apply(_ text: String) -> String {
        switch self {
        case .dropped: Self.marker
        case .prefix(let count): String(text.prefix(count))
        }
    }

    static func shrink(_ groups: [[(id: String, count: Int)]], _ cuts: [String: ToolOutputCut]) -> [String:
        ToolOutputCut]?
    {
        var cuts = cuts
        if let group = groups.dropLast(protectedGroups).first(where: { group in
            group.contains { cuts[$0.id] != .dropped }
        }) {
            for output in group { cuts[output.id] = .dropped }
            return cuts
        }
        var halved = false
        for output in groups.last ?? [] {
            let count =
                switch cuts[output.id] {
                case .dropped?: marker.count
                case .prefix(let kept)?: min(kept, output.count)
                case nil: output.count
                }
            if count > floor {
                cuts[output.id] = .prefix(max(floor, count / 2))
                halved = true
            }
        }
        return halved ? cuts : nil
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
    private(set) var random: SplitMix64
    private(set) var scratch: [Float] = []
    private(set) var pool: [Int] = []

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
        fillPool(count: logits.count, total: total)
        var target = Double.random(in: 0..<1, using: &random) * pool.reduce(0) { $0 + Double(scratch[$1]) }
        for index in pool {
            target -= Double(scratch[index])
            if target < 0 { return Int32(index) }
        }
        return Int32(pool.last!)
    }

    private mutating func fillPool(count: Int, total: Double) {
        if mode == .full {
            if pool.count != count {
                pool.removeAll(keepingCapacity: true)
                pool.append(contentsOf: 0..<count)
            }
            return
        }
        var floor: Float = 1e-4
        while true {
            pool.removeAll(keepingCapacity: true)
            for i in 0..<count where scratch[i] >= floor { pool.append(i) }
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
            pool.removeSubrange(min(max(k, 1), pool.count)...)
        case .topP(let p):
            var sum = 0.0
            let last =
                pool.firstIndex {
                    sum += Double(scratch[$0])
                    return sum >= p * total
                } ?? pool.count - 1
            pool.removeSubrange((last + 1)...)
        default:
            break
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
    static let promptTail = 16

    private(set) var reasoning: Bool
    private var pending = ""
    private var fresh = true

    init(reasoning: Bool) {
        self.reasoning = reasoning
    }

    static func opensReasoning(after prompt: [Int32], decode: ([Int]) -> String) -> Bool {
        opensReasoning(decode(prompt.suffix(promptTail).map(Int.init)))
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

struct ToolCallParser {
    enum Event: Equatable {
        case text(String)
        case call(name: String, arguments: String)
    }

    static let open = "<tool_call>"
    static let close = "</tool_call>"

    let types: [String: [String: String]]
    private var buffer = ""
    private var separator = ""
    private var inside = false

    init(types: [String: [String: String]]) {
        self.types = types
    }

    mutating func feed(_ text: String) -> [Event] {
        buffer += text
        var events: [Event] = []
        while true {
            if inside {
                guard let end = buffer.range(of: Self.close) else { return events }
                let body = String(buffer[..<end.lowerBound])
                buffer = String(buffer[end.upperBound...])
                inside = false
                events.append(call(body) ?? .text(separator + Self.open + body + Self.close))
                separator = ""
            } else if let start = buffer.range(of: Self.open) {
                let text = Self.dropTrailingWhitespace(buffer[..<start.lowerBound])
                if !text.isEmpty { events.append(.text(String(text))) }
                separator = String(buffer[text.endIndex..<start.lowerBound])
                buffer = String(buffer[start.upperBound...])
                inside = true
            } else {
                let held = (1..<Self.open.count).last { buffer.hasSuffix(Self.open.prefix($0)) } ?? 0
                let text = Self.dropTrailingWhitespace(buffer.dropLast(held))
                if !text.isEmpty { events.append(.text(String(text))) }
                buffer = String(buffer[text.endIndex...])
                return events
            }
        }
    }

    mutating func finish() -> [Event] {
        let rest = inside ? separator + Self.open + buffer : buffer
        buffer = ""
        separator = ""
        inside = false
        return rest.isEmpty ? [] : [.text(rest)]
    }

    private func call(_ body: String) -> Event? {
        let body = body.trimmingCharacters(in: .whitespacesAndNewlines)
        if body.hasPrefix("<function=") { return function(body) }
        guard let object = try? JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any],
            let name = object["name"] as? String, !name.isEmpty,
            let arguments = Self.members(
                object["arguments"] as? String ?? Self.members(body)?.first(where: { $0.key == "arguments" })?.value
                    ?? "{}")
        else { return nil }
        return .call(
            name: name,
            arguments: "{" + arguments.map { Self.quoted($0.key) + ":" + $0.value }.joined(separator: ",") + "}")
    }

    static func members(_ json: String) -> [(key: String, value: String)]? {
        guard (try? JSONSerialization.jsonObject(with: Data(json.utf8))) is [String: Any] else { return nil }
        let scalars = json.unicodeScalars
        var members: [(key: String, value: String)] = []
        var depth = 0
        var quoted = false
        var escaped = false
        var start = scalars.startIndex
        var colon: String.Index?
        for index in scalars.indices {
            switch scalars[index] {
            case _ where escaped: escaped = false
            case "\\" where quoted: escaped = true
            case "\"": quoted.toggle()
            case _ where quoted: break
            case "{", "[":
                depth += 1
                if depth == 1 { start = scalars.index(after: index) }
            case "}" where depth > 1, "]": depth -= 1
            case ":" where depth == 1 && colon == nil: colon = index
            case "," where depth == 1, "}":
                if let colon,
                    let key = try? JSONSerialization.jsonObject(
                        with: Data(String(scalars[start..<colon]).utf8), options: .fragmentsAllowed) as? String
                {
                    members.append(
                        (
                            key,
                            String(scalars[scalars.index(after: colon)..<index]).trimmingCharacters(
                                in: .whitespacesAndNewlines)
                        ))
                }
                start = scalars.index(after: index)
                colon = nil
            default: break
            }
        }
        return members
    }

    private func function(_ body: String) -> Event? {
        guard body.hasSuffix("</function>"), let nameEnd = body.firstIndex(of: ">") else { return nil }
        let name = body[body.index(body.startIndex, offsetBy: 10)..<nameEnd].trimmingCharacters(in: .whitespaces)
        var rest = body[body.index(after: nameEnd)...].dropLast(11)
        var pairs: [String] = []
        while let start = rest.firstIndex(where: { !$0.isWhitespace }) {
            rest = rest[start...]
            guard rest.hasPrefix("<parameter="), let keyEnd = rest.firstIndex(of: ">"),
                let end = rest.range(of: "</parameter>"), keyEnd < end.lowerBound
            else { return nil }
            let key = rest[rest.index(rest.startIndex, offsetBy: 11)..<keyEnd].trimmingCharacters(in: .whitespaces)
            var value = rest[rest.index(after: keyEnd)..<end.lowerBound]
            if value.hasPrefix("\n") { value = value.dropFirst() }
            if value.hasSuffix("\n") { value = value.dropLast() }
            guard !key.isEmpty else { return nil }
            pairs.append(Self.quoted(key) + ":" + json(String(value), type: types[name]?[key]))
            rest = rest[end.upperBound...]
        }
        guard !name.isEmpty else { return nil }
        return .call(name: name, arguments: "{" + pairs.joined(separator: ",") + "}")
    }

    private func json(_ value: String, type: String?) -> String {
        if type != "string",
            (try? JSONSerialization.jsonObject(with: Data(value.utf8), options: .fragmentsAllowed)) != nil
        {
            return value.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return Self.quoted(value)
    }

    static func quoted(_ text: String) -> String {
        let data = try? JSONSerialization.data(
            withJSONObject: text, options: [.fragmentsAllowed, .withoutEscapingSlashes])
        return String(decoding: data ?? Data(#""""#.utf8), as: UTF8.self)
    }

    private static func dropTrailingWhitespace(_ text: Substring) -> Substring {
        text[..<(text.lastIndex { !$0.isWhitespace }.map(text.index(after:)) ?? text.startIndex)]
    }
}

@_spi(Verification) public struct StateCheck: Sendable {
    static let toleranceTicks = 8

    public let currentTop: Int
    public let rereadTop: Int
    public let marginTicks: Int
    public let maxAbsDiff: Float
    public let passed: Bool

    public init(current: [Float16], reread: [Float16]) {
        precondition(current.count == reread.count && current.count > 1)
        let a = Self.top2(current)
        let b = Self.top2(reread)
        currentTop = a.first
        rereadTop = b.first
        marginTicks = max(
            Self.ticks(current[a.first], current[a.second]), Self.ticks(reread[b.first], reread[b.second]))
        maxAbsDiff = zip(current, reread).reduce(0) { max($0, abs(Float($1.0) - Float($1.1))) }
        passed =
            a.first == b.first || (a.first == b.second && a.second == b.first && marginTicks <= Self.toleranceTicks)
    }

    static func top2(_ values: [Float16]) -> (first: Int, second: Int) {
        var first = 0
        var second = 1
        if values[1] > values[0] { (first, second) = (1, 0) }
        for i in values.indices.dropFirst(2) {
            if values[i] > values[first] {
                (first, second) = (i, first)
            } else if values[i] > values[second] {
                second = i
            }
        }
        return (first, second)
    }

    static func ticks(_ a: Float16, _ b: Float16) -> Int {
        func ordered(_ x: Float16) -> Int { x.sign == .minus ? -Int(x.bitPattern & 0x7FFF) : Int(x.bitPattern) }
        return abs(ordered(a) - ordered(b))
    }
}
