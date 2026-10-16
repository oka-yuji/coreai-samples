import Foundation

public enum IrodoriNoise {
    public static func generate(frames: Int, seed: UInt64) throws -> [Float] {
        guard (13...750).contains(frames) else { throw IrodoriInputError.invalidFrameCount }
        var generator = MT19937(seed: seed)
        var values = (0..<(frames * 32)).map { _ in Float(generator.next() & 0x00ff_ffff) * (1 / Float(1 << 24)) }
        for offset in stride(from: 0, to: values.count, by: 16) {
            for index in 0..<8 {
                let first = 1 - values[offset + index]
                let second = values[offset + index + 8]
                let radius = sqrtf(-2 * logf(first))
                let theta = Float(2 * Double.pi * Double(second))
                var sine: Float = 0
                var cosine: Float = 0
                __sincosf(theta, &sine, &cosine)
                values[offset + index] = radius * cosine + 0
                values[offset + index + 8] = radius * sine + 0
            }
        }
        return values
    }
}

private struct MT19937 {
    private var state: [UInt32]
    private var index = 624

    init(seed: UInt64) {
        state = [UInt32](repeating: 0, count: 624)
        state[0] = UInt32(truncatingIfNeeded: seed)
        for position in 1..<624 {
            let previous = state[position - 1]
            state[position] = 1_812_433_253 &* (previous ^ (previous >> 30)) &+ UInt32(position)
        }
    }

    mutating func next() -> UInt32 {
        if index == 624 {
            for position in 0..<624 {
                let combined = (state[position] & 0x8000_0000) | (state[(position + 1) % 624] & 0x7fff_ffff)
                let odd: UInt32 = (combined & 1) == 0 ? 0 : 0x9908_b0df
                state[position] = state[(position + 397) % 624] ^ (combined >> 1) ^ odd
            }
            index = 0
        }
        var result = state[index]
        index += 1
        result ^= result >> 11
        result ^= (result << 7) & 0x9d2c_5680
        result ^= (result << 15) & 0xefc6_0000
        result ^= result >> 18
        return result
    }
}
