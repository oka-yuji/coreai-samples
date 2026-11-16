import Foundation

public struct VoxCPMNoise: Sendable {
    private static let stateSize = 624
    private static let shiftSize = 397
    private static let patchSize = 256
    private static let blockSize = 16

    private var state: [UInt32]
    private var position: Int

    public init(seed: UInt64) {
        var state = [UInt32](repeating: 0, count: Self.stateSize)
        state[0] = UInt32(truncatingIfNeeded: seed)
        for index in 1..<Self.stateSize {
            let previous = state[index - 1]
            state[index] = 1_812_433_253 &* (previous ^ (previous >> 30)) &+ UInt32(index)
        }
        self.state = state
        position = Self.stateSize
    }

    public mutating func nextPatch() -> [Float] {
        var values = [Float](repeating: 0, count: Self.patchSize)
        for index in values.indices {
            values[index] = Float(nextWord() & 0x00ff_ffff) * (1 / Float(1 << 24))
        }
        let half = Self.blockSize / 2
        for offset in stride(from: 0, to: values.count, by: Self.blockSize) {
            for index in offset..<(offset + half) {
                let radius = sqrtf(-2 * logf(1 - values[index]))
                let theta = Float(2 * Double.pi * Double(values[index + half]))
                var sine: Float = 0
                var cosine: Float = 0
                __sincosf(theta, &sine, &cosine)
                values[index] = radius * cosine + 0
                values[index + half] = radius * sine + 0
            }
        }
        return values
    }

    private mutating func nextWord() -> UInt32 {
        if position == Self.stateSize {
            twist()
        }
        var word = state[position]
        position += 1
        word ^= word >> 11
        word ^= (word << 7) & 0x9d2c_5680
        word ^= (word << 15) & 0xefc6_0000
        word ^= word >> 18
        return word
    }

    private mutating func twist() {
        for index in 0..<Self.stateSize {
            let next = state[(index + 1) % Self.stateSize]
            let mixed = (state[index] & 0x8000_0000) | (next & 0x7fff_ffff)
            let odd: UInt32 = (mixed & 1) == 0 ? 0 : 0x9908_b0df
            state[index] = state[(index + Self.shiftSize) % Self.stateSize] ^ (mixed >> 1) ^ odd
        }
        position = 0
    }
}
