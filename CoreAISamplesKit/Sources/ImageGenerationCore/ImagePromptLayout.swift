import Foundation

public struct ImagePromptLayout: Equatable, Sendable {
    public static let imageToken = 151_655
    public let tokens: [Int32]
    public let drop: Int
    public let imageSlots: [Bool]
    public let positions: [[Int32]]
    public let textMask: [UInt8]
    public let textCosine: [Float]
    public let textSine: [Float]

    public init(tokens: [Int], drop: Int, hasReference: Bool) throws {
        guard !tokens.isEmpty, tokens.count <= 512 else { throw ImageGenerationError.promptTooLong(tokens.count) }
        guard drop >= 0, drop < tokens.count, tokens.allSatisfy({ (0..<151_936).contains($0) }) else {
            throw ImageGenerationError.invalidTokens
        }
        let visual = tokens.map { $0 == Self.imageToken }
        let indices = visual.indices.filter { visual[$0] }
        guard indices.count == (hasReference ? 256 : 0),
            !visual.prefix(drop).contains(true),
            indices.isEmpty || indices == Array(indices[0]..<(indices[0] + 256))
        else { throw ImageGenerationError.invalidTokens }
        self.tokens = tokens.map(Int32.init)
        self.drop = drop
        imageSlots = Array(visual.dropFirst(drop))
        var positions = Array(repeating: [Int32](repeating: 0, count: tokens.count), count: 3)
        var cursor = 0
        var position: Int32 = 0
        while cursor < tokens.count {
            if visual[cursor] {
                for index in 0..<256 {
                    positions[0][cursor + index] = position
                    positions[1][cursor + index] = position + Int32(index / 16)
                    positions[2][cursor + index] = position + Int32(index % 16)
                }
                cursor += 256
                position += 16
            } else {
                for axis in 0..<3 { positions[axis][cursor] = position }
                cursor += 1
                position += 1
            }
        }
        self.positions = positions
        var mask = [UInt8](repeating: 0, count: 512 * 512)
        for query in 0..<512 {
            for key in 0..<tokens.count where query >= key { mask[query * 512 + key] = 1 }
        }
        textMask = mask
        var cosine = [Float](repeating: 0, count: 512 * 128)
        var sine = cosine
        for frequency in 0..<64 {
            let inverse: Float = 1 / powf(5_000_000, Float(2 * frequency) / 128)
            let axis = frequency < 60 && frequency % 3 != 0 ? frequency % 3 : 0
            for index in tokens.indices {
                let angle = Float(positions[axis][index]) * inverse
                for half in 0..<2 {
                    cosine[index * 128 + half * 64 + frequency] = cosf(angle)
                    sine[index * 128 + half * 64 + frequency] = sinf(angle)
                }
            }
        }
        textCosine = cosine
        textSine = sine
    }

    public var paddedTokens: [Int32] { tokens + Array(repeating: 0, count: 512 - tokens.count) }

    public var visualMask: [UInt8] {
        tokens.map { $0 == Self.imageToken ? 1 : 0 } + Array(repeating: 0, count: 512 - tokens.count)
    }
}

public struct ImageDenoiserLayout: Sendable {
    public let capacity: Int
    public let length: Int
    public let referenceCount: Int
    public let textIndex: [Int32]
    public let imageIndex: [Int32]
    public let imageMask: [UInt8]
    public let targetMask: [UInt8]
    public let targetIndices: [Int]
    public let attentionMask: [UInt8]
    public let cosine: [Float]
    public let sine: [Float]

    public init(imageSlots: [Bool], capacity: Int = 2_304) throws {
        let count = imageSlots.filter { $0 }.count
        guard !imageSlots.isEmpty, imageSlots.count <= 512, count == 0 || count == 256,
            capacity == 2_304 || (capacity == 1_536 && count == 0)
        else { throw ImageGenerationError.invalidShape }
        let slots = imageSlots + Array(repeating: true, count: 256)
        var text: [Int32] = []
        var images: [UInt8] = []
        for (index, image) in slots.enumerated() {
            text.append(contentsOf: Array(repeating: image ? 0 : Int32(index), count: image ? 4 : 1))
            images.append(contentsOf: Array(repeating: image ? 1 : 0, count: image ? 4 : 1))
        }
        let length = images.count
        guard length <= capacity else { throw ImageGenerationError.invalidShape }
        let positions = images.indices.filter { images[$0] == 1 }
        let references = count / 256
        for block in 0...references {
            let blockPositions = Array(positions[(block * 1_024)..<((block + 1) * 1_024)])
            guard let first = blockPositions.first, blockPositions == Array(first..<(first + 1_024)) else {
                throw ImageGenerationError.invalidTokens
            }
        }
        var imageIndex = [Int32](repeating: 0, count: capacity)
        var blockIDs = [Int](repeating: -1, count: length)
        for (index, position) in positions.enumerated() {
            imageIndex[position] = Int32(index)
            blockIDs[position] = index / 1_024
        }
        let targets = Array(positions.suffix(1_024))
        var target = [UInt8](repeating: 0, count: capacity)
        for index in targets { target[index] = 1 }
        var mask = [UInt8](repeating: 0, count: capacity * capacity)
        for query in 0..<capacity {
            for key in 0..<length {
                if query >= key || (query < length && blockIDs[query] >= 0 && blockIDs[query] == blockIDs[key]) {
                    mask[query * capacity + key] = 1
                }
            }
        }
        var axes = Array(repeating: [Int](repeating: 0, count: length), count: 3)
        var cursor = 0
        var position = 0
        while cursor < length {
            if images[cursor] == 1 {
                for index in 0..<1_024 {
                    axes[0][cursor + index] = position
                    axes[1][cursor + index] = index / 32 - 16
                    axes[2][cursor + index] = index % 32 - 16
                }
                cursor += 1_024
                position += 32
            } else {
                for axis in 0..<3 { axes[axis][cursor] = position }
                cursor += 1
                position += 1
            }
        }
        var cosine = [Float](repeating: 1, count: capacity * 64)
        var sine = [Float](repeating: 0, count: capacity * 64)
        var frequencyOffset = 0
        for (axis, dimension) in [16, 56, 56].enumerated() {
            for frequency in 0..<(dimension / 2) {
                let inverse: Float = 1 / powf(10_000, Float(2 * frequency) / Float(dimension))
                for index in 0..<length {
                    let angle = Float(axes[axis][index]) * inverse
                    cosine[index * 64 + frequencyOffset + frequency] = cosf(angle)
                    sine[index * 64 + frequencyOffset + frequency] = sinf(angle)
                }
            }
            frequencyOffset += dimension / 2
        }
        self.capacity = capacity
        self.length = length
        referenceCount = references
        textIndex = text + Array(repeating: 0, count: capacity - length)
        self.imageIndex = imageIndex
        imageMask = images + Array(repeating: 0, count: capacity - length)
        targetMask = target
        targetIndices = targets
        attentionMask = mask
        self.cosine = cosine
        self.sine = sine
    }
}
