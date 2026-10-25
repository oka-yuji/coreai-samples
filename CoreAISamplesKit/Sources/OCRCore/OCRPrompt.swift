import Foundation

public struct OCRRopePositions: Equatable, Sendable {
    public let temporal: [Int]
    public let height: [Int]
    public let width: [Int]

    public init(temporal: [Int], height: [Int], width: [Int]) {
        self.temporal = temporal
        self.height = height
        self.width = width
    }

    public init(writePosition: Int, count: Int, delta: Int) {
        let positions = (0..<count).map { writePosition + delta + $0 }
        self.init(temporal: positions, height: positions, width: positions)
    }
}

public struct OCRPrompt: Equatable, Sendable {
    public static let text = "Please output the text content from the image."
    public static let table = "This is the image of a table. Please output the table in OTSL format."
    public static let layout = "Analyze the image layout."
    public static let imageToken: Int32 = 151_655
    static let start: Int32 = 151_644
    static let end: Int32 = 151_645
    static let visionStart: Int32 = 151_652
    static let visionEnd: Int32 = 151_653

    public let ids: [Int32]
    public let imageRange: Range<Int>
    public let positions: OCRRopePositions
    public let ropeDelta: Int

    public init(instruction: String, gridHeight: Int, gridWidth: Int, encode: (String) throws -> [Int32]) throws {
        guard gridHeight > 0, gridWidth > 0, gridHeight % 2 == 0, gridWidth % 2 == 0 else {
            throw OCRError.invalidShape
        }
        let rows = gridHeight / 2
        let columns = gridWidth / 2
        var ids = [Self.start] + (try encode("system\nYou are a helpful assistant.")) + [Self.end]
        ids += try encode("\n") + [Self.start] + (try encode("user\n")) + [Self.visionStart]
        let imageStart = ids.count
        ids += [Int32](repeating: Self.imageToken, count: rows * columns)
        let imageEnd = ids.count
        ids += [Self.visionEnd] + (try encode(instruction)) + [Self.end]
        ids += try encode("\n") + [Self.start] + (try encode("assistant\n"))
        var temporal = Array(0..<imageStart)
        var height = temporal
        var width = temporal
        for index in 0..<(rows * columns) {
            temporal.append(imageStart)
            height.append(imageStart + index / columns)
            width.append(imageStart + index % columns)
        }
        let next = imageStart + max(rows, columns)
        let tail = Array(next..<(next + ids.count - imageEnd))
        temporal += tail
        height += tail
        width += tail
        self.ids = ids
        imageRange = imageStart..<imageEnd
        positions = OCRRopePositions(temporal: temporal, height: height, width: width)
        ropeDelta = (tail.last ?? next - 1) + 1 - ids.count
    }
}

public enum OCRTextRotary {
    public static let width = 128
    static let sections = [16, 24, 24, 16, 24, 24]
    static let inverseFrequencies = (0..<64).map { Float(1) / powf(1_000_000, Float(2 * $0) / Float(128)) }
    static let axes = sections.enumerated().flatMap { [Int](repeating: $0.offset % 3, count: $0.element) }

    public static func cosSin(_ positions: OCRRopePositions) -> (cos: [Float], sin: [Float]) {
        let count = positions.temporal.count
        var cos = [Float](repeating: 0, count: count * width)
        var sin = [Float](repeating: 0, count: count * width)
        for row in 0..<count {
            let axisPositions = [positions.temporal[row], positions.height[row], positions.width[row]]
            for index in 0..<width {
                let angle = Float(axisPositions[axes[index]]) * inverseFrequencies[index % 64]
                cos[row * width + index] = Foundation.cos(angle)
                sin[row * width + index] = Foundation.sin(angle)
            }
        }
        return (cos, sin)
    }
}
