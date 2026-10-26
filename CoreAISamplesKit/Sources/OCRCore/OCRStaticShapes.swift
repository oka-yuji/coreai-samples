import Foundation

public struct OCRStaticShapes: Equatable, Sendable {
    public let context: Int
    public let masked: Float
    public let visionWindows: [Int]
    public let downscaleSide: Int
    public let embedTokens: Int
    public let prefillTokens: [Int]

    public init(
        context: Int, masked: Float, visionWindows: [Int], downscaleSide: Int, embedTokens: Int, prefillTokens: [Int]
    ) {
        self.context = context
        self.masked = masked
        self.visionWindows = visionWindows.sorted()
        self.downscaleSide = downscaleSide
        self.embedTokens = embedTokens
        self.prefillTokens = prefillTokens.sorted()
    }

    public func visionBucket(windows: Int) -> Int? {
        visionWindows.first { $0 >= windows }
    }

    public func prefillBucket(tokens: Int) -> Int? {
        prefillTokens.first { $0 >= tokens }
    }

    public func fittedSize(width: Int, height: Int) -> (width: Int, height: Int) {
        let side = max(width, height)
        guard side > downscaleSide else { return (width, height) }
        let scale = Double(downscaleSide) / Double(side)
        return (
            max(1, Int((Double(width) * scale).rounded())), max(1, Int((Double(height) * scale).rounded()))
        )
    }

    public func prefillMask(rows: Int, bucket: Int) -> [Float] {
        var mask = [Float](repeating: masked, count: bucket * bucket)
        for row in 0..<min(rows, bucket) {
            for column in 0...row { mask[row * bucket + column] = 0 }
        }
        return mask
    }

    public func decodeMask(position: Int) -> [Float] {
        (0..<context).map { $0 <= position ? 0 : masked }
    }

    public func writeMask(position: Int) -> [Float] {
        (0..<context).map { $0 == position ? 1 : 0 }
    }

    public static func padded(_ values: [Float], to count: Int, with fill: Float) -> [Float] {
        values + [Float](repeating: fill, count: max(0, count - values.count))
    }
}
