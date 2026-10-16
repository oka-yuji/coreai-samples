import Foundation

public struct RGBImage: Sendable {
    public let width: Int
    public let height: Int
    public let pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) throws {
        guard width > 0, height > 0, width <= 32_768, height <= 32_768,
            width * height <= 40_000_000, pixels.count == width * height * 3
        else { throw EmbeddingError.invalidImage }
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    public func oriented(_ orientation: Int) throws -> Self {
        guard (1...8).contains(orientation) else { throw EmbeddingError.invalidImage }
        if orientation == 1 { return self }
        let outputWidth = orientation >= 5 ? height : width
        let outputHeight = orientation >= 5 ? width : height
        var output = [UInt8](repeating: 0, count: pixels.count)
        for y in 0..<height {
            for x in 0..<width {
                let destination: (Int, Int)
                switch orientation {
                case 2: destination = (width - 1 - x, y)
                case 3: destination = (width - 1 - x, height - 1 - y)
                case 4: destination = (x, height - 1 - y)
                case 5: destination = (y, x)
                case 6: destination = (height - 1 - y, x)
                case 7: destination = (height - 1 - y, width - 1 - x)
                default: destination = (y, width - 1 - x)
                }
                for channel in 0..<3 {
                    output[(destination.1 * outputWidth + destination.0) * 3 + channel] =
                        pixels[(y * width + x) * 3 + channel]
                }
            }
        }
        return try Self(width: outputWidth, height: outputHeight, pixels: output)
    }
}

public struct EmbeddingImage: Sendable {
    public let width: Int
    public let height: Int
    public let pixels: [Float]
    public let positions: [Int32]
    public let pooling: [Float]
    public let softTokenCount: Int

    public static func size(width: Int, height: Int) throws -> (width: Int, height: Int) {
        guard width > 0, height > 0, width <= 32_768, height <= 32_768 else {
            throw EmbeddingError.invalidImage
        }
        let factor = sqrt(161_280 / (Double(width) * Double(height)))
        var targetWidth = Int(floor(Double(width) * factor / 48)) * 48
        var targetHeight = Int(floor(Double(height) * factor / 48)) * 48
        if targetWidth == 0 {
            targetWidth = 48
            targetHeight = min(height / width * 48, 3_360)
        } else if targetHeight == 0 {
            targetHeight = 48
            targetWidth = min(width / height * 48, 3_360)
        }
        guard targetWidth > 0, targetHeight > 0, targetWidth * targetHeight <= 161_280 else {
            throw EmbeddingError.invalidImage
        }
        return (targetWidth, targetHeight)
    }

    public init(_ image: RGBImage) throws {
        let size = try Self.size(width: image.width, height: image.height)
        let resized = BicubicResize.resize(image, width: size.width, height: size.height)
        width = size.width
        height = size.height
        let columns = width / 16
        let rows = height / 16
        softTokenCount = columns * rows / 9
        var pixels = [Float](repeating: 0, count: 630 * 768)
        var positions = [Int32](repeating: -1, count: 630 * 2)
        var pooling = [Float](repeating: 0, count: 70 * 630)
        for row in 0..<rows {
            for column in 0..<columns {
                let patch = row * columns + column
                positions[patch * 2] = Int32(column)
                positions[patch * 2 + 1] = Int32(row)
                pooling[((row / 3) * (columns / 3) + column / 3) * 630 + patch] = 1 / 9
                for y in 0..<16 {
                    for x in 0..<16 {
                        for channel in 0..<3 {
                            let source = ((row * 16 + y) * width + column * 16 + x) * 3 + channel
                            pixels[patch * 768 + (y * 16 + x) * 3 + channel] =
                                Float(resized[source]) * (1 / 255)
                        }
                    }
                }
            }
        }
        self.pixels = pixels
        self.positions = positions
        self.pooling = pooling
    }
}
