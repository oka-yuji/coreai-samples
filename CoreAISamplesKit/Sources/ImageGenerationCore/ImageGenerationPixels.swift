import Foundation

public struct ImageGenerationPixels: Equatable, Sendable {
    public static let referenceVersion = "center-crop-bicubic-straight-rgba8-v1"
    public let width: Int
    public let height: Int
    public let rgba: [UInt8]

    public init(width: Int, height: Int, rgba: [UInt8]) throws {
        guard width > 0, height > 0, width <= 32_768, height <= 32_768,
            width * height <= 40_000_000, rgba.count == width * height * 4
        else { throw ImageGenerationError.unsupportedImage }
        self.width = width
        self.height = height
        self.rgba = rgba
    }

    public func oriented(_ orientation: Int) throws -> Self {
        guard (1...8).contains(orientation) else { throw ImageGenerationError.unsupportedImage }
        if orientation == 1 { return self }
        let outputWidth = orientation >= 5 ? height : width
        let outputHeight = orientation >= 5 ? width : height
        var result = [UInt8](repeating: 0, count: rgba.count)
        for y in 0..<height {
            for x in 0..<width {
                let destination: (Int, Int)
                switch orientation {
                case 2: destination = (width - x - 1, y)
                case 3: destination = (width - x - 1, height - y - 1)
                case 4: destination = (x, height - y - 1)
                case 5: destination = (y, x)
                case 6: destination = (height - y - 1, x)
                case 7: destination = (height - y - 1, width - x - 1)
                default: destination = (y, width - x - 1)
                }
                for channel in 0..<4 {
                    result[(destination.1 * outputWidth + destination.0) * 4 + channel] =
                        rgba[(y * width + x) * 4 + channel]
                }
            }
        }
        return try Self(width: outputWidth, height: outputHeight, rgba: result)
    }

    public func resized(width: Int, height: Int) throws -> Self {
        guard width > 0, height > 0, width <= 4_096, height <= 4_096, width * self.height <= 40_000_000 else {
            throw ImageGenerationError.unsupportedImage
        }
        if width == self.width && height == self.height { return self }
        return try Self(
            width: width, height: height,
            rgba: ImageRGBAResize.resize(rgba, width: self.width, height: self.height, toWidth: width, toHeight: height)
        )
    }

    public func squareReference() throws -> Self {
        let side = min(width, height)
        let left = (width - side) / 2
        let top = (height - side) / 2
        var square = [UInt8]()
        square.reserveCapacity(side * side * 4)
        for row in top..<(top + side) {
            square.append(contentsOf: rgba[(row * width + left) * 4..<(row * width + left + side) * 4])
        }
        return try Self(width: side, height: side, rgba: square).resized(width: 512, height: 512)
    }

    public func whiteRGB() -> [UInt8] {
        var result = [UInt8](repeating: 0, count: width * height * 3)
        for pixel in 0..<(width * height) {
            let alpha = Int(rgba[pixel * 4 + 3])
            for channel in 0..<3 {
                let total = Int(rgba[pixel * 4 + channel]) * alpha + 255 * (255 - alpha)
                result[pixel * 3 + channel] = UInt8((total + 127) / 255)
            }
        }
        return result
    }

    public func vaeInput() throws -> [Float] {
        guard width == 512, height == 512 else { throw ImageGenerationError.invalidShape }
        var result = [Float](repeating: 0, count: rgba.count)
        for pixel in 0..<(512 * 512) {
            for channel in 0..<4 {
                result[channel * 512 * 512 + pixel] = Float(rgba[pixel * 4 + channel]) / 255 * 2 - 1
            }
        }
        return result
    }

    public func visionInput() throws -> [Float] {
        guard width == 512, height == 512 else { throw ImageGenerationError.invalidShape }
        let rgb = whiteRGB()
        var result = [Float](repeating: 0, count: 1_024 * 1_536)
        for row in 0..<32 {
            for column in 0..<32 {
                let patch = ((row / 2) * 16 + column / 2) * 4 + (row % 2) * 2 + column % 2
                for channel in 0..<3 {
                    for temporal in 0..<2 {
                        for y in 0..<16 {
                            for x in 0..<16 {
                                let pixel = ((row * 16 + y) * 512 + column * 16 + x) * 3 + channel
                                let destination = patch * 1_536 + (channel * 2 + temporal) * 256 + y * 16 + x
                                result[destination] = (Float(rgb[pixel]) - 127.5) / 127.5
                            }
                        }
                    }
                }
            }
        }
        return result
    }

    public static func decodeVAE(_ values: [Float]) throws -> Self {
        let pixels = 512 * 512
        guard values.count == pixels * 4 else { throw ImageGenerationError.invalidShape }
        guard values.allSatisfy(\.isFinite) else { throw ImageGenerationError.nonFinite }
        var result = [UInt8](repeating: 0, count: values.count)
        for pixel in 0..<pixels {
            for channel in 0..<4 {
                let value = min(1, max(0, values[channel * pixels + pixel] / 2 + 0.5))
                result[pixel * 4 + channel] = UInt8((value * 255).rounded(.toNearestOrEven))
            }
        }
        return try Self(width: 512, height: 512, rgba: result)
    }
}
