import Foundation

enum BicubicResize {
    struct Weight {
        let index: Int
        let value: Double
    }

    static func kernel(_ distance: Double) -> Double {
        let value = abs(distance)
        if value < 1 { return ((1.5 * value - 2.5) * value) * value + 1 }
        if value < 2 { return ((-0.5 * value + 2.5) * value - 4) * value + 2 }
        return 0
    }

    static func weights(input: Int, output: Int) -> [[Weight]] {
        let scale = Double(input) / Double(output)
        let filterScale = max(1, scale)
        return (0..<output).map { destination in
            let center = (Double(destination) + 0.5) * scale
            let lower = max(0, Int(center - 2 * filterScale + 0.5))
            let upper = min(input, Int(center + 2 * filterScale + 0.5))
            let weights = (lower..<upper).map { source in
                Weight(index: source, value: kernel((Double(source) + 0.5 - center) / filterScale))
            }
            let total = weights.reduce(0) { $0 + $1.value }
            return weights.map { Weight(index: $0.index, value: $0.value / total) }
        }
    }

    static func resize(_ image: RGBImage, width: Int, height: Int) -> [UInt8] {
        if image.width == width && image.height == height { return image.pixels }
        let horizontal = weights(input: image.width, output: width)
        let vertical = weights(input: image.height, output: height)
        var intermediate = [UInt8](repeating: 0, count: width * image.height * 3)
        for y in 0..<image.height {
            for x in 0..<width {
                for channel in 0..<3 {
                    let value = horizontal[x].reduce(0.0) {
                        $0 + Double(image.pixels[(y * image.width + $1.index) * 3 + channel]) * $1.value
                    }
                    intermediate[(y * width + x) * 3 + channel] = byte(value)
                }
            }
        }
        var result = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                for channel in 0..<3 {
                    let value = vertical[y].reduce(0.0) {
                        $0 + Double(intermediate[($1.index * width + x) * 3 + channel]) * $1.value
                    }
                    result[(y * width + x) * 3 + channel] = byte(value)
                }
            }
        }
        return result
    }

    static func byte(_ value: Double) -> UInt8 {
        UInt8(min(255, max(0, floor(value + 0.5))))
    }
}
