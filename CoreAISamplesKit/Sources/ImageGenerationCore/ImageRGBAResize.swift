import Foundation

enum ImageRGBAResize {
    struct Weight {
        let source: Int
        let coefficient: Int
    }

    static func resize(_ pixels: [UInt8], width: Int, height: Int, toWidth: Int, toHeight: Int) -> [UInt8] {
        var premultiplied = pixels
        for index in stride(from: 0, to: pixels.count, by: 4) {
            for channel in 0..<3 {
                let product = Int(pixels[index + channel]) * Int(pixels[index + 3]) + 128
                premultiplied[index + channel] = UInt8((product + (product >> 8)) >> 8)
            }
        }
        var output: [UInt8]
        if height - toHeight > 0 && height - toHeight > (width - toWidth) * 2 {
            let intermediate = filter(premultiplied, width: width, height: height, size: toHeight, horizontal: false)
            output = filter(intermediate, width: width, height: toHeight, size: toWidth, horizontal: true)
        } else {
            let intermediate = filter(premultiplied, width: width, height: height, size: toWidth, horizontal: true)
            output = filter(intermediate, width: toWidth, height: height, size: toHeight, horizontal: false)
        }
        for index in stride(from: 0, to: output.count, by: 4) {
            let alpha = Int(output[index + 3])
            if alpha > 0 && alpha < 255 {
                for channel in 0..<3 {
                    output[index + channel] = UInt8(clamping: 255 * Int(output[index + channel]) / alpha)
                }
            }
        }
        return output
    }

    private static func filter(
        _ pixels: [UInt8], width: Int, height: Int, size: Int, horizontal: Bool
    ) -> [UInt8] {
        let coefficients = weights(input: horizontal ? width : height, output: size)
        let outputWidth = horizontal ? size : width
        let outputHeight = horizontal ? height : size
        var result = [UInt8](repeating: 0, count: outputWidth * outputHeight * 4)
        for y in 0..<outputHeight {
            for x in 0..<outputWidth {
                for channel in 0..<4 {
                    let sum = coefficients[horizontal ? x : y].reduce(1 << 21) { partial, weight in
                        let pixel = horizontal ? y * width + weight.source : weight.source * width + x
                        return partial + Int(pixels[pixel * 4 + channel]) * weight.coefficient
                    }
                    result[(y * outputWidth + x) * 4 + channel] = UInt8(clamping: sum >> 22)
                }
            }
        }
        return result
    }

    private static func weights(input: Int, output: Int) -> [[Weight]] {
        let ratio = Double(input) / Double(output)
        let scale = max(1, ratio)
        return (0..<output).map { index in
            let center = (Double(index) + 0.5) * ratio
            let begin = max(0, Int(center - 2 * scale + 0.5))
            let end = min(input, Int(center + 2 * scale + 0.5))
            let values = (begin..<end).map { kernel((Double($0) - center + 0.5) * (1 / scale)) }
            let total = values.reduce(0, +)
            return values.enumerated().map {
                Weight(source: begin + $0.offset, coefficient: Int(($0.element / total * 4_194_304).rounded()))
            }
        }
    }

    private static func kernel(_ distance: Double) -> Double {
        let value = abs(distance)
        if value < 1 { return (1.5 * value - 2.5) * value * value + 1 }
        if value < 2 { return (((value - 5) * value + 8) * value - 4) * -0.5 }
        return 0
    }
}
