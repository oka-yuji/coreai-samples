import Foundation

public struct OCRImage: Equatable, Sendable {
    public enum Resampling: Equatable, Sendable {
        case torchVision
        case pillow
    }

    public static let sizeFactor = 28
    public static let minimumPixels = 3_136
    public static let maximumPixels = 12_845_056

    public let width: Int
    public let height: Int
    public let rgb: [UInt8]

    public init(width: Int, height: Int, rgb: [UInt8]) throws {
        guard Self.isSupported(width: width, height: height), rgb.count == width * height * 3 else {
            throw OCRError.invalidImage
        }
        self.width = width
        self.height = height
        self.rgb = rgb
    }

    public static func smartResize(width: Int, height: Int) throws -> (width: Int, height: Int) {
        guard width > 0, height > 0, !width.multipliedReportingOverflow(by: height).overflow,
            max(width, height) <= 200 * min(width, height)
        else { throw OCRError.invalidImage }
        let factor = Double(sizeFactor)
        var resizedHeight = Int((Double(height) / factor).rounded(.toNearestOrEven)) * sizeFactor
        var resizedWidth = Int((Double(width) / factor).rounded(.toNearestOrEven)) * sizeFactor
        let rounded = resizedHeight.multipliedReportingOverflow(by: resizedWidth)
        guard !rounded.overflow else { throw OCRError.invalidImage }
        if rounded.partialValue > maximumPixels {
            let beta = (Double(height * width) / Double(maximumPixels)).squareRoot()
            resizedHeight = max(sizeFactor, Int((Double(height) / beta / factor).rounded(.down)) * sizeFactor)
            resizedWidth = max(sizeFactor, Int((Double(width) / beta / factor).rounded(.down)) * sizeFactor)
        } else if rounded.partialValue < minimumPixels {
            let beta = (Double(minimumPixels) / Double(height * width)).squareRoot()
            resizedHeight = Int((Double(height) * beta / factor).rounded(.up)) * sizeFactor
            resizedWidth = Int((Double(width) * beta / factor).rounded(.up)) * sizeFactor
        }
        return (resizedWidth, resizedHeight)
    }

    public func resized(width: Int, height: Int, resampling: Resampling) throws -> OCRImage {
        guard Self.isSupported(width: width, height: height) else { throw OCRError.invalidImage }
        if width == self.width && height == self.height { return self }
        let verticalFirst =
            resampling == .pillow && self.height - height > 0 && self.height - height > (self.width - width) * 2
        var pixels = rgb
        var currentWidth = self.width
        var currentHeight = self.height
        for horizontal in verticalFirst ? [false, true] : [true, false] {
            if horizontal && width != currentWidth {
                let kernels = Kernels(input: currentWidth, output: width, resampling: resampling)
                pixels = kernels.apply(pixels, width: currentWidth, height: currentHeight, horizontal: true)
                currentWidth = width
            } else if !horizontal && height != currentHeight {
                let kernels = Kernels(input: currentHeight, output: height, resampling: resampling)
                pixels = kernels.apply(pixels, width: currentWidth, height: currentHeight, horizontal: false)
                currentHeight = height
            }
        }
        return try OCRImage(width: width, height: height, rgb: pixels)
    }

    public func cropped(to box: OCRBox) throws -> OCRImage {
        let width = box.x1.subtractingReportingOverflow(box.x0)
        let height = box.y1.subtractingReportingOverflow(box.y0)
        guard !width.overflow, !height.overflow,
            Self.isSupported(width: width.partialValue, height: height.partialValue)
        else { throw OCRError.invalidImage }
        var pixels = [UInt8](repeating: 0, count: width.partialValue * height.partialValue * 3)
        let left = min(max(box.x0, 0), self.width)
        let span = (max(min(box.x1, self.width), left) - left) * 3
        let top = min(max(box.y0, 0), self.height)
        for y in top..<max(min(box.y1, self.height), top) where span > 0 {
            let source = (y * self.width + left) * 3
            let target = ((y - box.y0) * width.partialValue + left - box.x0) * 3
            pixels.replaceSubrange(target..<(target + span), with: rgb[source..<(source + span)])
        }
        return try OCRImage(width: width.partialValue, height: height.partialValue, rgb: pixels)
    }

    package static func isSupported(width: Int, height: Int) -> Bool {
        (1...32_768).contains(width) && (1...32_768).contains(height) && width * height <= 100_000_000
    }
}

private struct Kernels {
    let starts: [Int]
    let counts: [Int]
    let weights: [Int]
    let stride: Int
    let precision: Int

    init(input: Int, output: Int, resampling: OCRImage.Resampling) {
        let scale = Double(input) / Double(output)
        let filterScale = max(1, scale)
        let support = 2 * filterScale
        let inverse = 1 / filterScale
        let limit = Int(support.rounded(.up)) * 2 + 1
        var starts: [Int] = []
        var normalized: [[Double]] = []
        for index in 0..<output {
            let center = (Double(index) + 0.5) * scale
            let start = max(Int(center - support + 0.5), 0)
            let count = min(max(min(Int(center + support + 0.5), input) - start, 0), limit)
            let values = (0..<count).map { Self.cubic((Double(start + $0) - center + 0.5) * inverse, resampling) }
            let total = values.reduce(0, +)
            starts.append(start)
            normalized.append(total == 0 ? values : values.map { $0 / total })
        }
        var precision = 22
        if resampling == .torchVision {
            let largest = normalized.joined().reduce(0, max)
            precision = 0
            while precision < 22 && Int(0.5 + largest * Double(1 << (precision + 1))) < 1 << 15 { precision += 1 }
        }
        let unit = Double(1 << precision)
        self.starts = starts
        counts = normalized.map(\.count)
        stride = limit
        weights = normalized.flatMap { row in
            row.map { value in
                let scaled = value * unit
                return Int(scaled < 0 ? -0.5 + scaled : 0.5 + scaled)
            } + [Int](repeating: 0, count: limit - row.count)
        }
        self.precision = precision
    }

    static func cubic(_ distance: Double, _ resampling: OCRImage.Resampling) -> Double {
        let x = abs(distance)
        if x < 1 { return (1.5 * x - 2.5) * x * x + 1 }
        if x < 2 {
            return resampling == .torchVision ? ((-0.5 * x + 2.5) * x - 4) * x + 2 : (((x - 5) * x + 8) * x - 4) * -0.5
        }
        return 0
    }

    func apply(_ pixels: [UInt8], width: Int, height: Int, horizontal: Bool) -> [UInt8] {
        let outputWidth = horizontal ? starts.count : width
        let outputHeight = horizontal ? height : starts.count
        let step = horizontal ? 3 : width * 3
        let rounding = 1 << (precision - 1)
        var result = [UInt8](repeating: 0, count: outputWidth * outputHeight * 3)
        pixels.withUnsafeBufferPointer { source in
            weights.withUnsafeBufferPointer { weights in
                result.withUnsafeMutableBufferPointer { output in
                    for y in 0..<outputHeight {
                        for x in 0..<outputWidth {
                            let kernel = horizontal ? x : y
                            let base = horizontal ? (y * width + starts[kernel]) * 3 : (starts[kernel] * width + x) * 3
                            let first = kernel * stride
                            let end = first + counts[kernel]
                            for channel in 0..<3 {
                                var sum = rounding
                                var offset = base + channel
                                var tap = first
                                while tap < end {
                                    sum += Int(source[offset]) * weights[tap]
                                    offset += step
                                    tap += 1
                                }
                                output[(y * outputWidth + x) * 3 + channel] = UInt8(clamping: sum >> precision)
                            }
                        }
                    }
                }
            }
        }
        return result
    }
}
