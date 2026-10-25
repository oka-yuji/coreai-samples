import Foundation

public struct OCRVisionInput: Sendable {
    public static let patchSize = 14
    public static let rowWidth = 1_176
    public static let rowsPerWindow = 64
    public static let rotaryWidth = 80
    static let windowSide = 4
    static let tokensPerWindow = 16
    static let mean = [0.481_454_66, 0.457_827_5, 0.408_210_73].map { Float($0) * 255 }
    static let deviation = [0.268_629_54, 0.261_302_58, 0.275_777_11].map { Float($0) * 255 }
    static let inverseFrequencies = (0..<20).map { Float(1) / powf(10_000, Float(2 * $0) / Float(40)) }

    public let gridHeight: Int
    public let gridWidth: Int
    public let windowCount: Int
    public let pixelValues: [Float]
    public let keyValid: [Float]
    public let cos: [Float]
    public let sin: [Float]
    public let tokenRows: [Int]

    public var rowCount: Int { windowCount * Self.rowsPerWindow }

    public init(_ image: OCRImage) throws {
        let size = try OCRImage.smartResize(width: image.width, height: image.height)
        let resized = try image.resized(width: size.width, height: size.height, resampling: .torchVision)
        gridHeight = size.height / Self.patchSize
        gridWidth = size.width / Self.patchSize
        let patches = Self.patches(resized, gridHeight: gridHeight, gridWidth: gridWidth)
        let units = Self.windowUnits(mergedRows: gridHeight / 2, mergedColumns: gridWidth / 2)
        windowCount = units.count
        let rows = windowCount * Self.rowsPerWindow
        var pixelValues = [Float](repeating: 0, count: rows * Self.rowWidth)
        var keyValid = [Float](repeating: 0, count: rows)
        var cos = [Float](repeating: 1, count: rows * Self.rotaryWidth)
        var sin = [Float](repeating: 0, count: rows * Self.rotaryWidth)
        var tokenRows = [Int](repeating: 0, count: gridHeight * gridWidth / 4)
        let mergedColumns = gridWidth / 2
        for (window, members) in units.enumerated() {
            for (slot, unit) in members.enumerated() {
                tokenRows[unit] = window * Self.tokensPerWindow + slot
                for part in 0..<4 {
                    let row = window * Self.rowsPerWindow + slot * 4 + part
                    let patch = unit * 4 + part
                    keyValid[row] = 1
                    pixelValues.replaceSubrange(
                        row * Self.rowWidth..<(row + 1) * Self.rowWidth,
                        with: patches[patch * Self.rowWidth..<(patch + 1) * Self.rowWidth])
                    let height = Float(unit / mergedColumns * 2 + part / 2)
                    let width = Float(unit % mergedColumns * 2 + part % 2)
                    for index in 0..<Self.rotaryWidth {
                        let position = index % 40 < 20 ? height : width
                        let angle = position * Self.inverseFrequencies[index % 20]
                        cos[row * Self.rotaryWidth + index] = Foundation.cos(angle)
                        sin[row * Self.rotaryWidth + index] = Foundation.sin(angle)
                    }
                }
            }
        }
        self.pixelValues = pixelValues
        self.keyValid = keyValid
        self.cos = cos
        self.sin = sin
        self.tokenRows = tokenRows
    }

    public func tokenFeatures(from output: [Float]) throws -> [Float] {
        let outputRows = windowCount * Self.tokensPerWindow
        guard !output.isEmpty, output.count % outputRows == 0 else { throw OCRError.invalidShape }
        let width = output.count / outputRows
        var result = [Float]()
        result.reserveCapacity(tokenRows.count * width)
        for row in tokenRows { result.append(contentsOf: output[row * width..<(row + 1) * width]) }
        return result
    }

    static func patches(_ image: OCRImage, gridHeight: Int, gridWidth: Int) -> [Float] {
        var result = [Float](repeating: 0, count: gridHeight * gridWidth * rowWidth)
        let mean = (mean[0], mean[1], mean[2])
        let deviation = (deviation[0], deviation[1], deviation[2])
        let plane = patchSize * patchSize
        image.rgb.withUnsafeBufferPointer { pixels in
            result.withUnsafeMutableBufferPointer { output in
                for row in 0..<gridHeight {
                    for column in 0..<gridWidth {
                        let patch = ((row / 2) * (gridWidth / 2) + column / 2) * 4 + (row % 2) * 2 + column % 2
                        for y in 0..<patchSize {
                            var source = ((row * patchSize + y) * image.width + column * patchSize) * 3
                            var offset = patch * rowWidth + y * patchSize
                            for _ in 0..<patchSize {
                                let red = (Float(pixels[source]) - mean.0) / deviation.0
                                let green = (Float(pixels[source + 1]) - mean.1) / deviation.1
                                let blue = (Float(pixels[source + 2]) - mean.2) / deviation.2
                                output[offset] = red
                                output[offset + plane] = red
                                output[offset + 2 * plane] = green
                                output[offset + 3 * plane] = green
                                output[offset + 4 * plane] = blue
                                output[offset + 5 * plane] = blue
                                source += 3
                                offset += 1
                            }
                        }
                    }
                }
            }
        }
        return result
    }

    static func windowUnits(mergedRows: Int, mergedColumns: Int) -> [[Int]] {
        var windows: [[Int]] = []
        for windowRow in stride(from: 0, to: mergedRows, by: windowSide) {
            for windowColumn in stride(from: 0, to: mergedColumns, by: windowSide) {
                var members: [Int] = []
                for row in windowRow..<min(windowRow + windowSide, mergedRows) {
                    for column in windowColumn..<min(windowColumn + windowSide, mergedColumns) {
                        members.append(row * mergedColumns + column)
                    }
                }
                windows.append(members)
            }
        }
        return windows
    }
}
