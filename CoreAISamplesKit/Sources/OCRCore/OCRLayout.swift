import Foundation

public struct OCRBox: Equatable, Sendable {
    public let x0: Int
    public let y0: Int
    public let x1: Int
    public let y1: Int

    public init(x0: Int, y0: Int, x1: Int, y1: Int) {
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
    }
}

public struct OCRLayoutBlock: Equatable, Sendable {
    public let label: String
    public let orientation: String?
    public let normalizedBox: OCRBox
    public let cropBox: OCRBox
}

public enum OCRLayout {
    public static func blocks(in text: String, width: Int, height: Int) -> [OCRLayoutBlock] {
        let scalars = Array(text.unicodeScalars)
        var blocks: [OCRLayoutBlock] = []
        var index = 0
        while index < scalars.count {
            if let (block, end) = match(scalars, at: index, width: width, height: height) {
                blocks.append(block)
                index = end
            } else {
                index += 1
            }
        }
        return blocks
    }

    private static func match(_ scalars: [Unicode.Scalar], at start: Int, width: Int, height: Int)
        -> (OCRLayoutBlock, Int)?
    {
        var index = start
        guard literal("<box:", scalars, &index) else { return nil }
        var values: [Int] = []
        for separator in [" ", " ", " ", ">"] {
            guard let value = number(scalars, &index), literal(separator, scalars, &index) else { return nil }
            values.append(value)
        }
        guard literal("<label:", scalars, &index) else { return nil }
        let labelStart = index
        while index < scalars.count && scalars[index] != ">" { index += 1 }
        guard index > labelStart, index < scalars.count else { return nil }
        let label = String(String.UnicodeScalarView(scalars[labelStart..<index]))
        index += 1
        var orientation: String?
        var lookahead = index
        if literal("<", scalars, &lookahead) {
            let wordStart = lookahead
            while lookahead < scalars.count && ("a"..."z").contains(scalars[lookahead]) { lookahead += 1 }
            if lookahead > wordStart, literal(">", scalars, &lookahead) {
                orientation = String(String.UnicodeScalarView(scalars[wordStart..<(lookahead - 1)]))
                index = lookahead
            }
        }
        guard let crop = cropBox(values, width: width, height: height) else { return nil }
        let normalized = OCRBox(x0: values[0], y0: values[1], x1: values[2], y1: values[3])
        return (OCRLayoutBlock(label: label, orientation: orientation, normalizedBox: normalized, cropBox: crop), index)
    }

    private static func cropBox(_ values: [Int], width: Int, height: Int) -> OCRBox? {
        var scaled: [Int] = []
        for (value, size) in zip(values, [width, height, width, height]) {
            let product = value.multipliedReportingOverflow(by: size)
            guard !product.overflow, product.partialValue <= Int.max - 999 else { return nil }
            scaled.append(product.partialValue)
        }
        return OCRBox(
            x0: scaled[0] / 1000, y0: scaled[1] / 1000, x1: (scaled[2] + 999) / 1000, y1: (scaled[3] + 999) / 1000)
    }

    private static func literal(_ text: String, _ scalars: [Unicode.Scalar], _ index: inout Int) -> Bool {
        let expected = Array(text.unicodeScalars)
        guard index + expected.count <= scalars.count, Array(scalars[index..<(index + expected.count)]) == expected
        else { return false }
        index += expected.count
        return true
    }

    private static func number(_ scalars: [Unicode.Scalar], _ index: inout Int) -> Int? {
        var value = 0
        let start = index
        while index < scalars.count, scalars[index].properties.numericType == .decimal,
            let digit = scalars[index].properties.numericValue
        {
            let next = value.multipliedReportingOverflow(by: 10)
            guard !next.overflow, next.partialValue <= Int.max - Int(digit) else { return nil }
            value = next.partialValue + Int(digit)
            index += 1
        }
        return index > start ? value : nil
    }
}
