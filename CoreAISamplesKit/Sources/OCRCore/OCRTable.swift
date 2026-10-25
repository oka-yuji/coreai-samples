import Foundation

public struct OCRTable: Equatable, Sendable {
    public struct Cell: Equatable, Sendable {
        public let text: String
        public let row: Int
        public let column: Int
        public let rowSpan: Int
        public let columnSpan: Int
    }

    enum Token: String, CaseIterable {
        case newline = "<nl>"
        case filled = "<fcel>"
        case empty = "<ecel>"
        case left = "<lcel>"
        case up = "<ucel>"
        case cross = "<xcel>"
    }

    enum Part: Equatable {
        case token(Token)
        case text(String)
    }

    public let rowWidths: [Int]
    public let columnCount: Int
    public let cells: [Cell]
    public let grid: [[String]]

    public init(otsl: String) {
        let parts = Self.parts(otsl)
        var rows: [[Token]] = [[]]
        for case .token(let token) in parts {
            if token == .newline {
                if !rows[rows.count - 1].isEmpty { rows.append([]) }
            } else {
                rows[rows.count - 1].append(token)
            }
        }
        rows.removeAll { $0.isEmpty }
        let widths = rows.map(\.count)
        let columns = widths.max() ?? 0
        var cells: [Cell] = []
        var row = 0
        var column = 0
        for (index, part) in parts.enumerated() {
            guard case .token(let token) = part else { continue }
            if token == .filled || token == .empty {
                var text = ""
                var offset = 1
                if token == .filled, index + 1 < parts.count, case .text(let value) = parts[index + 1] {
                    text = Self.pythonStripped(value)
                    offset = 2
                }
                let right = index + offset < parts.count ? parts[index + offset] : nil
                let below = row + 1 < rows.count && column < rows[row + 1].count ? rows[row + 1][column] : nil
                let spansRight = right == .token(.left) || right == .token(.cross)
                let spansDown = below == .up || below == .cross
                let columnSpan = 1 + (spansRight ? Self.run(rows, row, column + 1, down: false) : 0)
                let rowSpan = 1 + (spansDown ? Self.run(rows, row + 1, column, down: true) : 0)
                cells.append(Cell(text: text, row: row, column: column, rowSpan: rowSpan, columnSpan: columnSpan))
            }
            if token == .newline {
                row += 1
                column = 0
            } else {
                column += 1
            }
        }
        var grid =
            cells.isEmpty ? [] : [[String]](repeating: [String](repeating: "", count: columns), count: rows.count)
        for cell in cells where cell.row < rows.count && cell.column < columns {
            for gridRow in cell.row..<min(cell.row + cell.rowSpan, rows.count) {
                for gridColumn in cell.column..<min(cell.column + cell.columnSpan, columns) {
                    grid[gridRow][gridColumn] = cell.text
                }
            }
        }
        rowWidths = widths
        columnCount = columns
        self.cells = cells
        self.grid = grid
    }

    static func run(_ rows: [[Token]], _ row: Int, _ column: Int, down: Bool) -> Int {
        let continuation: [Token] = down ? [.up, .cross] : [.left, .cross]
        var span = 0
        var row = row
        var column = column
        while row < rows.count && column < rows[row].count && continuation.contains(rows[row][column]) {
            span += 1
            if down { row += 1 } else { column += 1 }
        }
        return span
    }

    static func parts(_ text: String) -> [Part] {
        let scalars = Array(text.unicodeScalars)
        let tokens = Token.allCases.map { ($0, Array($0.rawValue.unicodeScalars)) }
        var parts: [Part] = []
        var pending: [Unicode.Scalar] = []
        var index = 0
        func flush() {
            if pending.contains(where: { !isPythonWhitespace($0) }) {
                parts.append(.text(String(String.UnicodeScalarView(pending))))
            }
            pending.removeAll()
        }
        while index < scalars.count {
            if let (token, pattern) = tokens.first(where: { scalars[index...].starts(with: $0.1) }) {
                flush()
                parts.append(.token(token))
                index += pattern.count
            } else {
                pending.append(scalars[index])
                index += 1
            }
        }
        flush()
        return parts
    }

    static func pythonStripped(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        guard let first = scalars.firstIndex(where: { !isPythonWhitespace($0) }),
            let last = scalars.lastIndex(where: { !isPythonWhitespace($0) })
        else { return "" }
        return String(String.UnicodeScalarView(scalars[first...last]))
    }

    static func isPythonWhitespace(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
            return true
        default:
            return false
        }
    }
}
