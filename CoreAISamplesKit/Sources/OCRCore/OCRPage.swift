import Foundation

public struct OCRPageBlock: Equatable, Sendable {
    public let layout: OCRLayoutBlock?
    public let box: OCRBox

    public var isTable: Bool { layout?.label == "table" }
    public var instruction: String { isTable ? OCRPrompt.table : OCRPrompt.text }

    public func markdown(_ text: String) -> String {
        if isTable {
            let html = OCRTable.html(otsl: text)
            return html.isEmpty ? text : html
        }
        return layout?.label == "title" && !text.isEmpty ? "# " + text : text
    }
}

public enum OCRPage {
    public static let layoutSide = 1_036

    public static func blocks(layout: String, width: Int, height: Int) -> [OCRPageBlock] {
        plan(layout: layout, width: width, height: height).blocks
    }

    public static func plan(layout: String, width: Int, height: Int)
        -> (blocks: [OCRPageBlock], skippedBoxes: [OCRLayoutBlock])
    {
        var blocks: [OCRPageBlock] = []
        var skipped: [OCRLayoutBlock] = []
        for block in OCRLayout.blocks(in: layout, width: width, height: height) {
            if isUsable(block.cropBox) {
                blocks.append(OCRPageBlock(layout: block, box: block.cropBox))
            } else {
                skipped.append(block)
            }
        }
        if blocks.isEmpty {
            blocks.append(OCRPageBlock(layout: nil, box: OCRBox(x0: 0, y0: 0, x1: width, y1: height)))
        }
        return (blocks, skipped)
    }

    private static func isUsable(_ box: OCRBox) -> Bool {
        let width = box.x1.subtractingReportingOverflow(box.x0)
        let height = box.y1.subtractingReportingOverflow(box.y0)
        guard !width.overflow, !height.overflow,
            OCRImage.isSupported(width: width.partialValue, height: height.partialValue)
        else { return false }
        return (try? OCRImage.smartResize(width: width.partialValue, height: height.partialValue)) != nil
    }
}

extension OCRTable {
    public static func html(otsl: String) -> String {
        let scalars = otsl.unicodeScalars
        if scalars.starts(with: "<table".unicodeScalars),
            scalars.reversed().starts(with: "</table>".unicodeScalars.reversed())
        {
            return otsl
        }
        let table = OCRTable(otsl: otsl)
        guard !table.cells.isEmpty, !table.rowWidths.isEmpty else { return "" }
        var owners = [[Int?]](
            repeating: [Int?](repeating: nil, count: table.columnCount), count: table.rowWidths.count)
        for (index, cell) in table.cells.enumerated() where cell.row < owners.count && cell.column < table.columnCount {
            for row in cell.row..<min(cell.row + cell.rowSpan, owners.count) {
                for column in cell.column..<min(cell.column + cell.columnSpan, table.columnCount) {
                    owners[row][column] = index
                }
            }
        }
        var html = "<table>"
        for (row, line) in owners.enumerated() {
            html += "<tr>"
            for (column, owner) in line.enumerated() {
                guard let owner, table.cells[owner].row == row, table.cells[owner].column == column else { continue }
                let cell = table.cells[owner]
                let rowSpan = cell.rowSpan > 1 ? " rowspan=\"\(cell.rowSpan)\"" : ""
                let columnSpan = cell.columnSpan > 1 ? " colspan=\"\(cell.columnSpan)\"" : ""
                html += "<td\(rowSpan)\(columnSpan)>\(escaped(cell.text))</td>"
            }
            html += "</tr>"
        }
        return html + "</table>"
    }

    static func escaped(_ text: String) -> String {
        var result = ""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "&": result += "&amp;"
            case "<": result += "&lt;"
            case ">": result += "&gt;"
            case "\"": result += "&quot;"
            case "'": result += "&#x27;"
            default: result.unicodeScalars.append(scalar)
            }
        }
        return result
    }
}
