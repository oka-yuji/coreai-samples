import Foundation
import OCRCore
import Testing

private struct LayoutFixture: Decodable {
    struct Block: Decodable {
        let label: String
        let orientation: String?
        let box: [Int]
        let crop: [Int]
    }

    struct Case: Decodable {
        let name: String
        let text: String
        let width: Int
        let height: Int
        let blocks: [Block]
    }

    let cases: [Case]
}

private struct TableFixture: Decodable {
    struct Cell: Decodable {
        let text: String
        let row: Int
        let column: Int
        let rowSpan: Int
        let columnSpan: Int
    }

    struct Case: Decodable {
        let name: String
        let otsl: String
        let cardRaises: Bool
        let rowWidths: [Int]
        let columnCount: Int
        let cells: [Cell]
        let grid: [[String]]
    }

    let cases: [Case]
}

private func values(_ box: OCRBox) -> [Int] { [box.x0, box.y0, box.x1, box.y1] }

@Suite("OCR layout and OTSL parsing")
struct OCRParsingTests {
    @Test("Parse layout boxes, labels, orientations, and outward-rounded crops like the reference parser")
    func layout() throws {
        let fixture = try OCRFixtures.load("layout", as: LayoutFixture.self)
        let f4 = try #require(fixture.cases.first { $0.name == "f4" })
        #expect(OCRLayout.blocks(in: f4.text, width: f4.width, height: f4.height).count == 3)
        for item in fixture.cases {
            let blocks = OCRLayout.blocks(in: item.text, width: item.width, height: item.height)
            #expect(blocks.map(\.label) == item.blocks.map(\.label), "\(item.name)")
            #expect(blocks.map(\.orientation) == item.blocks.map(\.orientation), "\(item.name)")
            #expect(blocks.map { values($0.normalizedBox) } == item.blocks.map(\.box), "\(item.name)")
            #expect(blocks.map { values($0.cropBox) } == item.blocks.map(\.crop), "\(item.name)")
        }
    }

    @Test("Parse OTSL into the reference cells and grid and keep the row widths before padding")
    func tables() throws {
        let fixture = try OCRFixtures.load("tables", as: TableFixture.self)
        let ragged = try #require(fixture.cases.first { $0.name == "raggedRows" })
        #expect(OCRTable(otsl: ragged.otsl).rowWidths == [3, 1, 2])
        for item in fixture.cases {
            let table = OCRTable(otsl: item.otsl)
            #expect(table.rowWidths == item.rowWidths, "\(item.name)")
            #expect(table.columnCount == item.columnCount, "\(item.name)")
            #expect(table.grid == item.grid, "\(item.name)")
            #expect(table.cells.map(\.text) == item.cells.map(\.text), "\(item.name)")
            #expect(
                table.cells.map { [$0.row, $0.column, $0.rowSpan, $0.columnSpan] }
                    == item.cells.map { [$0.row, $0.column, $0.rowSpan, $0.columnSpan] }, "\(item.name)")
        }
        #expect(fixture.cases.filter(\.cardRaises).map(\.name) == ["spanAfterRepeatedNewlines"])
    }
}
