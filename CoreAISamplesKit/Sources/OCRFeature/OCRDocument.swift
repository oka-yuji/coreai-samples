import Foundation
import OCRCore

public struct OCRInput: Equatable, Sendable {
    public let name: String
    public let image: OCRImage
    public let preview: Data
    public let sha256: String

    public init(name: String, image: OCRImage, preview: Data, sha256: String) {
        self.name = name
        self.image = image
        self.preview = preview
        self.sha256 = sha256
    }
}

public struct OCRDocument: Equatable, Encodable, Sendable {
    public struct Seconds: Equatable, Encodable, Sendable {
        public let vision: Double
        public let firstToken: Double
        public let decode: Double
        public let total: Double

        public init(vision: Double, firstToken: Double, total: Double) {
            self.vision = vision
            self.firstToken = firstToken
            decode = total - firstToken
            self.total = total
        }
    }

    public struct Tokens: Equatable, Encodable, Sendable {
        public let prompt: Int
        public let image: Int
        public let output: Int

        public init(prompt: Int, image: Int, output: Int) {
            self.prompt = prompt
            self.image = image
            self.output = output
        }
    }

    public struct SkippedBox: Equatable, Encodable, Sendable {
        public let label: String
        public let boxNorm: [Int]
        public let boxPx: [Int]

        public init(_ block: OCRLayoutBlock) {
            label = block.label
            boxNorm = block.normalizedBox.coordinates
            boxPx = block.cropBox.coordinates
        }
    }

    public struct Layout: Equatable, Encodable, Sendable {
        public let status: String
        public let resize = OCRPage.layoutSide
        public let text: String
        public let skippedBoxes: [SkippedBox]
        public let seconds: Seconds
        public let tokens: Tokens
        public let finishedBy: String

        public init(
            parsed: Bool, text: String, skippedBoxes: [OCRLayoutBlock], seconds: Seconds, tokens: Tokens,
            finishedBy: String
        ) {
            status = parsed ? "parsed" : "unparsed"
            self.text = text
            self.skippedBoxes = skippedBoxes.map(SkippedBox.init)
            self.seconds = seconds
            self.tokens = tokens
            self.finishedBy = finishedBy
        }
    }

    public struct Cell: Equatable, Encodable, Sendable {
        public let row: Int
        public let column: Int
        public let rowSpan: Int
        public let columnSpan: Int
        public let text: String

        init(_ cell: OCRTable.Cell) {
            row = cell.row
            column = cell.column
            rowSpan = cell.rowSpan
            columnSpan = cell.columnSpan
            text = cell.text
        }
    }

    public struct Block: Equatable, Encodable, Identifiable, Sendable {
        public let index: Int
        public let label: String?
        public let boxNorm: [Int]?
        public let boxPx: [Int]
        public let prompt: String
        public let text: String
        public let otsl: String?
        public let cells: [Cell]?
        public let grid: [[String]]?
        public let seconds: Seconds
        public let tokens: Tokens
        public let finishedBy: String
        public var id: Int { index }

        public init(
            index: Int, layout: OCRLayoutBlock?, box: OCRBox, text: String, seconds: Seconds, tokens: Tokens,
            finishedBy: String
        ) {
            self.index = index
            label = layout?.label
            boxNorm = layout?.normalizedBox.coordinates
            boxPx = box.coordinates
            prompt = layout?.label == "table" ? "table" : "text"
            self.text = text
            otsl = prompt == "table" ? text : nil
            let table = otsl.map(OCRTable.init(otsl:))
            cells = table?.cells.map(Cell.init)
            grid = table?.grid
            self.seconds = seconds
            self.tokens = tokens
            self.finishedBy = finishedBy
        }

        public var markdown: String {
            if let otsl {
                let html = OCRTable.html(otsl: otsl)
                return html.isEmpty ? text : html
            }
            return label == "title" && !text.isEmpty ? "# " + text : text
        }
    }

    public let record = "ocr"
    public let schema = "teleocr-ocr-result/1"
    public let bundle: String
    public let image: String
    public let imageSha256: String
    public let imagePx: [Int]
    public let layout: Layout
    public let blocks: [Block]
    public let generateS: Double
    public let loadS: Double

    public init(bundle: String, input: OCRInput, layout: Layout, blocks: [Block], generateS: Double, loadS: Double) {
        self.bundle = bundle
        image = input.name
        imageSha256 = input.sha256
        imagePx = [input.image.width, input.image.height]
        self.layout = layout
        self.blocks = blocks
        self.generateS = generateS
        self.loadS = loadS
    }

    public var isComplete: Bool { layout.finishedBy == "eos" && blocks.allSatisfy { $0.finishedBy == "eos" } }
    public var markdown: String { blocks.map(\.markdown).filter { !$0.isEmpty }.joined(separator: "\n\n") }

    public func json() throws -> Data {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(self) + Data("\n".utf8)
    }
}

extension OCRBox {
    fileprivate var coordinates: [Int] { [x0, y0, x1, y1] }
}
