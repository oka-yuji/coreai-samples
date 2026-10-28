import ArgumentParser
import CryptoKit
import Darwin
import Foundation
import OCRCore
import OCRRuntime

struct OCRRead: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ocr",
        abstract: "Read a page with the TeleOCR bundle on Core AI: layout, crops, blocks and OTSL tables."
    )

    @Argument(help: "TeleOCR bundle folder.")
    var bundle: String

    @Argument(help: "PNG or JPEG page, read as 8-bit RGB without alpha compositing or EXIF rotation.")
    var image: String

    @Option(help: "Folder for <image name>.json (the record) and <image name>.md.")
    var out: String?

    @Flag(help: "Delete the specialization cache entries of the three assets before loading.")
    var clearCache = false

    func run() async throws {
        let wiredBefore = wiredBytes()
        let bundleURL = URL(filePath: bundle)
        if clearCache { try OCRRuntime.deleteSpecializationCache(bundle: bundleURL) }
        let runtime = try await OCRRuntime.load(bundle: bundleURL)
        let file = try Data(contentsOf: URL(filePath: image))
        let clock = ContinuousClock()
        let pipelineStart = clock.now
        let source = try OCRImage(data: file)
        let page = try await runtime.read(source)
        let layout = OCRLayoutRecord(page)
        let blocks = page.blocks.enumerated().map { OCRPageBlockRecord(index: $0.offset, $0.element) }
        let markdown = page.markdown
        let pipelineSeconds = (clock.now - pipelineStart) / .seconds(1)
        var usage = rusage()
        let measured = getrusage(RUSAGE_SELF, &usage) == 0
        let executable = try Bundle.main.executableURL.map { try hex(Data(SHA256.hash(data: Data(contentsOf: $0)))) }
        let record = OCRPageRecord(
            bundle: bundle, image: image, imageSha256: hex(Data(SHA256.hash(data: file))),
            imagePx: [source.width, source.height], layout: layout, blocks: blocks,
            generateS: page.totalSeconds, loadS: runtime.loadStats.totalSeconds,
            tokenizerS: runtime.loadStats.tokenizerSeconds,
            assets: runtime.loadStats.assets.map { OCRAssetRecord(bundle: bundle, $0) },
            memory: .now(), peakRssBytes: measured ? usage.ru_maxrss : nil, footprintPeakBytes: footprintPeak(),
            pageins: measured ? usage.ru_majflt : nil,
            wiredDeltaBytes: wiredBytes().flatMap { now in wiredBefore.map { Int64(now) - Int64($0) } },
            os: ProcessInfo.processInfo.operatingSystemVersionString, executableSha256: executable,
            pipelineS: pipelineSeconds)
        try emit(record)
        if let out {
            let folder = URL(filePath: out)
            let name = URL(filePath: image).deletingPathExtension().lastPathComponent
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try (JSONEncoder.records.encode(record) + Data("\n".utf8)).write(to: folder.appending(path: name + ".json"))
            try Data((markdown + "\n").utf8).write(to: folder.appending(path: name + ".md"))
        }
    }
}

struct OCRSecondsRecord: Encodable {
    let vision: Double
    let firstToken: Double
    let decode: Double
    let total: Double

    init(_ result: OCRRecognition) {
        vision = result.visionSeconds
        firstToken = result.firstTokenSeconds
        decode = result.totalSeconds - result.firstTokenSeconds
        total = result.totalSeconds
    }
}

struct OCRTokensRecord: Encodable {
    let prompt: Int
    let image: Int
    let output: Int

    init(_ result: OCRRecognition) {
        prompt = result.promptTokens
        image = result.imageTokens
        output = result.tokens.count
    }
}

struct OCRLayoutRecord: Encodable {
    let status: String
    let resize = OCRPage.layoutSide
    let text: String
    let skippedBoxes: [OCRSkippedBoxRecord]
    let seconds: OCRSecondsRecord
    let tokens: OCRTokensRecord
    let finishedBy: String

    init(_ page: OCRPageReading) {
        status = page.layoutParsed ? "parsed" : "unparsed"
        text = page.layout.text
        skippedBoxes = page.skippedBoxes.map(OCRSkippedBoxRecord.init)
        seconds = OCRSecondsRecord(page.layout)
        tokens = OCRTokensRecord(page.layout)
        finishedBy = page.layout.finishedBy
    }
}

struct OCRSkippedBoxRecord: Encodable {
    let label: String
    let boxNorm: [Int]
    let boxPx: [Int]

    init(_ block: OCRLayoutBlock) {
        label = block.label
        let normalized = block.normalizedBox
        let crop = block.cropBox
        boxNorm = [normalized.x0, normalized.y0, normalized.x1, normalized.y1]
        boxPx = [crop.x0, crop.y0, crop.x1, crop.y1]
    }
}

struct OCRCellRecord: Encodable {
    let row: Int
    let column: Int
    let rowSpan: Int
    let columnSpan: Int
    let text: String
}

struct OCRPageBlockRecord: Encodable {
    let index: Int
    let label: String?
    let boxNorm: [Int]?
    let boxPx: [Int]
    let prompt: String
    let text: String
    let otsl: String?
    let cells: [OCRCellRecord]?
    let grid: [[String]]?
    let seconds: OCRSecondsRecord
    let tokens: OCRTokensRecord
    let finishedBy: String

    init(index: Int, _ block: OCRPageReading.Block) {
        self.index = index
        label = block.block.layout?.label
        boxNorm = block.block.layout.map(\.normalizedBox).map { [$0.x0, $0.y0, $0.x1, $0.y1] }
        boxPx = [block.block.box.x0, block.block.box.y0, block.block.box.x1, block.block.box.y1]
        prompt = block.block.isTable ? "table" : "text"
        text = block.recognition.text
        let table = block.block.isTable ? OCRTable(otsl: text) : nil
        otsl = block.block.isTable ? text : nil
        cells = table?.cells.map {
            OCRCellRecord(row: $0.row, column: $0.column, rowSpan: $0.rowSpan, columnSpan: $0.columnSpan, text: $0.text)
        }
        grid = table?.grid
        seconds = OCRSecondsRecord(block.recognition)
        tokens = OCRTokensRecord(block.recognition)
        finishedBy = block.recognition.finishedBy
    }
}

struct OCRPageRecord: Encodable {
    var record = "ocr"
    var schema = "teleocr-ocr-result/1"
    let bundle: String
    let image: String
    let imageSha256: String
    let imagePx: [Int]
    let layout: OCRLayoutRecord
    let blocks: [OCRPageBlockRecord]
    let generateS: Double
    let loadS: Double
    let tokenizerS: Double
    let assets: [OCRAssetRecord]
    let memory: Memory
    let peakRssBytes: Int?
    let footprintPeakBytes: UInt64?
    let pageins: Int?
    let wiredDeltaBytes: Int64?
    let os: String
    let executableSha256: String?
    var pipelineS: Double?
}
