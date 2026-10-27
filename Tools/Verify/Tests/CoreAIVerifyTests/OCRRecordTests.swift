import Foundation
import OCRCore
import Testing

@testable import OCRRuntime
@testable import coreai_verify

@Suite("OCR result JSON contract")
struct OCRRecordTests {
    private func recognition(_ text: String) -> OCRRecognition {
        OCRRecognition(
            text: text, rawText: text, tokens: [151_645], finish: .stopToken,
            gridHeight: 4, gridWidth: 4, windowCount: 1, imageTokens: 4, promptTokens: 20,
            visionSeconds: 0.1, firstTokenSeconds: 0.2, totalSeconds: 0.5)
    }

    @Test("The OCR encoder preserves the scorer's keys, table cells and skipped crop evidence")
    func encodedPage() throws {
        let layout =
            "<box:0 0 1000 100><label:title><up>"
            + "<box:500 100 500 200><label:text><up>"
            + "<box:0 200 1000 1000><label:table><up>"
        let plan = OCRPage.plan(layout: layout, width: 1_000, height: 800)
        let otsl = "<fcel>A<fcel>B<nl><fcel>1<fcel>2<nl>"
        let page = OCRPageReading(
            layout: recognition(layout),
            blocks: plan.blocks.map { .init(block: $0, recognition: recognition($0.isTable ? otsl : "Title")) },
            skippedBoxes: plan.skippedBoxes, totalSeconds: 1.5)
        let record = OCRPageRecord(
            bundle: "bundle", image: "page.png", imageSha256: "image-hash", imagePx: [1_000, 800],
            layout: OCRLayoutRecord(page),
            blocks: page.blocks.enumerated().map { OCRPageBlockRecord(index: $0.offset, $0.element) },
            generateS: page.totalSeconds, loadS: 1, tokenizerS: 0.1, assets: [], memory: .now(),
            peakRssBytes: nil, footprintPeakBytes: nil, pageins: nil, wiredDeltaBytes: nil,
            os: "test", executableSha256: nil)
        let doc = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder.records.encode(record)) as? [String: Any])
        #expect(doc["schema"] as? String == "teleocr-ocr-result/1")
        #expect(doc["record"] as? String == "ocr")
        #expect(doc["image_px"] as? [Int] == [1_000, 800])
        let encodedLayout = try #require(doc["layout"] as? [String: Any])
        #expect(encodedLayout["status"] as? String == "parsed")
        let skipped = try #require(encodedLayout["skipped_boxes"] as? [[String: Any]])
        #expect(skipped.count == 1)
        #expect(skipped.first?["label"] as? String == "text")
        #expect(skipped.first?["box_norm"] as? [Int] == [500, 100, 500, 200])
        #expect(skipped.first?["box_px"] as? [Int] == [500, 80, 500, 160])
        let blocks = try #require(doc["blocks"] as? [[String: Any]])
        #expect(blocks.map { $0["index"] as? Int } == [0, 1])
        let table = try #require(blocks.last)
        #expect(table["label"] as? String == "table")
        #expect(table["box_norm"] as? [Int] == [0, 200, 1000, 1000])
        #expect(table["box_px"] as? [Int] == [0, 160, 1000, 800])
        #expect(table["text"] as? String == otsl && table["otsl"] as? String == otsl)
        #expect(table["grid"] as? [[String]] == [["A", "B"], ["1", "2"]])
        let cells = try #require(table["cells"] as? [[String: Any]])
        #expect(cells.count == 4)
        #expect(cells.last?["text"] as? String == "2")
        for key in ["row", "column", "row_span", "column_span"] {
            #expect(cells.last?[key] as? Int == 1)
        }
        #expect(
            table["seconds"] as? [String: Double] == ["vision": 0.1, "first_token": 0.2, "decode": 0.3, "total": 0.5])
        #expect(table["tokens"] as? [String: Int] == ["prompt": 20, "image": 4, "output": 1])
        #expect(table["finished_by"] as? String == "eos")
    }

    @Test("A page with only unusable boxes encodes the full page fallback and the rejected box")
    func encodedFallback() throws {
        let layout = "<box:500 100 500 200><label:text><up>"
        let plan = OCRPage.plan(layout: layout, width: 1_000, height: 800)
        let page = OCRPageReading(
            layout: recognition(layout),
            blocks: plan.blocks.map { .init(block: $0, recognition: recognition("Text")) },
            skippedBoxes: plan.skippedBoxes, totalSeconds: 1)
        let encoded = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder.records.encode(OCRLayoutRecord(page))) as? [String: Any])
        #expect(encoded["status"] as? String == "unparsed")
        #expect((encoded["skipped_boxes"] as? [[String: Any]])?.count == 1)
        let block = try #require(page.blocks.first)
        let fields = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder.records.encode(OCRPageBlockRecord(index: 0, block)))
                as? [String: Any])
        #expect(fields["box_px"] as? [Int] == [0, 0, 1_000, 800])
        #expect(fields["label"] == nil && fields["box_norm"] == nil && fields["otsl"] == nil)
        #expect(fields["text"] as? String == "Text")
    }
}
