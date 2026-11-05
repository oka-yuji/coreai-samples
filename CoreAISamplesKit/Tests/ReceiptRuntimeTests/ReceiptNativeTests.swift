import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import ReceiptCore
import Testing

@testable import ReceiptRuntime

@Suite(.serialized, .timeLimit(.minutes(5)))
struct ReceiptNativeTests {
    @Test(
        "Native Vision reads the synthetic images and PDF without hiding page limits",
        .enabled(if: ProcessInfo.processInfo.environment["RECEIPT_VISION_TEST"] == "1"))
    func nativeOCR() async throws {
        let examples = try await ReceiptPipeline.samples()
        let expected = try expectations(examples)
        let pipeline = ReceiptPipeline()
        do {
            for (index, example) in examples.enumerated() {
                let read = try await pipeline.read(id: index + 1, url: example.url)
                let item = try #require(expected[example.id])
                #expect(read.record.fields.amount == item.expectedAmount, "\(read.record.text)")
                #expect(read.record.fields.date == item.expectedDate)
                #expect(read.previews.count == 1)
                #expect(!read.record.issues.contains(.unreadablePage))
            }
            let files = try ReceiptTestFiles()
            defer { files.cleanUp() }
            let image = try Data(contentsOf: #require(examples.first).url)
            let pdf = try files.file("receipt.pdf", bytes: makePDF(image: image, pages: 1))
            let read = try await pipeline.read(id: 10, url: pdf)
            if let path = ProcessInfo.processInfo.environment["RECEIPT_NATIVE_PREVIEW"] {
                try read.previews[0].write(to: URL(filePath: path), options: .withoutOverwriting)
            }
            #expect(read.record.sourceExtension == "pdf")
            #expect(read.record.fields.amount == "1320", "\(read.record.text)")
            let tooLong = try files.file("six-pages.pdf", bytes: makePDF(image: image, pages: 6))
            await #expect(throws: ReceiptRuntimeError.unsupportedPDF) { try await pipeline.read(id: 11, url: tooLong) }
            await pipeline.release()
        } catch {
            await pipeline.release()
            throw error
        }
    }

    @Test(
        "The published JEV bundle classifies OCR text and exports a confirmed synthetic ledger",
        .enabled(if: ProcessInfo.processInfo.environment["RECEIPT_JEV_BUNDLE"] != nil))
    func nativeJEV() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["RECEIPT_JEV_BUNDLE"])
        let bundle = URL(filePath: path)
        let manifest = try Data(contentsOf: bundle.appending(path: "decision.json"))
        let digest = SHA256.hash(data: manifest).map { String(format: "%02x", $0) }.joined()
        try #require(digest == "2f34243a3d28500075e09d02d9f9f565a7c4b02f94a7d47b2a3caba7262d357d")
        let examples = try await ReceiptPipeline.samples()
        let expected = try expectations(examples)
        let pipeline = ReceiptPipeline()
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        var rows: [Result] = []
        var records: [ReceiptRecord] = []
        do {
            try await pipeline.prepare(bundle: bundle)
            for (index, example) in examples.enumerated() {
                let read = try await pipeline.read(id: index + 1, url: example.url)
                var record = read.record
                record.edit(.purpose, value: example.purpose)
                let decision = try await pipeline.classify(record)
                record.apply(decision.classification)
                let item = try #require(expected[example.id])
                #expect(record.fields.amount == item.expectedAmount)
                #expect(record.fields.date == item.expectedDate)
                #expect(decision.classification.category == item.expectedCategory)
                #expect(decision.tokenCount > 0)
                rows.append(
                    Result(
                        sample: example.id, category: record.fields.category.rawValue,
                        expectedCategory: item.expectedCategory.rawValue, amount: record.fields.amount,
                        probability: decision.classification.probability, tokens: decision.tokenCount,
                        ocrSeconds: read.seconds, decisionSeconds: decision.seconds))
                if record.canConfirm {
                    try record.confirm()
                    records.append(record)
                }
            }
            let ledger = try ReceiptLedger(records: records)
            let destination = try await pipeline.export(ledger, to: files.folder())
            #expect(try Data(contentsOf: destination.directory.appending(path: "ledger.csv")) == ledger.csv)
            for row in ledger.rows {
                let bytes = try Data(contentsOf: destination.directory.appending(path: row.relativePath))
                #expect(ReceiptFileReader.digest(bytes) == row.sourceSHA256)
            }
            if let output = ProcessInfo.processInfo.environment["RECEIPT_NATIVE_REPORT"] {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let report = Report(
                    format: "receipt-native-check/1", os: ProcessInfo.processInfo.operatingSystemVersionString,
                    decisionSHA256: digest, rows: rows, confirmedCount: ledger.rows.count,
                    totalYen: String(ledger.totalYen))
                try encoder.encode(report).write(to: URL(filePath: output), options: .withoutOverwriting)
            }
            await pipeline.release()
        } catch {
            await pipeline.release()
            throw error
        }
    }

    private struct Expected: Decodable {
        let file: String
        let expectedCategory: ReceiptCategory
        let expectedAmount: String
        let expectedDate: String
    }

    private struct Result: Encodable {
        let sample: String
        let category: String
        let expectedCategory: String
        let amount: String
        let probability: Float
        let tokens: Int
        let ocrSeconds: Double
        let decisionSeconds: Double
    }

    private struct Report: Encodable {
        let format: String
        let os: String
        let decisionSHA256: String
        let rows: [Result]
        let confirmedCount: Int
        let totalYen: String
    }

    private func expectations(_ examples: [ReceiptSample]) throws -> [String: Expected] {
        let sample = try #require(examples.first)
        let data = try Data(contentsOf: sample.url.deletingLastPathComponent().appending(path: "receipts.json"))
        let rows = try JSONDecoder().decode([Expected].self, from: data)
        return Dictionary(uniqueKeysWithValues: rows.map { ($0.file, $0) })
    }

    private func makePDF(image: Data, pages: Int) throws -> Data {
        let source = try #require(CGImageSourceCreateWithData(image as CFData, nil))
        let raster = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        let output = NSMutableData()
        let consumer = try #require(CGDataConsumer(data: output))
        var bounds = CGRect(x: 0, y: 0, width: 450, height: 600)
        let context = try #require(CGContext(consumer: consumer, mediaBox: &bounds, nil))
        for _ in 0..<pages {
            context.beginPDFPage(nil)
            context.draw(raster, in: bounds)
            context.endPDFPage()
        }
        context.closePDF()
        return output as Data
    }
}
