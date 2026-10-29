import Foundation
import OCRFeature
import Testing

@testable import OCRDemoRuntime

struct OCRNativePageTests {
    @Test(
        "The native page service retains CLI predictions and exports for two different fixed images",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_OCR_DEMO_BUNDLE"] != nil))
    func fixedPages() async throws {
        let environment = ProcessInfo.processInfo.environment
        let bundle = URL(filePath: try #require(environment["COREAI_OCR_DEMO_BUNDLE"]))
        let images = URL(filePath: try #require(environment["COREAI_OCR_DEMO_IMAGES"]))
        let references = URL(filePath: try #require(environment["COREAI_OCR_DEMO_REFERENCES"]))
        let output = URL(filePath: try #require(environment["COREAI_OCR_DEMO_OUTPUT"]))
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let pipeline = OCRPipeline()
        do {
            try await pipeline.load(bundle: bundle)
            for name in ["F2", "F4"] {
                let input = try await pipeline.image(at: images.appending(path: name + ".png"))
                let document = try await pipeline.read(input)
                #expect(document.isComplete)
                let data = try document.json()
                let actual = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
                let reference = try #require(
                    JSONSerialization.jsonObject(with: Data(contentsOf: references.appending(path: name + ".json")))
                        as? [String: Any])
                for key in ["record", "schema", "image_sha256", "image_px"] {
                    #expect(equal(actual[key], reference[key]))
                }
                let layout = try #require(actual["layout"] as? [String: Any])
                let expectedLayout = try #require(reference["layout"] as? [String: Any])
                for key in ["status", "resize", "text", "skipped_boxes", "tokens", "finished_by"] {
                    #expect(equal(layout[key], expectedLayout[key]))
                }
                let blocks = try #require(actual["blocks"] as? [[String: Any]])
                let expectedBlocks = try #require(reference["blocks"] as? [[String: Any]])
                #expect(blocks.count == expectedBlocks.count)
                for (block, expected) in zip(blocks, expectedBlocks) {
                    for key in [
                        "index", "label", "box_norm", "box_px", "prompt", "text", "otsl", "cells", "grid", "tokens",
                        "finished_by",
                    ] {
                        #expect(equal(block[key], expected[key]))
                    }
                }
                let markdown = document.markdown + "\n"
                #expect(try String(contentsOf: references.appending(path: name + ".md"), encoding: .utf8) == markdown)
                try data.write(to: output.appending(path: name + ".json"), options: .atomic)
                try Data(markdown.utf8).write(to: output.appending(path: name + ".md"), options: .atomic)
            }
        } catch {
            await pipeline.release()
            throw error
        }
        await pipeline.release()
    }

    private func equal(_ lhs: Any?, _ rhs: Any?) -> Bool {
        if lhs == nil && rhs == nil { return true }
        guard let lhs = lhs as? NSObject, let rhs = rhs as? NSObject else { return false }
        return lhs.isEqual(rhs)
    }
}
