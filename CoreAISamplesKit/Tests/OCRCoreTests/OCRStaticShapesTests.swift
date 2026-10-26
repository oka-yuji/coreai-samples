import Foundation
import Testing

@testable import OCRCore

@Suite("OCR static shape buckets and masks")
struct OCRStaticShapesTests {
    private let shapes = OCRStaticShapes(
        context: 8, masked: -40_000, visionWindows: [100, 16, 64, 32], downscaleSide: 896, embedTokens: 2_048,
        prefillTokens: [2_048, 256, 1_024, 512])

    @Test(
        "Choose the smallest bucket that holds the request and refuse a request above every bucket",
        arguments: [(1, 16), (16, 16), (17, 32), (44, 64), (65, 100), (100, 100)])
    func visionBuckets(windows: Int, expected: Int) {
        #expect(shapes.visionBucket(windows: windows) == expected)
        #expect(shapes.visionBucket(windows: 101) == nil)
        #expect(shapes.prefillBucket(tokens: 286) == 512)
        #expect(shapes.prefillBucket(tokens: 1_396) == 2_048)
        #expect(shapes.prefillBucket(tokens: 2_049) == nil)
    }

    @Test("Keep an image that fits the downscale side and shrink a larger one to fit while keeping its aspect")
    func fittedSize() throws {
        #expect(shapes.fittedSize(width: 896, height: 300) == (896, 300))
        #expect(shapes.fittedSize(width: 1_240, height: 1_754) == (633, 896))
        let resized = try OCRImage.smartResize(width: 633, height: 896)
        #expect(resized.width <= 896 && resized.height <= 896)
        #expect(shapes.fittedSize(width: 30_000, height: 10) == (896, 1))
    }

    @Test("Reveal causal keys to prompt rows, hide every key of padding rows, and mark one decode position")
    func masks() {
        let mask = shapes.prefillMask(rows: 2, bucket: 3)
        #expect(mask == [0, -40_000, -40_000, 0, 0, -40_000, -40_000, -40_000, -40_000])
        #expect(shapes.decodeMask(position: 2) == [0, 0, 0, -40_000, -40_000, -40_000, -40_000, -40_000])
        #expect(shapes.writeMask(position: 2) == [0, 0, 1, 0, 0, 0, 0, 0])
        #expect(OCRStaticShapes.padded([1, 2], to: 4, with: 9) == [1, 2, 9, 9])
        #expect(OCRStaticShapes.padded([1, 2], to: 1, with: 9) == [1, 2])
    }
}
