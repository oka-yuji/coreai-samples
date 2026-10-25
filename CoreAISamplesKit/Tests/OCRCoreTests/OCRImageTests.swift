import Foundation
import OCRCore
import Testing

private struct ImageFixture: Decodable {
    struct SmartResize: Decodable {
        let width: Int
        let height: Int
        let resizedWidth: Int?
        let resizedHeight: Int?
        let rejected: Bool?
    }

    struct Resize: Decodable {
        let resampling: String
        let width: Int
        let height: Int
        let resizedWidth: Int
        let resizedHeight: Int
        let rgb: String?
        let sha256: String?
    }

    let smartResize: [SmartResize]
    let resize: [Resize]
    let resizeDigests: [Resize]
}

private func resampling(_ name: String) throws -> OCRImage.Resampling {
    switch name {
    case "torchVision": return .torchVision
    case "pillow": return .pillow
    default: throw OCRError.invalidImage
    }
}

@Suite("OCR image geometry and resampling")
struct OCRImageTests {
    @Test("Match the processor smart resize for every reference shape, including rejected aspect ratios")
    func smartResize() throws {
        let fixture = try OCRFixtures.load("images", as: ImageFixture.self)
        for item in fixture.smartResize {
            if item.rejected == true {
                #expect(throws: OCRError.invalidImage) {
                    try OCRImage.smartResize(width: item.width, height: item.height)
                }
            } else {
                let size = try OCRImage.smartResize(width: item.width, height: item.height)
                #expect(size.width == item.resizedWidth && size.height == item.resizedHeight, "\(item)")
            }
        }
    }

    @Test("Reproduce torchvision and Pillow bicubic bytes exactly for upscaling, downscaling, and pass order cases")
    func resizeBytes() throws {
        let fixture = try OCRFixtures.load("images", as: ImageFixture.self)
        for item in fixture.resize {
            let source = try OCRFixtures.pattern(width: item.width, height: item.height)
            let resized = try source.resized(
                width: item.resizedWidth, height: item.resizedHeight, resampling: resampling(item.resampling))
            let encoded = try #require(item.rgb)
            let expected = try #require(Data(base64Encoded: encoded))
            let differences = zip(resized.rgb, expected).filter { $0 != $1 }.count
            #expect(
                resized.rgb.count == expected.count && differences == 0,
                "\(item.resampling) \(item.width)x\(item.height)")
        }
    }

    @Test("Reproduce the page-sized processor resize and the layout resize to 1036 pixels")
    func pageResizeDigests() throws {
        let fixture = try OCRFixtures.load("images", as: ImageFixture.self)
        for item in fixture.resizeDigests {
            let source = try OCRFixtures.pattern(width: item.width, height: item.height)
            let resized = try source.resized(
                width: item.resizedWidth, height: item.resizedHeight, resampling: resampling(item.resampling))
            #expect(OCRFixtures.digest(resized.rgb) == item.sha256, "\(item.resampling)")
        }
    }

    @Test("Reject empty, mismatched, and oversized images")
    func invalidImages() {
        #expect(throws: OCRError.invalidImage) { try OCRImage(width: 0, height: 1, rgb: []) }
        #expect(throws: OCRError.invalidImage) { try OCRImage(width: 1, height: 1, rgb: [0, 0]) }
        #expect(throws: OCRError.invalidImage) { try OCRImage(width: 40_000, height: 1, rgb: []) }
        #expect(throws: OCRError.invalidImage) { try OCRImage.smartResize(width: 0, height: 10) }
        #expect(throws: OCRError.invalidImage) { try OCRImage.smartResize(width: Int.max, height: 2) }
    }
}
