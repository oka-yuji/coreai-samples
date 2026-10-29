import CryptoKit
import Foundation
import ImageGenerationRuntime
import Testing

struct ImageReferencePixelIdentityTests {
    @Test("URL decoding and the ordinary preview preserve the independent transparent PNG pixel identity")
    func transparentReference() async throws {
        let url = try #require(
            Bundle.module.url(
                forResource: "transparent-edit-source", withExtension: "png", subdirectory: "Resources"))
        let original = try Data(contentsOf: url)
        #expect(digest(original) == "543c3e5e71bb254e1ca77ca62c900eda4f1bcf74589d1f61e262bf7827455bd9")
        let pixels = try ImageGenerationImageReader.read(url)
        #expect(pixels.width == 512 && pixels.height == 512)
        #expect(Array(pixels.rgba.prefix(4)) == [255, 255, 255, 0])
        #expect(digest(Data(pixels.rgba)) == "1ba143d368861e3014b5aa142c751e64e24707f6d8be743fe76049799bf948e2")
        let verifiedBytes = try ImageGenerationImageReader.read(original)
        #expect(verifiedBytes == pixels)
        #expect(digest(Data(verifiedBytes.rgba)) == "1ba143d368861e3014b5aa142c751e64e24707f6d8be743fe76049799bf948e2")
        let sliced = (Data(repeating: 0, count: 32) + original).dropFirst(32)
        #expect(try ImageGenerationImageReader.read(sliced) == pixels)
        let preview = try await ImageGenerationMedia().reference(
            url, ticket: .init(owner: UUID(), sequence: 1))
        #expect(preview.pixels.width == 512 && preview.pixels.height == 512)
        #expect(Array(preview.pixels.rgba.prefix(4)) == [255, 255, 255, 0])
        #expect(digest(Data(preview.pixels.rgba)) == "1ba143d368861e3014b5aa142c751e64e24707f6d8be743fe76049799bf948e2")
        #expect(try Data(contentsOf: url) == original)
    }

    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
