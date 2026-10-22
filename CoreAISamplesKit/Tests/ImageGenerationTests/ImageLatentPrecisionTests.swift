import CryptoKit
import Foundation
import ImageGenerationCore
import Testing

struct ImageLatentPrecisionTests {
    private struct Golden: Decodable {
        struct Sample: Decodable {
            let precision: ImageGenerationPrecision
            let normalize: String
            let denormalize: String
        }
        let input: [Float]
        let mean: [Float]
        let standardDeviation: [Float]
        let cases: [Sample]
    }

    @Test("VAE statistics and intermediate rounding reproduce the independent Torch formulas in all three precisions")
    func scalingMatchesOriginal() throws {
        let file = try #require(
            Bundle.module.url(forResource: "latent-scaling", withExtension: "json", subdirectory: "Resources"))
        let golden = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: file))
        #expect(golden.cases.map { $0.precision.rawValue }.sorted() == ["bfloat16", "float16", "float32"])
        let input = (0..<ImageEuler.latentCount).map { golden.input[$0 % golden.input.count] }
        let scaling = try ImageLatentScaling(
            mean: (0..<64).map { golden.mean[$0 % golden.mean.count] },
            standardDeviation: (0..<64).map { golden.standardDeviation[$0 % golden.standardDeviation.count] })
        for sample in golden.cases {
            let normalized = try scaling.normalize(input, precision: sample.precision)
            let denormalized = try scaling.denormalize(input, precision: sample.precision)
            #expect(hash(normalized) == sample.normalize, "\(sample.precision)")
            #expect(hash(denormalized) == sample.denormalize, "\(sample.precision)")
        }
    }

    @Test("A scale that rounds to zero cannot silently produce nonfinite latents")
    func scaleUnderflow() throws {
        let scaling = try ImageLatentScaling(
            mean: [Float](repeating: 0, count: 64), standardDeviation: [Float](repeating: 1e-20, count: 64))
        #expect(throws: ImageGenerationError.nonFinite) {
            try scaling.normalize([Float](repeating: 1, count: ImageEuler.latentCount), precision: .float16)
        }
    }

    private func hash(_ values: [Float]) -> String {
        values.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
    }
}
