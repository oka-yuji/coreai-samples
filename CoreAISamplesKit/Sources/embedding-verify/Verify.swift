import EmbeddingCore
import EmbeddingRuntime
import Foundation

struct VerificationPlan: Decodable {
    struct Sample: Decodable {
        let id: String
        let text: String?
        let role: EmbeddingTextRole?
        let image: String?
        let tokens: [Int]
        let reference: [Float]
        let pixelFile: String?
    }
    let samples: [Sample]
}

struct VerificationRecord: Encodable {
    let id: String
    let tokensMatch: Bool
    let pixelMaxDifference: Float?
    let pixelMeanDifference: Double?
    let cosine: Double?
    let maxDifference: Float?
    let seconds: Double
    let vector: [Float]?
    let passed: Bool
}

@main
enum Verify {
    static func main() async {
        do { try await run() } catch {
            FileHandle.standardError.write(Data("Verification failed: \(error)\n".utf8))
            exit(1)
        }
    }

    static func run() async throws {
        guard CommandLine.arguments.count >= 4 else {
            throw EmbeddingError.invalidBundle("Usage: embedding-verify BUNDLE PLAN OUTPUT [--preprocess]")
        }
        let directory = URL(filePath: CommandLine.arguments[1])
        let plan = try JSONDecoder().decode(
            VerificationPlan.self, from: Data(contentsOf: URL(filePath: CommandLine.arguments[2])))
        let output = URL(filePath: CommandLine.arguments[3])
        let preprocessing = CommandLine.arguments.contains("--preprocess")
        let tokenizer = try EmbeddingTokenizer(directory: directory)
        let model = EmbeddingModel()
        if !preprocessing { try await model.load(directory: directory) }
        var records: [VerificationRecord] = []
        for sample in plan.samples {
            let start = ContinuousClock.now
            var tokens: [Int]
            var vector: EmbeddingVector?
            var pixelMax: Float?
            var pixelMean: Double?
            if let path = sample.image {
                let imageURL = URL(filePath: path)
                let image = try EmbeddingImage(EmbeddingImageReader.read(imageURL))
                tokens = [2, 255_999] + Array(repeating: 258_880, count: image.softTokenCount) + [258_882, 1]
                if let path = sample.pixelFile {
                    let expected = try floats(URL(filePath: path))
                    guard expected.count == image.pixels.count else { throw EmbeddingError.invalidImage }
                    let differences = zip(expected, image.pixels).map { abs($0 - $1) }
                    pixelMax = differences.max()
                    pixelMean = differences.reduce(0.0) { $0 + Double($1) } / Double(differences.count)
                }
                if !preprocessing { vector = try await model.encode(imageURL: imageURL) }
            } else {
                tokens = try tokenizer.tokens(for: sample.text ?? "", role: sample.role ?? .query)
                if !preprocessing { vector = try await model.encode(sample.text ?? "", role: sample.role ?? .query) }
            }
            let match = tokens == sample.tokens
            let reference = try EmbeddingVector(sample.reference)
            let cosine = vector.map { $0.similarity(to: reference) }
            let difference = vector.flatMap { zip($0.values, reference.values).map { abs($0 - $1) }.max() }
            let passed = match && (preprocessing || ((cosine ?? 0) >= 0.999 && (difference ?? 1) <= 0.01))
            let duration = (ContinuousClock.now - start).components
            records.append(
                VerificationRecord(
                    id: sample.id, tokensMatch: match, pixelMaxDifference: pixelMax, pixelMeanDifference: pixelMean,
                    cosine: cosine, maxDifference: difference,
                    seconds: Double(duration.seconds) + Double(duration.attoseconds) / 1e18,
                    vector: vector?.values, passed: passed))
            print(
                "\(passed ? "PASS" : "FAIL") \(sample.id) tokens=\(match) cosine=\(cosine ?? 0) pixelMax=\(pixelMax ?? 0)"
            )
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(records).write(to: output, options: .atomic)
        try await model.unload()
        guard records.allSatisfy(\.passed) else { throw EmbeddingError.invalidVector }
    }

    static func floats(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count.isMultiple(of: 4) else { throw EmbeddingError.invalidImage }
        return data.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: 4).map {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self)))
            }
        }
    }
}
