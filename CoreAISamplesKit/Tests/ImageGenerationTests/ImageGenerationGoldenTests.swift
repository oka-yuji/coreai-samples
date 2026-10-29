import CryptoKit
import Foundation
import ImageGenerationCore
import ImageGenerationRuntime
import Testing

struct ImageGenerationGoldenTests {
    @Test("Prompt positions and expanded image layouts match the frozen original for all inputs and the exact limit")
    func layoutsMatchOriginal() throws {
        let golden = try NativeGolden.load()
        for sample in golden.cases where !sample.overflow {
            let layout = try ImagePromptLayout(
                tokens: #require(sample.tokens), drop: #require(sample.drop), hasReference: sample.hasReference)
            #expect(layout.positions == sample.positions)
            #expect(fingerprint(layout.textMask) == sample.textMask)
            let denoiser = try ImageDenoiserLayout(imageSlots: layout.imageSlots)
            #expect(denoiser.length == sample.expanded)
            #expect(fingerprint(denoiser.textIndex.map(\.littleEndian)) == sample.layoutHashes?["text_index"])
            #expect(fingerprint(denoiser.imageIndex.map(\.littleEndian)) == sample.layoutHashes?["image_index"])
            #expect(fingerprint(denoiser.imageMask) == sample.layoutHashes?["image_mask"])
            #expect(fingerprint(denoiser.attentionMask) == sample.layoutHashes?["mask"])
            #expect(fingerprint(denoiser.targetMask) == sample.layoutHashes?["target"])
            #expect(denoiser.targetIndices.count == 1_024)
            if let rotary = sample.rotary {
                for (name, values) in [
                    ("textCosine", layout.textCosine), ("textSine", layout.textSine),
                    ("denoiserCosine", denoiser.cosine), ("denoiserSine", denoiser.sine),
                ] {
                    let expected = try NativeGolden.floats(#require(rotary[name]))
                    #expect(values.count == expected.count)
                    let maximum = zip(values, expected).map { abs($0 - $1) }.max() ?? .infinity
                    #expect(maximum <= golden.rotaryAbsoluteTolerance)
                }
            }
        }
    }

    @Test(
        "The native tokenizer matches Japanese English image slots and exact prompt boundaries",
        .enabled(if: ProcessInfo.processInfo.environment["QMI_TOKENIZER"] != nil))
    func tokenizerMatchesOriginal() throws {
        let directory = URL(filePath: try #require(ProcessInfo.processInfo.environment["QMI_TOKENIZER"]))
        let tokenizer = try ImageGenerationTokenizer(directory: directory)
        #expect(throws: ImageGenerationError.inputTooLarge(262_145)) {
            try tokenizer.prepare(String(repeating: "x", count: 262_145), hasReference: false)
        }
        for sample in try NativeGolden.load().cases {
            if sample.overflow {
                #expect(throws: ImageGenerationError.promptTooLong(513)) {
                    try tokenizer.prepare(sample.prompt, hasReference: sample.hasReference)
                }
            } else {
                let layout = try tokenizer.prepare(sample.prompt, hasReference: sample.hasReference)
                #expect(layout.tokens.map(Int.init) == sample.tokens)
                #expect(layout.drop == sample.drop)
            }
        }
    }

    @Test("Every Euler step returns to the original prediction dtype including BF16 and FP16 rounding")
    func schedulerMatchesOriginal() throws {
        for sample in try NativeGolden.load().scheduler {
            let precision = try #require(ImageGenerationPrecision(rawValue: sample.precision))
            func expand(_ values: [Float]) -> [Float] {
                (0..<ImageEuler.latentCount).map { values[$0 % values.count] }
            }
            var latents = expand(sample.initial)
            let prediction = expand(sample.prediction)
            for (index, expected) in sample.outputs.enumerated() {
                #expect(
                    try ImageEuler.modelTimestep(index: index, precision: precision) == sample.modelTimesteps[index])
                latents = try ImageEuler.step(
                    latents: latents, prediction: prediction, index: index, precision: precision)
                let matches = latents == expand(expected)
                #expect(
                    matches,
                    "\(sample.precision), step \(index): \(Array(latents.prefix(8))) / \(Array(expected.prefix(8)))")
            }
        }
    }

    @Test("Saved noise and actual source predictions reproduce every scheduler step at all supported host dtypes")
    func savedNoiseReplayMatchesOriginal() throws {
        let golden = try NativeGolden.load().savedNoise
        let noise = try NativeGolden.floats(golden.file)
        let predictions = try golden.predictions.map(NativeGolden.floats)
        for replay in golden.replay {
            let precision = try #require(ImageGenerationPrecision(rawValue: replay.precision))
            var latents = try ImageEuler.initialLatents(noise: noise, precision: precision)
            for (index, prediction) in predictions.enumerated() {
                latents = try ImageEuler.step(
                    latents: latents, prediction: prediction, index: index, precision: precision)
                #expect(fingerprint(latents) == replay.hashes[index], "\(replay.precision), step \(index)")
            }
        }
    }

    @Test("Vision and DeepStack features occupy only the source image slots and hidden states remove the system prefix")
    func conditioningMatchesOriginal() throws {
        let sample = try #require(NativeGolden.load().cases.first { $0.id == "e06" })
        let expected = try #require(sample.conditioning)
        let layout = try ImagePromptLayout(
            tokens: #require(sample.tokens), drop: #require(sample.drop), hasReference: true)
        let features = (0..<4).map { tap in
            (0..<(256 * 4_096)).map { Float(($0 % 1_024) + (tap + 1) * 1_000) }
        }
        let conditioning = try ImageTextConditioning(layout: layout, visionOutputs: features)
        #expect(fingerprint(conditioning.vision) == expected["vision"])
        #expect(fingerprint(conditioning.deepstack) == expected["deepstack"])
        let hidden = (0..<(512 * 4_096)).map { Float($0 % 997) }
        #expect(fingerprint(try layout.condition(fromHidden: hidden)) == expected["text"])
        #expect(throws: ImageGenerationError.invalidShape) {
            try ImageTextConditioning(layout: layout, visionOutputs: nil)
        }
        let denoiser = try ImageDenoiserLayout(imageSlots: layout.imageSlots)
        let reference = [Float](repeating: 1, count: ImageEuler.latentCount)
        let target = [Float](repeating: 2, count: ImageEuler.latentCount)
        #expect(try denoiser.packedLatents(target: target, reference: reference) == reference + target)
        let output = (0..<(denoiser.capacity * 64)).map(Float.init)
        let prediction = try denoiser.prediction(fromOutput: output)
        #expect(prediction.first == Float(try #require(denoiser.targetIndices.first) * 64))
        #expect(prediction.last == Float((try #require(denoiser.targetIndices.last) + 1) * 64 - 1))
        #expect(throws: ImageGenerationError.invalidShape) {
            try denoiser.packedLatents(target: target, reference: nil)
        }
    }

    @Test("Invalid tokens shapes and nonfinite latents fail before a model can execute")
    func invalidInputsFailEarly() throws {
        #expect(throws: ImageGenerationError.promptTooLong(513)) {
            try ImagePromptLayout(tokens: Array(repeating: 1, count: 513), drop: 0, hasReference: false)
        }
        #expect(throws: ImageGenerationError.invalidTokens) {
            try ImagePromptLayout(tokens: [1, 151_936], drop: 0, hasReference: false)
        }
        #expect(throws: ImageGenerationError.invalidTokens) {
            try ImagePromptLayout(tokens: [1, ImagePromptLayout.imageToken], drop: 0, hasReference: true)
        }
        #expect(throws: ImageGenerationError.invalidShape) {
            try ImageDenoiserLayout(imageSlots: Array(repeating: true, count: 256), capacity: 1_536)
        }
        let zeros = [Float](repeating: 0, count: ImageEuler.latentCount)
        var invalid = zeros
        invalid[31] = .nan
        #expect(throws: ImageGenerationError.nonFinite) {
            try ImageEuler.step(latents: invalid, prediction: zeros, index: 0, precision: .float32)
        }
        #expect(throws: ImageGenerationError.invalidShape) {
            try ImageEuler.step(latents: zeros, prediction: zeros, index: 8, precision: .float32)
        }
        #expect(try ImageGenerationPrecision.bfloat16.rounded(1.00390625) == 1)
        #expect(try ImageGenerationPrecision.bfloat16.rounded(1.01171875) == 1.015625)
        #expect(try ImageGenerationPrecision.float16.rounded(1 + 1 / 2_048) == 1)
    }

    @Test("The versioned native RNG is deterministic without claiming PyTorch seed equivalence")
    func nativeNoiseHasItsOwnVersion() throws {
        let first = try ImageGenerationNoise.make(seed: 42)
        #expect(first.count == ImageEuler.latentCount && first.allSatisfy(\.isFinite))
        #expect(first == (try ImageGenerationNoise.make(seed: 42)))
        #expect(first != (try ImageGenerationNoise.make(seed: 43)))
        #expect(ImageGenerationNoise.version != "torch-2.9.0-cpu-randn-packed-v1")
        #expect(throws: ImageGenerationError.invalidSeed) { try ImageGenerationNoise.make(seed: .max) }
    }

    private func fingerprint<Value>(_ values: [Value]) -> String {
        values.withUnsafeBytes { SHA256.hash(data: $0).map { String(format: "%02x", $0) }.joined() }
    }
}

struct NativeGolden: Decodable {
    struct Sample: Decodable {
        let id: String
        let prompt: String
        let hasReference: Bool
        let overflow: Bool
        let tokens: [Int]?
        let drop: Int?
        let positions: [[Int32]]?
        let expanded: Int?
        let textMask: String?
        let layoutHashes: [String: String]?
        let rotary: [String: String]?
        let conditioning: [String: String]?
    }

    struct Scheduler: Decodable {
        let precision: String
        let initial: [Float]
        let prediction: [Float]
        let outputs: [[Float]]
        let modelTimesteps: [Float]
    }

    struct SavedNoise: Decodable {
        struct Replay: Decodable {
            let precision: String
            let hashes: [String]
        }
        let file: String
        let predictions: [String]
        let replay: [Replay]
    }

    let format: Int
    let sourceRevision: String
    let fixtureManifestSHA256: String
    let cases: [Sample]
    let scheduler: [Scheduler]
    let rotaryAbsoluteTolerance: Float
    let scaling: ImageLatentScaling
    let savedNoise: SavedNoise

    static func resource(_ name: String) throws -> URL {
        try #require(Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Resources"))
    }

    static func load() throws -> Self {
        let golden = try JSONDecoder().decode(Self.self, from: Data(contentsOf: resource("native-golden.json")))
        let fixtureIDs = ["g", "e"].flatMap { prefix in
            (1...15).map { prefix + String(format: "%02d", $0) }
        }
        let boundaries = ["wide-token", "limit-generate", "limit-edit", "over-generate", "over-edit"]
        let expected = Set(fixtureIDs + boundaries)
        try #require(golden.format == 1)
        try #require(golden.sourceRevision == "fc13ba1986fcb905854dc8de2f52eee395f49b5b")
        try #require(
            golden.fixtureManifestSHA256 == "b26820d8a34afd01f4f2c8e1ab2b0b66d951fe38e5b0a6b79104c1f555fe3ee9")
        try #require(golden.cases.count == expected.count && Set(golden.cases.map(\.id)) == expected)
        return golden
    }

    static func floats(_ name: String) throws -> [Float] {
        let data = try Data(contentsOf: resource(name))
        try #require(data.count.isMultiple(of: 4))
        return data.withUnsafeBytes { bytes in
            (0..<(bytes.count / 4)).map { index in
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)))
            }
        }
    }
}
