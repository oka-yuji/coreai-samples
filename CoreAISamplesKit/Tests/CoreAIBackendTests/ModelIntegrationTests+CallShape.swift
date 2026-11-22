import Foundation
import Synchronization
import Testing

@_spi(Verification) @testable import CoreAIBackend

let callShapeBundlePaths = (ProcessInfo.processInfo.environment["COREAI_GEMMA4_SHAPE_BUNDLES"] ?? "")
    .split(separator: ":").map(String.init)

extension ModelIntegrationTests {
    private static let callShapeText =
        "A small boat crossed the quiet lake at dawn while the town was still asleep. "
        + "The fisherman counted his nets, looked at the low clouds and turned toward the northern shore. "
        + "By noon the wind had changed, so he rowed back slowly and hummed an old song about the sea. "

    private func callShapeTokens(_ bundle: ModelBundle) async throws -> [Int32] {
        let tokenizer = try await bundle.loadTokenizer()
        let bos = try #require(tokenizer.bosTokenId)
        let text = tokenizer.encode(text: String(repeating: Self.callShapeText, count: 8), addSpecialTokens: false)
        let tokens = [bos] + text
        try #require(tokens.count >= 300)
        return tokens.prefix(300).map(Int32.init)
    }

    private func reportNumber(_ value: Float) -> Any {
        value.isFinite ? Double(value) : String(describing: value)
    }

    private func callShapeRecord(_ path: String) async throws -> [String: Any] {
        let bundle = try ModelBundle(url: URL(filePath: path))
        try #require(bundle.sourceModelID == "google/gemma-4-31B-it")
        let tokens = try await callShapeTokens(bundle)
        let clock = ContinuousClock()
        let loadStart = clock.now
        let engine = try await Engine(bundle: bundle)
        let referenceStart = clock.now
        var reference: [[Float16]] = []
        for token in tokens {
            try await engine.step(token)
            reference.append(engine.logits())
        }
        let sweepStart = clock.now
        var calls: [[Int]] = []
        var prefillDiffs: [Float] = []
        var oneCallDiffs: [Float] = []
        var prefillNonFinite: [Int] = []
        var oneCallNonFinite: [Int] = []
        var shapeFailures: [String] = []
        for length in 1...tokens.count {
            let input = Array(tokens.prefix(length))
            engine.reset()
            let ends = Mutex<[Int]>([])
            try await engine.prefill(input, usePrefillFunction: true) { done in ends.withLock { $0.append(done) } }
            let read = ends.withLock { zip($0.dropFirst(), $0).map { $0 - $1 } }
            let planned =
                engine.hasPrefillFunction
                ? engine.callLengths(from: 0, count: length - 1) + [1] : engine.callLengths(from: 0, count: length)
            var position = 0
            for call in read {
                if CallShapeTests.isAvoided(position: position, length: call) {
                    shapeFailures.append("length \(length): \(read)")
                }
                position += call
            }
            if read != planned { shapeFailures.append("length \(length): \(read), planned \(planned)") }
            let prefilled = engine.logits()
            engine.reset()
            try await engine.readInOneCall(input)
            let oneCall = engine.logits()
            calls.append(read)
            prefillDiffs.append(StateCheck(current: reference[length - 1], reread: prefilled).maxAbsDiff)
            oneCallDiffs.append(StateCheck(current: reference[length - 1], reread: oneCall).maxAbsDiff)
            prefillNonFinite.append(prefilled.count(where: { !$0.isFinite }))
            oneCallNonFinite.append(oneCall.count(where: { !$0.isFinite }))
        }
        let end = clock.now
        let referenceNonFinite = reference.reduce(0) { $0 + $1.count(where: { !$0.isFinite }) }
        let lengths = Array(1...tokens.count)
        let prefillOver = lengths.filter { prefillDiffs[$0 - 1] >= 2.0 }
        let oneCallOverOutside = lengths.filter { !(97...112).contains($0) && oneCallDiffs[$0 - 1] >= 2.0 }
        let oneCallOverInside = (97...112).filter { oneCallDiffs[$0 - 1] >= 2.0 }
        let nonFinite = referenceNonFinite + prefillNonFinite.reduce(0, +) + oneCallNonFinite.reduce(0, +)
        #expect(shapeFailures.isEmpty, "\(bundle.name): \(shapeFailures)")
        #expect(nonFinite == 0, "\(bundle.name): \(nonFinite) non-finite logits")
        #expect(prefillOver.isEmpty, "\(bundle.name): prefill differs by 2.0 or more at lengths \(prefillOver)")
        #expect(
            oneCallOverOutside.isEmpty,
            "\(bundle.name): one call differs by 2.0 or more at lengths \(oneCallOverOutside)")
        #expect(!oneCallOverInside.isEmpty, "\(bundle.name): no one-call input of 97 to 112 tokens reached 2.0")
        func largest(_ diffs: [Float], in range: [Int]) -> [String: Any] {
            let length = range.max { diffs[$0 - 1] < diffs[$1 - 1] } ?? 0
            return ["length": length, "max_abs_diff": length == 0 ? NSNull() : reportNumber(diffs[length - 1])]
        }
        let prefillLargest = largest(prefillDiffs, in: lengths)
        let insideLargest = largest(oneCallDiffs, in: Array(97...112))
        let outsideLargest = largest(oneCallDiffs, in: lengths.filter { !(97...112).contains($0) })
        print(
            "call-shape \(bundle.name) prefill_max=\(prefillLargest) one_call_97_112_max=\(insideLargest)"
                + " one_call_other_max=\(outsideLargest) one_call_97_112_at_least_2=\(oneCallOverInside)"
                + " non_finite=\(nonFinite) sweep_s=\((end - sweepStart) / .seconds(1))")
        let rule: Any =
            engine.callShapeRule.map {
                [
                    "avoided_ends": [$0.avoidedEnds.lowerBound, $0.avoidedEnds.upperBound],
                    "shortest_affected_length": $0.shortestAffectedLength,
                ] as [String: Any]
            } ?? NSNull()
        let record: [String: Any] = [
            "name": bundle.name, "source_model_id": bundle.sourceModelID ?? NSNull(),
            "main_hash": bundle.mainHash ?? NSNull(), "has_prefill_function": engine.hasPrefillFunction,
            "prefill_chunk_size": engine.prefillChunkSize, "call_shape_rule": rule, "tokens": tokens.map(Int.init),
            "cached": engine.loadStats.cached, "specialize_s": engine.loadStats.specializeSeconds,
            "load_s": (referenceStart - loadStart) / .seconds(1),
            "reference_s": (sweepStart - referenceStart) / .seconds(1),
            "sweep_s": (end - sweepStart) / .seconds(1), "reference_non_finite": referenceNonFinite,
            "non_finite": nonFinite, "prefill_largest": prefillLargest, "one_call_largest_97_112": insideLargest,
            "one_call_largest_other": outsideLargest, "one_call_at_least_2_97_112": oneCallOverInside,
            "prefill_at_least_2": prefillOver, "one_call_at_least_2_other": oneCallOverOutside,
            "rows": lengths.map { length -> [String: Any] in
                [
                    "length": length, "prefill_calls": calls[length - 1],
                    "prefill_max_abs_diff": reportNumber(prefillDiffs[length - 1]),
                    "one_call_max_abs_diff": reportNumber(oneCallDiffs[length - 1]),
                    "prefill_non_finite": prefillNonFinite[length - 1],
                    "one_call_non_finite": oneCallNonFinite[length - 1],
                ]
            },
        ]
        engine.unload()
        return record
    }

    @Test(
        "On a Gemma 4 31B asset, every input of 1 to 300 tokens read by prefill stays below 2.0 from single-token reading at its last position, while one call differs by 2.0 or more for some input of 97 to 112 tokens and for no other length",
        .enabled(if: !callShapeBundlePaths.isEmpty))
    func gemma4CallShapes() async throws {
        let report = try #require(
            ProcessInfo.processInfo.environment["COREAI_GEMMA4_SHAPE_REPORT"],
            "COREAI_GEMMA4_SHAPE_REPORT must name a new report file")
        try #require(!FileManager.default.fileExists(atPath: report), "The report file already exists: \(report)")
        var assets: [[String: Any]] = []
        for path in callShapeBundlePaths { assets.append(try await callShapeRecord(path)) }
        let data = try JSONSerialization.data(
            withJSONObject: ["threshold": 2.0, "assets": assets], options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(filePath: report), options: .withoutOverwriting)
    }
}
