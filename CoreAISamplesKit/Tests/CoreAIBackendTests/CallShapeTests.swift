import Foundation
import Testing

@testable import CoreAIBackend

@Suite struct CallShapeTests {
    static let gemma31B = "google/gemma-4-31B-it"
    static let otherSources: [String?] = [
        nil, "", "google/gemma-4-31b-it", "google/gemma-4-31B", "google/gemma-4-E4B-it", "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3-0.6B",
    ]

    static func isAvoided(position: Int, length: Int) -> Bool {
        length >= 8 && (97...112).contains(position + length)
    }

    static func previousLengths(count: Int, limit: Int) -> [Int] {
        stride(from: 0, to: count, by: limit).map { min(limit, count - $0) }
    }

    static func ruleFailures(from position: Int, count: Int, limit: Int, lengths: [Int]) -> String? {
        var end = position
        for length in lengths {
            let longest = min(limit, position + count - end)
            guard (1...max(1, longest)).contains(length), !isAvoided(position: end, length: length),
                stride(from: length + 1, through: longest, by: 1).allSatisfy({ isAvoided(position: end, length: $0) })
            else { return "position \(position) count \(count) limit \(limit): \(lengths)" }
            end += length
        }
        return end == position + count ? nil : "position \(position) count \(count) limit \(limit): \(lengths)"
    }

    @Test(
        "For every position from 0 to 300, remaining count from 1 to 300 and chunk limit of 8, 256 or 4096, no call of 8 or more tokens ends at 97 to 112 positions, the calls add up to the remaining count and each is the longest call within the limit that the rule allows",
        arguments: [8, 256, 4096])
    func gemmaRuleAvoidsTheFailingShapes(limit: Int) throws {
        let rule = try #require(Engine.callShapeRule(modelID: Self.gemma31B))
        var failures: [String] = []
        for position in 0...300 {
            for count in 1...300 {
                let lengths = Engine.callLengths(from: position, count: count, limit: limit, rule: rule)
                if let failure = Self.ruleFailures(from: position, count: count, limit: limit, lengths: lengths) {
                    failures.append(failure)
                }
            }
        }
        #expect(failures.isEmpty, "\(failures.count) cases, first: \(failures.prefix(5))")
    }

    @Test(
        "From position 0, main alone reads 100 tokens as 96 + 4 and 112 tokens as 96 + 7 + 7 + 2, and with a prefill function the prefill function reads 96 + 3 and 96 + 7 + 7 + 1 before main reads the last token",
        arguments: [256, 2048, 4096])
    func gemmaRuleSplitsTheExamples(limit: Int) throws {
        let rule = try #require(Engine.callShapeRule(modelID: Self.gemma31B))
        #expect(Engine.callLengths(from: 0, count: 100, limit: limit, rule: rule) == [96, 4])
        #expect(Engine.callLengths(from: 0, count: 112, limit: limit, rule: rule) == [96, 7, 7, 2])
        #expect(Engine.callLengths(from: 0, count: 99, limit: limit, rule: rule) == [96, 3])
        #expect(Engine.callLengths(from: 0, count: 111, limit: limit, rule: rule) == [96, 7, 7, 1])
        #expect(Engine.callLengths(from: 99, count: 1, limit: limit, rule: rule) == [1])
        #expect(Engine.callLengths(from: 111, count: 1, limit: limit, rule: rule) == [1])
    }

    @Test(
        "An asset without the rule keeps the previous call lengths on the prefill-function path and on the main path",
        arguments: otherSources)
    func otherAssetsKeepThePreviousLengths(modelID: String?) {
        let rule = Engine.callShapeRule(modelID: modelID)
        #expect(rule?.avoidedEnds == nil)
        var failures: [String] = []
        for limit in [1, 8, 256, 2048, 4096] {
            for position in [0, 1, 89, 96, 97, 104, 112, 113, 300, 4096] {
                for count in 1...300 {
                    let main = Engine.callLengths(from: position, count: count, limit: limit, rule: rule)
                    let prefill =
                        Engine.callLengths(from: position, count: count - 1, limit: limit, rule: rule)
                        + Engine.callLengths(from: position + count - 1, count: 1, limit: limit, rule: rule)
                    if main != Self.previousLengths(count: count, limit: limit)
                        || prefill != Self.previousLengths(count: count - 1, limit: limit) + [1]
                    {
                        failures.append("position \(position) count \(count) limit \(limit): \(main) \(prefill)")
                    }
                }
            }
        }
        #expect(failures.isEmpty, "\(failures.count) cases, first: \(failures.prefix(5))")
    }

    @Test(
        "The rule applies to an asset whose source model ID is google/gemma-4-31B-it and to no asset with another or no source model ID",
        arguments: [(gemma31B, true)] + otherSources.map { ($0, false) } as [(String?, Bool)])
    func ruleAppliesOnlyToGemma4_31B(modelID: String?, applies: Bool) {
        #expect((Engine.callShapeRule(modelID: modelID) != nil) == applies)
    }

    @Test(
        "The shipped table holds one rule, for google/gemma-4-31B-it: calls of 8 or more tokens avoid ends of 97 to 112"
    )
    func shippedTableHoldsTheGemmaRule() {
        #expect(Engine.callShapeRules.keys.sorted() == [Self.gemma31B])
        #expect(Engine.callShapeRules[Self.gemma31B]?.avoidedEnds == 97...112)
        #expect(Engine.callShapeRules[Self.gemma31B]?.shortestAffectedLength == 8)
    }

    @Test(
        "A bundle gets the rule through source.hf_model_id from the table, and neither its name nor a missing or other source selects it",
        arguments: [
            ("gemma4_31b_l6_slicepos_fp16", #"{"hf_model_id": "google/gemma-4-31B-it"}"#, true),
            ("renamed-copy", #"{"model_definition": "torch", "hf_model_id": "google/gemma-4-31B-it"}"#, true),
            ("gemma-4-31B-it", nil, false), ("gemma-4-31B-it", #"{"hf_model_id": "Qwen/Qwen3.5-0.8B"}"#, false),
            ("gemma-4-31B-it", #"{"hf_model_id": 31}"#, false),
        ] as [(String, String?, Bool)])
    func bundleSelectsTheRuleBySource(name: String, source: String?, applies: Bool) throws {
        let folder = try PrefillChunkTests.bundle(name: name, source: source, hash: nil)
        defer { try? FileManager.default.removeItem(at: folder) }
        let bundle = try ModelBundle(url: folder)
        #expect((Engine.callShapeRule(modelID: bundle.sourceModelID) != nil) == applies)
        #expect(Engine.callShapeRule(modelID: bundle.sourceModelID, rules: [:])?.avoidedEnds == nil)
    }

    @Test(
        "The prefix calls of coreai-verify logits contain no call of 8 or more tokens ending at 97 to 112 positions with the rule, and keep the previous pieces without it",
        arguments: [256, 2048, 4096])
    func verifyPrefixFollowsTheRule(limit: Int) throws {
        let rule = try #require(Engine.callShapeRule(modelID: Self.gemma31B))
        var failures: [String] = []
        for start in 0...4_200 {
            let lengths = Engine.callLengths(from: 0, count: start, limit: limit, rule: rule)
            if let failure = Self.ruleFailures(from: 0, count: start, limit: limit, lengths: lengths) {
                failures.append(failure)
            }
            for modelID in Self.otherSources {
                let other = Engine.callLengths(
                    from: 0, count: start, limit: limit, rule: Engine.callShapeRule(modelID: modelID))
                if other != Self.previousLengths(count: start, limit: limit) {
                    failures.append("start \(start) limit \(limit) \(modelID ?? "nil"): \(other)")
                }
            }
        }
        #expect(failures.isEmpty, "\(failures.count) cases, first: \(failures.prefix(5))")
    }
}
