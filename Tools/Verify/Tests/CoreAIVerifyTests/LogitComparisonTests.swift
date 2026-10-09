import ArgumentParser
import Foundation
import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct LogitComparisonTests {
    @Test("Finite comparisons retain bit identity, Float differences and token-match state")
    func finiteValues() throws {
        #expect(LogitComparison(own: [2, 1], official: [2, 1]) == .finite(identical: true, maxAbsDiff: 0))
        #expect(LogitComparison(own: [2, 1], official: [2, 0]) == .finite(identical: false, maxAbsDiff: 1))
        #expect(LogitComparison(own: [.zero], official: [-Float16.zero]) == .finite(identical: false, maxAbsDiff: 0))
        #expect(LogitComparison(own: [2], official: [2, 1]) == .finite(identical: false, maxAbsDiff: 0))
        var record = CompareRecord(prefill: "prefill-function", promptTokens: 2)
        record.tokens = [0]
        record.officialTokens = [0]
        record.match = true
        record.stop = "eos"
        let accepted = record.observeLogits(own: [2, 1], official: [2, 1])
        #expect(accepted)
        #expect(record.ownNonFiniteLogitCount == 0)
        #expect(record.officialNonFiniteLogitCount == 0)
        #expect(record.identicalSteps == 1)
        #expect(record.maxAbsDiff == 0)
        #expect(record.tokens == [0] && record.officialTokens == [0] && record.match)
        #expect(record.stop == "eos" && record.error == nil)
    }

    @Test(
        "Identical NaN and infinities are rejected instead of counted as identical steps",
        arguments: [UInt16(0x7e00), 0x7c00, 0xfc00])
    func identicalNonFinite(_ bits: UInt16) throws {
        let values: [Float16] = [2, Float16(bitPattern: bits)]
        #expect(LogitComparison(own: values, official: values) == .nonFinite(own: 1, official: 1))
        var record = CompareRecord(prefill: "prefill-function", promptTokens: 2)
        let accepted = record.observeLogits(own: values, official: values)
        #expect(!accepted)
        #expect(record.ownNonFiniteLogitCount == 1 && record.officialNonFiniteLogitCount == 1)
        #expect(record.identicalSteps == 0 && record.maxAbsDiff == 0)
        #expect(record.error != nil && record.stop == "non-finite")
        _ = try JSONEncoder().encode(record)
    }

    @Test("Each complete side is checked independently including elements beyond zip", arguments: [true, false])
    func independentSides(_ ownIsInvalid: Bool) {
        let invalid: [Float16] = [2, .nan, .infinity, -.infinity]
        let own: [Float16] = ownIsInvalid ? invalid : [2]
        let official: [Float16] = ownIsInvalid ? [2] : invalid
        #expect(
            LogitComparison(own: own, official: official)
                == .nonFinite(own: ownIsInvalid ? 3 : 0, official: ownIsInvalid ? 0 : 3))
    }

    @Test("A later invalid step preserves earlier finite results without adding a successful step")
    func laterStep() throws {
        var record = CompareRecord(prefill: "prefill-function", promptTokens: 2)
        record.tokens = [0, 0]
        record.officialTokens = [0, 0]
        record.match = true
        let identicalAccepted = record.observeLogits(own: [2, 1], official: [2, 1])
        #expect(identicalAccepted)
        let differentAccepted = record.observeLogits(own: [2, 1], official: [2, 0])
        #expect(differentAccepted)
        let nonFiniteAccepted = record.observeLogits(own: [2, .nan], official: [2, 1])
        #expect(!nonFiniteAccepted)
        #expect(record.identicalSteps == 1 && record.maxAbsDiff == 1)
        #expect(record.ownNonFiniteLogitCount == 1 && record.officialNonFiniteLogitCount == 0)
        #expect(record.tokens == [0, 0] && record.officialTokens == [0, 0] && record.match)
        _ = try JSONEncoder().encode(record)
    }

    @Test("The shared comparison output boundary emits a finite record and exits successfully")
    func finiteExit() async throws {
        let outcome = await #expect(
            processExitsWith: .exitCode(0), observing: [\.standardOutputContent, \.standardErrorContent]
        ) {
            var record = CompareRecord(prefill: "prefill-function", promptTokens: 2)
            record.prompt = "finite"
            record.tokens = [0]
            record.officialTokens = [0]
            record.match = true
            _ = record.observeLogits(own: [2, 1], official: [2, 1])
            do {
                try emitComparison(record)
                CoreAIVerify.exit()
            } catch {
                CoreAIVerify.exit(withError: error)
            }
        }
        let result = try #require(outcome)
        let data = Data(result.standardOutputContent)
        let row = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(row["own_non_finite_logit_count"] as? Int == 0)
        #expect(row["official_non_finite_logit_count"] as? Int == 0)
        #expect(row["error"] == nil)
        print("COMPARISON_FIXTURE finite " + data.base64EncodedString())
    }

    @Test("The shared comparison output boundary emits serializable error evidence before CLI failure")
    func nonFiniteExit() async throws {
        let outcome = await #expect(
            processExitsWith: .exitCode(1), observing: [\.standardOutputContent, \.standardErrorContent]
        ) {
            var record = CompareRecord(prefill: "prefill-function", promptTokens: 2)
            record.prompt = "nonfinite"
            record.tokens = [0]
            record.officialTokens = [0]
            record.match = true
            _ = record.observeLogits(own: [2, 1], official: [2, 1])
            _ = record.observeLogits(own: [2, .nan, -.infinity], official: [2, .infinity])
            do {
                try emitComparison(record)
                CoreAIVerify.exit()
            } catch {
                CoreAIVerify.exit(withError: error)
            }
        }
        let result = try #require(outcome)
        let data = Data(result.standardOutputContent)
        let row = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(row["own_non_finite_logit_count"] as? Int == 2)
        #expect(row["official_non_finite_logit_count"] as? Int == 1)
        #expect(row["identical_steps"] as? Int == 1)
        #expect(row["max_abs_diff"] as? Int == 0)
        #expect(row["match"] as? Bool == true)
        #expect((row["error"] as? String)?.contains("own=2, official=1") == true)
        #expect(String(decoding: result.standardErrorContent, as: UTF8.self).contains("Non-finite logits"))
        print("COMPARISON_FIXTURE nonfinite " + data.base64EncodedString())
    }
}
