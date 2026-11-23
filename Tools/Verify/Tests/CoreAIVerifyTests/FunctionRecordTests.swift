import Foundation
import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct FunctionRecordTests {
    private func record(cacheQueryError: String? = nil, differingCalls: [Int] = []) -> FunctionRecord {
        let ids = FunctionTensor(
            name: "input_ids", dtype: .int32, shape: [1, 2], data: FunctionFixture.integers([7, 8]))
        let tokens = FunctionTensor(
            name: "draft_tokens", dtype: .int32, shape: [1, 3], data: FunctionFixture.integers([1, 2, 3]))
        let scores = FunctionTensor(name: "scores", dtype: .float16, shape: [1, 2], data: Data([1, 2, 3, 4]))
        return FunctionRecord(
            mainHash: "abcd", function: "draft", inputs: [.init(ids)],
            outputs: [.init(tokens, file: "draft_tokens.bin"), .init(scores, file: "scores.bin")],
            load: FunctionLoad(
                requestedCompute: "default", expectFrequentReshapes: true, cached: false,
                cacheQueryError: cacheQueryError, specializeS: 0.5, loadS: 1.5),
            callS: [0.25, 0.125], differingCalls: differingCalls)
    }

    @Test(
        "A tensor entry has its byte count, which is the product of its shape and the width of its dtype, and the SHA-256 of its bytes",
        arguments: [
            (FunctionDtype.int32, [1, 3], FunctionFixture.integers([1, 2, 3])),
            (.float16, [1, 2], Data([1, 2, 3, 4])),
            (.float32, [2, 2, 1], Data(repeating: 9, count: 16)),
        ])
    func describesATensor(dtype: FunctionDtype, shape: [Int], data: Data) {
        let entry = FunctionRecord.Tensor(FunctionTensor(name: "t", dtype: dtype, shape: shape, data: data))
        #expect(entry.bytes == data.count)
        #expect(entry.bytes == shape.reduce(1, *) * dtype.width)
        #expect(entry.sha256 == digest(data))
        #expect(entry.file == nil)
        #expect(
            FunctionRecord.Tensor(FunctionTensor(name: "t", dtype: dtype, shape: shape, data: data), file: "t.bin").file
                == "t.bin")
    }

    @Test(
        "The record has the keys and the values of a fixed example, and a cache query error appears only when there was one"
    )
    func recordMatchesFixedExample() throws {
        let expectedInput: NSDictionary = [
            "name": "input_ids", "dtype": "int32", "shape": [1, 2], "bytes": 8,
            "sha256": digest(FunctionFixture.integers([7, 8])),
        ]
        let expectedTokens: NSDictionary = [
            "name": "draft_tokens", "dtype": "int32", "shape": [1, 3], "file": "draft_tokens.bin", "bytes": 12,
            "sha256": digest(FunctionFixture.integers([1, 2, 3])),
        ]
        let expectedScores: NSDictionary = [
            "name": "scores", "dtype": "float16", "shape": [1, 2], "file": "scores.bin", "bytes": 4,
            "sha256": digest(Data([1, 2, 3, 4])),
        ]
        let expected: NSDictionary = [
            "record": "function", "byte_order": "little", "main_hash": "abcd", "function": "draft",
            "inputs": [expectedInput], "outputs": [expectedTokens, expectedScores], "requested_compute": "default",
            "expect_frequent_reshapes": true, "cached": false, "specialize_s": 0.5, "load_s": 1.5,
            "call_s": [0.25, 0.125], "repeat_identical": true, "repeat_differing_calls": [] as [Int],
        ]
        let encoded = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder.records.encode(record())) as? NSDictionary)
        #expect(encoded == expected)
        let failed = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder.records.encode(record(cacheQueryError: "no entry")))
                as? NSDictionary)
        #expect(failed.value(forKey: "cache_query_error") as? String == "no entry")
        #expect(
            Set(failed.allKeys.compactMap { $0 as? String })
                == Set(expected.allKeys.compactMap { $0 as? String }).union(["cache_query_error"]))
    }

    @Test("The record says whether the repeated calls agreed from the numbers of the calls that differed")
    func derivesTheRepeatResult() {
        #expect(record().repeatIdentical)
        #expect(record().repeatDifferingCalls.isEmpty)
        #expect(!record(differingCalls: [2]).repeatIdentical)
        #expect(record(differingCalls: [2, 3]).repeatDifferingCalls == [2, 3])
    }

    @Test("A record read back from its JSON equals the record that was written")
    func roundTrips() throws {
        let original = record(cacheQueryError: "no entry", differingCalls: [3])
        let decoded = try JSONDecoder.snake.decode(FunctionRecord.self, from: JSONEncoder.records.encode(original))
        #expect(decoded == original)
    }

    @Test(
        "A failure record holds the name of the record and the kind of the error and nothing else",
        arguments: [
            (FunctionFailureRecord.Kind.incomplete, "incomplete"), (.load, "load"), (.call, "call"), (.write, "write"),
            (.repeatMismatch, "repeat_mismatch"), (.cancelled, "cancelled"),
        ])
    func failureRecordHoldsOnlyTheKind(kind: FunctionFailureRecord.Kind, name: String) throws {
        let encoded = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder.records.encode(FunctionFailureRecord(kind: kind)))
                as? NSDictionary)
        #expect(encoded == ["record": "function-failure", "kind": name] as NSDictionary)
    }
}
