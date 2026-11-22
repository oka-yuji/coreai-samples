import ArgumentParser
import Foundation
import Synchronization
import Testing

@testable import coreai_verify

struct ScriptedFailure: Error {}

final class ScriptedLogits: LogitsSource, Sendable {
    struct Item: Sendable {
        var rows: [[Float16]]
        var failsAtPrefill = false
        var failsAtStep: Int?
    }

    private struct State {
        var items: [Item]
        var current = Item(rows: [])
        var position = 0
        var calls: [String] = []
    }

    private let state: Mutex<State>

    init(_ items: [Item]) {
        state = Mutex(State(items: items))
    }

    var calls: [String] { state.withLock { $0.calls } }

    func reset() {
        state.withLock { $0.calls.append("reset") }
    }

    func prefill(_ input: [Int32], usePrefillFunction: Bool) async throws {
        try state.withLock { state in
            state.calls.append("prefill \(input.count) \(usePrefillFunction)")
            state.current = state.items.removeFirst()
            state.position = 0
            if state.current.failsAtPrefill { throw ScriptedFailure() }
        }
    }

    func step(_ token: Int32) async throws {
        try state.withLock { state in
            state.calls.append("step \(token)")
            state.position += 1
            if state.position == state.current.failsAtStep { throw ScriptedFailure() }
        }
    }

    func logits() -> [Float16] {
        state.withLock { $0.current.rows[$0.position] }
    }
}

func evaluationInput(_ id: String, _ ids: [Int32], maxNewTokens: Int = 8) -> EvaluationInput {
    EvaluationInput(id: id, inputIds: ids, inputSha256: tokensDigest(ids.map(Int.init)), maxNewTokens: maxNewTokens)
}

func jsonObject(_ data: Data) throws -> [String: Any] {
    try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
}

@Suite(.timeLimit(.minutes(1))) struct EvaluationRecordTests {
    @Test(
        "Each step's first top-2 id is the generated token, a tie goes to the lowest id, and the probabilities are the softmax over the whole vocabulary"
    )
    func greedyTopTwo() async throws {
        let source = ScriptedLogits([.init(rows: [[1, 3, 3, 0], [0, 0, 1, 0], [2, 2, 2, 2], [0, 0, 0, 4]])])
        let record = try await evaluationInput("a", [5, 6, 7]).generate(with: source, stopTokens: [3])
        #expect(record.top2.map(\.first) == [1, 2, 0, 3])
        #expect(record.top2.map(\.second) == [2, 0, 1, 0])
        #expect(record.outputIds == [1, 2, 0])
        let tied = 1 / (2 + exp(-2.0) + exp(-3.0))
        let expected = [
            (tied, tied), (exp(1.0) / (exp(1.0) + 3), 1 / (exp(1.0) + 3)), (0.25, 0.25),
            (exp(4.0) / (exp(4.0) + 3), 1 / (exp(4.0) + 3)),
        ]
        #expect(record.top2.count == expected.count)
        for (top, (first, second)) in zip(record.top2, expected) {
            #expect(abs(top.firstProbability - first) < 1e-12)
            #expect(abs(top.secondProbability - second) < 1e-12)
        }
        #expect(source.calls == ["reset", "prefill 3 true", "step 1", "step 2", "step 0"])
    }

    @Test(
        "The two candidates are ranked by their logits, so a second logit far below the largest stays second even when its probability rounds to zero"
    )
    func ranksByLogits() {
        #expect(
            LogitsRow([0, 1, 1024])
                == .finite(TopTwo(first: 2, firstProbability: 1, second: 1, secondProbability: 0)))
        #expect(
            LogitsRow([1024, 0, 1, 1])
                == .finite(TopTwo(first: 0, firstProbability: 1, second: 2, secondProbability: 0)))
    }

    @Test("A top-2 entry is written as an array of the first id, its probability, the second id and its probability")
    func topTwoWireForm() throws {
        let entry = TopTwo(first: 7, firstProbability: 0.5, second: 2, secondProbability: 0.25)
        #expect(String(decoding: try JSONEncoder().encode(entry), as: UTF8.self) == "[7,0.5,2,0.25]")
    }

    @Test(
        "A stop-token item has stop_token and its id but not the token in output_ids, a limit item has no stop_token, and both have one top-2 entry per generated token"
    )
    func stopKinds() async throws {
        let rows: [[Float16]] = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        let stopped = try await evaluationInput("stop", [9]).generate(
            with: ScriptedLogits([.init(rows: rows)]), stopTokens: [3])
        #expect(stopped.stop == "stop_token" && stopped.stopToken == 3)
        #expect(stopped.outputIds == [1, 2])
        #expect(stopped.top2.map(\.first) == [1, 2, 3])
        let source = ScriptedLogits([.init(rows: rows)])
        let limited = try await evaluationInput("limit", [9], maxNewTokens: 2).generate(with: source, stopTokens: [3])
        #expect(limited.stop == "limit" && limited.stopToken == nil)
        #expect(limited.outputIds == [1, 2])
        #expect(limited.top2.map(\.first) == [1, 2])
        #expect(source.calls == ["reset", "prefill 1 true", "step 1"])
        let fields: Set = ["id", "input_sha256", "input_tokens", "output_ids", "stop", "top2", "non_finite"]
        #expect(Set(try jsonObject(JSONEncoder.records.encode(stopped)).keys) == fields.union(["stop_token"]))
        #expect(Set(try jsonObject(JSONEncoder.records.encode(limited)).keys) == fields)
    }

    @Test(
        "Non-finite logits and errors thrown by prefill or step end the item with non_finite and error and keep what was generated, and an item without an error has no error field"
    )
    func errorsEndTheItem() async throws {
        let source = ScriptedLogits(failingScripts)
        var records: [OutputRecord] = []
        for item in failingItems { records.append(try await item.generate(with: source, stopTokens: [3])) }
        #expect(records.map(\.nonFinite) == [3, 0, 0, 0])
        #expect(records.map { $0.error != nil } == [true, true, true, false])
        #expect(records.map(\.outputIds) == [[1], [], [2], [1]])
        #expect(records.map { $0.top2.map(\.first) } == [[1], [], [2], [1, 3]])
        #expect(records.map(\.stop) == ["limit", "limit", "limit", "stop_token"])
        #expect(records.map(\.stopToken) == [nil, nil, nil, 3])
        let written = try records.map { try jsonObject(JSONEncoder.records.encode($0)) }
        #expect(written.map { $0.keys.contains("error") } == [true, true, true, false])
    }

    @Test(
        "Input records whose input_sha256 differs from their tokens, with no tokens, an id outside the vocabulary, no output budget or a repeated id, and a file without records, are rejected",
        arguments: [
            (#"{"id":"a","input_ids":[1,2],"input_sha256":"00","max_new_tokens":4}"#, "a: input_sha256 differs"),
            (#"{"id":"a","input_ids":[],"input_sha256":"00","max_new_tokens":4}"#, "a: input_ids must hold"),
            (#"{"id":"a","input_ids":[1,10],"input_sha256":"00","max_new_tokens":4}"#, "a: input_ids must hold"),
            (#"{"id":"a","input_ids":[-1],"input_sha256":"00","max_new_tokens":4}"#, "a: input_ids must hold"),
            (
                #"{"id":"a","input_ids":[1],"input_sha256":"\#(tokensDigest([1]))","max_new_tokens":0}"#,
                "a: max_new_tokens must be"
            ),
            (
                #"{"id":"a","input_ids":[1],"input_sha256":"\#(tokensDigest([1]))","max_new_tokens":1}"# + "\n"
                    + #"{"id":"a","input_ids":[1],"input_sha256":"\#(tokensDigest([1]))","max_new_tokens":1}"#,
                "The input id a appears more than once."
            ),
            ("\n", "The input file has no records."),
            (#"{"id":"a"}"#, "Input record 1 could not be read"),
        ])
    func rejectsInvalidInputs(text: String, message: String) {
        let error = #expect(throws: ValidationError.self) {
            try EvaluationInput.read(Data(text.utf8), vocabulary: 10)
        }
        #expect(error?.message.hasPrefix(message) == true)
    }

    @Test("Valid input records are read in file order")
    func readsInputs() throws {
        let lines = [1, 2].map {
            #"{"id":"i\#($0)","input_ids":[\#($0),3],"input_sha256":"\#(tokensDigest([$0, 3]))","max_new_tokens":5}"#
        }
        let read = try EvaluationInput.read(Data((lines.joined(separator: "\n") + "\n").utf8), vocabulary: 10)
        #expect(
            read == [evaluationInput("i1", [1, 3], maxNewTokens: 5), evaluationInput("i2", [2, 3], maxNewTokens: 5)])
    }
}

let failingItems = [
    evaluationInput("nan", [1, 2]), evaluationInput("prefill", [3]), evaluationInput("step", [4]),
    evaluationInput("clean", [5]),
]

let failingScripts: [ScriptedLogits.Item] = [
    .init(rows: [[0, 1, 0, 0], [.nan, .infinity, 0, -.infinity], [0, 0, 0, 1]]),
    .init(rows: [], failsAtPrefill: true),
    .init(rows: [[0, 0, 1, 0], [0, 1, 0, 0]], failsAtStep: 1),
    .init(rows: [[0, 1, 0, 0], [0, 0, 0, 1]]),
]
