import ArgumentParser
import Foundation
import Testing

@testable import coreai_verify

func temporaryFolder() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appending(path: "evaluate-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
    return url
}

func outputLines(_ folder: URL) throws -> [[String: Any]] {
    try Data(contentsOf: folder.appending(path: "outputs.jsonl")).split(separator: UInt8(ascii: "\n")).map(jsonObject)
}

func runRecord(
    count: Int, contract: String = "contract", set: String = "primary", role: String = "fp16",
    mainHash: String = "hash", inputs: String = "inputs", prefillChunk: Int = 256, commit: String = "c0ffee",
    started: Date = Date(timeIntervalSince1970: 0)
) -> EvaluationRun {
    EvaluationRun(
        contract: .init(version: "G31-QC-v1", sha256: contract), set: set, role: role,
        asset: .init(name: "bundle", mainHash: mainHash, sourceModelId: "org/model"),
        implementation: .init(commit: commit),
        callShape: .init(
            prefillChunk: prefillChunk, prefillFunction: true, expectFrequentReshapes: true,
            requestedCompute: "default"),
        inputs: .init(sha256: inputs, count: count), items: 0, started: started,
        load: .init(cached: true, specializeS: 0, loadS: 1.5))
}

@Suite(.timeLimit(.minutes(1))) struct EvaluationRunTests {
    private let stopAtOnce = ScriptedLogits.Item(rows: [[0, 0, 0, 1]])

    private func items(_ count: Int) -> [EvaluationInput] {
        (1...count).map { evaluationInput("item-\($0)", Array(repeating: Int32($0), count: $0)) }
    }

    private func storedRun(_ folder: URL) throws -> EvaluationRun {
        let decoder = JSONDecoder.snake
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(EvaluationRun.self, from: Data(contentsOf: folder.appending(path: "run.json")))
    }

    @Test("Items that end with non-finite logits or an error do not stop the run, and the next item is generated")
    func errorsDoNotStopTheRun() async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = base.appending(path: "run")
        let source = ScriptedLogits(failingScripts)
        let run = try await Evaluation.run(
            failingItems, in: folder, record: runRecord(count: failingItems.count), source: source, stopTokens: [3])
        let lines = try outputLines(folder)
        #expect(run.items == 4)
        #expect(lines.map { $0["id"] as? String } == failingItems.map(\.id))
        #expect(lines.map { $0.keys.contains("error") } == [true, true, true, false])
        #expect(lines.map { $0["non_finite"] as? Int } == [3, 0, 0, 0])
        #expect(lines[3]["stop"] as? String == "stop_token" && lines[3]["output_ids"] as? [Int] == [1])
    }

    @Test("A resumed run skips the items already written and continues with the rest")
    func resumeSkipsDoneItems() async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = base.appending(path: "run")
        let inputs = items(4)
        let first = ScriptedLogits([stopAtOnce, stopAtOnce])
        let partial = try await Evaluation.run(
            inputs, limit: 2, in: folder, record: runRecord(count: 4), source: first, stopTokens: [3])
        #expect(partial.items == 2 && partial.finished != nil)
        #expect(try storedRun(folder).items == 2)
        let second = ScriptedLogits([stopAtOnce, stopAtOnce])
        let resumed = try await Evaluation.run(
            inputs, in: folder, record: runRecord(count: 4, started: .now), source: second, stopTokens: [3])
        #expect(second.calls == ["reset", "prefill 3 true", "reset", "prefill 4 true"])
        #expect(try outputLines(folder).map { $0["id"] as? String } == inputs.map(\.id))
        let stored = try storedRun(folder)
        #expect(resumed.items == 4 && stored.items == 4 && stored.finished != nil)
        #expect(stored.started == Date(timeIntervalSince1970: 0))
    }

    @Test(
        "A run folder whose contract, set, role, asset, input records, call shape or implementation differs is rejected without generating",
        arguments: [
            ("contract", runRecord(count: 3, contract: "other")), ("set", runRecord(count: 3, set: "reserve")),
            ("role", runRecord(count: 3, role: "w8")), ("asset.main_hash", runRecord(count: 3, mainHash: "other")),
            ("inputs", runRecord(count: 3, inputs: "other")), ("call_shape", runRecord(count: 3, prefillChunk: 4096)),
            ("implementation", runRecord(count: 3, commit: "other")),
        ])
    func rejectsAnotherRun(field: String, other: EvaluationRun) async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = base.appending(path: "run")
        let inputs = items(3)
        _ = try await Evaluation.run(
            inputs, limit: 1, in: folder, record: runRecord(count: 3), source: ScriptedLogits([stopAtOnce]),
            stopTokens: [3])
        let before = try Data(contentsOf: folder.appending(path: "outputs.jsonl"))
        let source = ScriptedLogits([stopAtOnce, stopAtOnce])
        let error = await #expect(throws: ValidationError.self) {
            try await Evaluation.run(inputs, in: folder, record: other, source: source, stopTokens: [3])
        }
        #expect(error?.message.contains(field) == true)
        #expect(source.calls.isEmpty)
        #expect(try Data(contentsOf: folder.appending(path: "outputs.jsonl")) == before)
    }

    @Test("A last line cut before its newline is dropped and its item is generated again")
    func dropsCutLastLine() async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = base.appending(path: "run")
        let inputs = items(3)
        _ = try await Evaluation.run(
            inputs, limit: 2, in: folder, record: runRecord(count: 3),
            source: ScriptedLogits([stopAtOnce, stopAtOnce]), stopTokens: [3])
        let outputs = folder.appending(path: "outputs.jsonl")
        let written = try Data(contentsOf: outputs)
        try written.dropLast(5).write(to: outputs)
        let source = ScriptedLogits([stopAtOnce, stopAtOnce])
        _ = try await Evaluation.run(inputs, in: folder, record: runRecord(count: 3), source: source, stopTokens: [3])
        #expect(source.calls.filter { $0.hasPrefix("prefill") } == ["prefill 2 true", "prefill 3 true"])
        #expect(try outputLines(folder).map { $0["id"] as? String } == inputs.map(\.id))
        #expect(try Data(contentsOf: outputs).last == UInt8(ascii: "\n"))
    }

    @Test(
        "A run folder with an output record whose id is not in the input records or appears twice is rejected",
        arguments: [false, true])
    func rejectsUnknownOrRepeatedIDs(repeated: Bool) async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = base.appending(path: "run")
        let inputs = items(3)
        _ = try await Evaluation.run(
            inputs, limit: 1, in: folder, record: runRecord(count: 3), source: ScriptedLogits([stopAtOnce]),
            stopTokens: [3])
        let outputs = folder.appending(path: "outputs.jsonl")
        let written = try Data(contentsOf: outputs)
        let stranger =
            try JSONEncoder.records.encode(OutputRecord(id: "other", inputSha256: "", inputTokens: 1)) + Data("\n".utf8)
        try (written + (repeated ? written : stranger)).write(to: outputs)
        let source = ScriptedLogits([stopAtOnce, stopAtOnce])
        await #expect(throws: ValidationError.self) {
            try await Evaluation.run(inputs, in: folder, record: runRecord(count: 3), source: source, stopTokens: [3])
        }
        #expect(source.calls.isEmpty)
    }

    @Test("Once the time limit has passed, the run ends after the item in progress and records its finish")
    func endsAfterTheItemInProgress() async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = base.appending(path: "run")
        let source = ScriptedLogits([stopAtOnce, stopAtOnce, stopAtOnce])
        let result = try await Evaluation.run(
            items(3), in: folder, record: runRecord(count: 3), source: source, stopTokens: [3], maxMinutes: 0)
        #expect(result.items == 1 && result.finished != nil)
        #expect(try outputLines(folder).count == 1)
        #expect(try storedRun(folder).finished != nil)
        #expect(source.calls.filter { $0.hasPrefix("prefill") }.count == 1)
    }

    @Test("The command checks the input records before it loads the model and leaves no run folder when one differs")
    func commandRejectsInputsBeforeLoading() async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let bundle = base.appending(path: "bundle")
        try FileManager.default.createDirectory(
            at: bundle.appending(path: "fake.aimodel"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: bundle.appending(path: "tokenizer"), withIntermediateDirectories: true)
        try Data(
            #"{"name":"fake","assets":{"main":"fake.aimodel"},"language":{"vocab_size":10,"max_context_length":64}}"#
                .utf8
        ).write(to: bundle.appending(path: "metadata.json"))
        try Data("{}".utf8).write(to: bundle.appending(path: "tokenizer/tokenizer.json"))
        try Data(repeating: 1, count: 32).write(to: bundle.appending(path: "fake.aimodel/main.hash"))
        let contract = base.appending(path: "contract.json")
        try Data(#"{"version":"v","roles":{"fp16":"x"},"generation":{"eos_token_id":[3]}}"#.utf8).write(to: contract)
        let inputs = base.appending(path: "inputs.jsonl")
        try Data(#"{"id":"a","input_ids":[1,2],"input_sha256":"00","max_new_tokens":4}"#.utf8).write(to: inputs)
        let output = base.appending(path: "run")
        let command = try Evaluate.parse([
            "--model", bundle.path, "--inputs", inputs.path, "--contract", contract.path, "--set", "primary", "--role",
            "fp16", "--commit", "c0ffee", "--output", output.path,
        ])
        let error = await #expect(throws: ValidationError.self) { try await command.run() }
        #expect(error?.message.hasPrefix("a: input_sha256 differs") == true)
        #expect(!FileManager.default.fileExists(atPath: output.path))
    }

    @Test(
        "A limit or a time limit of zero or less is rejected",
        arguments: [["--limit", "0"], ["--max-minutes", "0"], ["--max-minutes", "-1"], ["--max-minutes", "nan"]])
    func rejectsLimits(_ extra: [String]) {
        let required = [
            "--model", "/not-loaded", "--inputs", "/in.jsonl", "--contract", "/contract.json", "--set", "primary",
            "--role", "fp16", "--commit", "c0ffee", "--output", "/out",
        ]
        #expect(throws: (any Error).self) { try Evaluate.parse(required + extra) }
    }

    @Test("The run record has the keys of a fixed example")
    func runRecordMatchesFixedExample() throws {
        var record = runRecord(count: 4)
        record.items = 2
        record.finished = Date(timeIntervalSince1970: 60)
        let encoded = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder.records.encode(record)) as? NSDictionary)
        let implementation = record.implementation
        let expected: NSDictionary = [
            "contract": ["version": "G31-QC-v1", "sha256": "contract"], "set": "primary", "role": "fp16",
            "asset": ["name": "bundle", "main_hash": "hash", "source_model_id": "org/model"],
            "implementation": [
                "command": "coreai-verify evaluate", "commit": "c0ffee", "os_version": implementation.osVersion,
                "os_build": implementation.osBuild,
            ],
            "call_shape": [
                "prefill_chunk": 256, "prefill_function": true, "expect_frequent_reshapes": true,
                "requested_compute": "default",
            ],
            "inputs": ["sha256": "inputs", "count": 4], "items": 2, "started": "1970-01-01T00:00:00Z",
            "finished": "1970-01-01T00:01:00Z", "load": ["cached": true, "specialize_s": 0, "load_s": 1.5],
        ]
        #expect(encoded == expected)
        #expect(!implementation.osVersion.isEmpty && !implementation.osBuild.isEmpty)
    }
}
