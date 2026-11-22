import ArgumentParser
import CryptoKit
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
    mainHash: String = "hash", inputs: String = "inputs", prefillChunk: Int = 256, contextLimit: Int = 262_144,
    commit: String = "c0ffee", started: Date = Date(timeIntervalSince1970: 0), id: String = "run-a"
) -> EvaluationRun {
    EvaluationRun(
        id: id, contract: .init(version: "G31-QC-v1", sha256: contract), set: set, role: role,
        asset: .init(name: "bundle", mainHash: mainHash, sourceModelId: "org/model"),
        implementation: .init(commit: commit),
        callShape: .init(
            prefillChunk: prefillChunk, prefillFunction: true, expectFrequentReshapes: true,
            requestedCompute: "default", contextLimit: contextLimit),
        inputs: .init(sha256: inputs, count: count), items: 0, started: started,
        load: .init(cached: true, specializeS: 0, loadS: 1.5))
}

let emptyProgress = #"{"lines":0,"run":"x","sha256":"\#(EvaluationCheckpoint.emptyDigest)"}"#

func digest(_ data: Data) -> String { hex(Data(SHA256.hash(data: data))) }

func completeLines(_ data: Data) -> [Data] {
    data.split(separator: UInt8(ascii: "\n")).map { Data($0) + Data("\n".utf8) }
}

func checkpoint(in folder: URL) throws -> EvaluationCheckpoint {
    let data = try Data(contentsOf: folder.appending(path: "progress.json"))
    return try JSONDecoder.snake.decode(EvaluationCheckpoint.self, from: data)
}

@Suite(.timeLimit(.minutes(1))) struct EvaluationRunTests {
    private let stopAtOnce = FakeEvaluationLogitsSource.Item(rows: [[0, 0, 0, 1]])
    private let stepThenStop = FakeEvaluationLogitsSource.Item(rows: [[0, 1, 0, 0], [0, 0, 0, 1]])

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
        let source = FakeEvaluationLogitsSource(failingScripts)
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
        let first = FakeEvaluationLogitsSource([stopAtOnce, stopAtOnce])
        let partial = try await Evaluation.run(
            inputs, limit: 2, in: folder, record: runRecord(count: 4), source: first, stopTokens: [3])
        #expect(partial.items == 2 && partial.finished != nil)
        #expect(try storedRun(folder).items == 2)
        let second = FakeEvaluationLogitsSource([stopAtOnce, stopAtOnce])
        let resumed = try await Evaluation.run(
            inputs, in: folder, record: runRecord(count: 4, started: .now), source: second, stopTokens: [3])
        #expect(second.calls == ["reset", "prefill 3 true", "reset", "prefill 4 true"])
        #expect(try outputLines(folder).map { $0["id"] as? String } == inputs.map(\.id))
        let stored = try storedRun(folder)
        #expect(resumed.items == 4 && stored.items == 4 && stored.finished != nil)
        #expect(stored.started == Date(timeIntervalSince1970: 0))
    }

    @Test(
        "A run folder whose contract, set, role, asset, input records, call shape, effective context limit or implementation differs is rejected without generating",
        arguments: [
            ("contract", runRecord(count: 3, contract: "other")), ("set", runRecord(count: 3, set: "reserve")),
            ("role", runRecord(count: 3, role: "w8")), ("asset.main_hash", runRecord(count: 3, mainHash: "other")),
            ("inputs", runRecord(count: 3, inputs: "other")), ("call_shape", runRecord(count: 3, prefillChunk: 4096)),
            ("call_shape", runRecord(count: 3, contextLimit: 8192)),
            ("implementation", runRecord(count: 3, commit: "other")),
        ])
    func rejectsAnotherRun(field: String, other: EvaluationRun) async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = base.appending(path: "run")
        let inputs = items(3)
        _ = try await Evaluation.run(
            inputs, limit: 1, in: folder, record: runRecord(count: 3), source: FakeEvaluationLogitsSource([stopAtOnce]),
            stopTokens: [3])
        let before = try Data(contentsOf: folder.appending(path: "outputs.jsonl"))
        let source = FakeEvaluationLogitsSource([stopAtOnce, stopAtOnce])
        let error = await #expect(throws: ValidationError.self) {
            try await Evaluation.run(inputs, in: folder, record: other, source: source, stopTokens: [3])
        }
        #expect(error?.message.contains(field) == true)
        #expect(source.calls.isEmpty)
        #expect(try Data(contentsOf: folder.appending(path: "outputs.jsonl")) == before)
    }

    @Test(
        "A folder left by a start that stopped before writing run.json starts a new run when it is empty or holds only an empty outputs.jsonl and a progress.json of no lines, and is rejected otherwise",
        arguments: [
            ([String: String](), true), (["outputs.jsonl": ""], true),
            (["outputs.jsonl": "", "progress.json": emptyProgress], true), (["outputs.jsonl": "{}\n"], false),
            (["notes.txt": ""], false), (["outputs.jsonl": "", "notes.txt": ""], false),
            (["outputs.jsonl": "", "progress.json": "{}"], false), (["progress.json": emptyProgress], false),
        ])
    func startsInAnUnstartedFolder(files: [String: String], starts: Bool) async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = base.appending(path: "run")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        for (name, text) in files { try Data(text.utf8).write(to: folder.appending(path: name)) }
        let inputs = items(2)
        let source = FakeEvaluationLogitsSource([stopAtOnce, stopAtOnce])
        if starts {
            let run = try await Evaluation.run(
                inputs, in: folder, record: runRecord(count: 2), source: source, stopTokens: [3])
            #expect(run.items == 2 && run.finished != nil)
            #expect(try outputLines(folder).map { $0["id"] as? String } == inputs.map(\.id))
        } else {
            await #expect(throws: ValidationError.self) {
                try await Evaluation.run(
                    inputs, in: folder, record: runRecord(count: 2), source: source, stopTokens: [3])
            }
            #expect(source.calls.isEmpty)
            #expect(try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted() == files.keys.sorted())
        }
    }

    private func fullRun(
        _ folder: URL, _ scripts: [FakeEvaluationLogitsSource.Item], _ inputs: [EvaluationInput], role: String = "fp16",
        id: String = "run-a"
    ) async throws -> Data {
        _ = try await Evaluation.run(
            inputs, in: folder, record: runRecord(count: inputs.count, role: role, id: id),
            source: FakeEvaluationLogitsSource(scripts), stopTokens: [3])
        return try Data(contentsOf: folder.appending(path: "outputs.jsonl"))
    }

    @Test(
        "A run stopped by its limit or by cancellation resumes after the last written item without writing it again, and ends with the outputs.jsonl of a run without a stop and a progress.json that announces no line",
        arguments: [("limit", 2), ("cancel", 0), ("cancel", 2)])
    func resumedRunMatchesAFullRun(stop: String, written: Int) async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let expected = try await fullRun(base.appending(path: "full"), failingScripts, failingItems)
        let folder = base.appending(path: "run")
        let scripts = Array(failingScripts.prefix(written))
        if stop == "cancel" {
            let source = FakeEvaluationLogitsSource(
                scripts + [FakeEvaluationLogitsSource.Item(rows: [], cancelsAtPrefill: true)])
            await #expect(throws: CancellationError.self) {
                try await Evaluation.run(
                    failingItems, in: folder, record: runRecord(count: 4), source: source, stopTokens: [3])
            }
        } else {
            _ = try await Evaluation.run(
                failingItems, limit: written, in: folder, record: runRecord(count: 4),
                source: FakeEvaluationLogitsSource(scripts), stopTokens: [3])
        }
        let identity = try runRecord(count: 4).identityDigest()
        let outputs = try Data(contentsOf: folder.appending(path: "outputs.jsonl"))
        let lines = completeLines(outputs)
        #expect(lines.count == written)
        let stopped =
            stop == "cancel" && written > 0
            ? EvaluationCheckpoint(
                lines: written - 1, sha256: digest(lines.dropLast().reduce(Data(), +)), nextSha256: digest(outputs),
                run: identity)
            : EvaluationCheckpoint(lines: written, sha256: digest(outputs), run: identity)
        #expect(try checkpoint(in: folder) == stopped)
        let source = FakeEvaluationLogitsSource(Array(failingScripts.dropFirst(written)))
        _ = try await Evaluation.run(
            failingItems, in: folder, record: runRecord(count: 4), source: source, stopTokens: [3])
        #expect(source.calls.filter { $0.hasPrefix("prefill") }.count == failingItems.count - written)
        #expect(try Data(contentsOf: folder.appending(path: "outputs.jsonl")) == expected)
        #expect(try checkpoint(in: folder) == EvaluationCheckpoint(lines: 4, sha256: digest(expected), run: identity))
    }

    @Test(
        "A run that stopped after announcing a line in progress.json and before or while writing it resumes and writes that item once",
        arguments: [false, true])
    func resumesBeforeTheAnnouncedLine(cut: Bool) async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let inputs = items(3)
        let expected = try await fullRun(
            base.appending(path: "full"), [stepThenStop, stepThenStop, stepThenStop], inputs)
        let folder = base.appending(path: "run")
        _ = try await Evaluation.run(
            inputs, limit: 2, in: folder, record: runRecord(count: 3),
            source: FakeEvaluationLogitsSource([stepThenStop, stepThenStop]), stopTokens: [3])
        var started = try storedRun(folder)
        started.finished = nil
        try Evaluation.write(started, in: folder)
        let lines = completeLines(expected)
        let outputs = folder.appending(path: "outputs.jsonl")
        try (lines[0] + lines[1] + (cut ? lines[2].prefix(20) : Data())).write(to: outputs)
        try Evaluation.write(
            EvaluationCheckpoint(
                lines: 2, sha256: digest(lines[0] + lines[1]), nextSha256: digest(expected),
                run: started.identityDigest()), in: folder)
        let source = FakeEvaluationLogitsSource([stepThenStop])
        _ = try await Evaluation.run(inputs, in: folder, record: runRecord(count: 3), source: source, stopTokens: [3])
        #expect(source.calls.filter { $0.hasPrefix("prefill") } == ["prefill 3 true"])
        #expect(try Data(contentsOf: outputs) == expected)
    }

    @Test(
        "A run that cannot replace progress.json before a line stops before writing that line, and the folder resumes without writing an item twice"
    )
    func failedAnnouncementKeepsTheFolderResumable() async throws {
        let base = try temporaryFolder()
        let folder = base.appending(path: "run")
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: folder.path)
            try? FileManager.default.removeItem(at: base)
        }
        let expected = try await fullRun(base.appending(path: "full"), failingScripts, failingItems)
        let path = folder.path
        var scripts = failingScripts
        scripts[2].beforePrefill = {
            try? FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: path)
        }
        await #expect(throws: (any Error).self) {
            try await Evaluation.run(
                failingItems, in: folder, record: runRecord(count: 4), source: FakeEvaluationLogitsSource(scripts),
                stopTokens: [3])
        }
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: path)
        #expect(completeLines(try Data(contentsOf: folder.appending(path: "outputs.jsonl"))).count == 2)
        let source = FakeEvaluationLogitsSource(Array(failingScripts.dropFirst(2)))
        _ = try await Evaluation.run(
            failingItems, in: folder, record: runRecord(count: 4), source: source, stopTokens: [3])
        #expect(source.calls.filter { $0.hasPrefix("prefill") }.count == 2)
        #expect(try Data(contentsOf: folder.appending(path: "outputs.jsonl")) == expected)
    }

    @Test(
        "Lines changed by a byte, removed, reordered, added without an announcement or other than the announced one, an older progress.json, files or run.json of another run, even one with the same conditions, and a missing progress.json stop the resumed run without generating or removing anything",
        arguments: [
            "trailing comma", "another stop token", "stop set to limit", "shorter output_ids", "line removed",
            "lines swapped", "another sha256", "an older progress.json", "a line that was not announced",
            "a line other than the announced one", "another run's files",
            "files of another run with the same conditions", "run.json of another run with the same conditions",
            "progress.json removed",
        ])
    func rejectsChangedOutputs(_ change: String) async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = base.appending(path: "run")
        let inputs = items(3)
        let written = try await fullRun(folder, [stepThenStop, stepThenStop, stepThenStop], inputs)
        let identity = try runRecord(count: 3).identityDigest()
        let original = completeLines(written)
        var lines = original.map { String(decoding: $0, as: UTF8.self) }
        var counted = 3
        var recorded = 3
        var message: String?
        switch change {
        case "trailing comma": lines[0] = lines[0].replacing(#""output_ids":[1]"#, with: #""output_ids":[1,]"#)
        case "another stop token": lines[0] = lines[0].replacing(#""stop_token":3"#, with: #""stop_token":2"#)
        case "stop set to limit": lines[0] = lines[0].replacing(#""stop":"stop_token""#, with: #""stop":"limit""#)
        case "shorter output_ids": lines[0] = lines[0].replacing(#""output_ids":[1]"#, with: #""output_ids":[]"#)
        case "line removed":
            lines.remove(at: 1)
            message =
                "outputs.jsonl has 2 complete lines, fewer than the 3 finished items that run.json records; the run folder was changed outside this command."
        case "lines swapped": lines.swapAt(0, 1)
        case "another sha256":
            try Evaluation.write(
                EvaluationCheckpoint(lines: 3, sha256: String(repeating: "0", count: 64), run: identity), in: folder)
        case "an older progress.json":
            try Evaluation.write(
                EvaluationCheckpoint(
                    lines: 1, sha256: digest(original[0]), nextSha256: digest(original[0] + original[1]),
                    run: identity), in: folder)
            recorded = 1
        case "a line that was not announced":
            lines.append(lines[0])
            counted = 4
        case "a line other than the announced one":
            try Evaluation.write(
                EvaluationCheckpoint(
                    lines: 2, sha256: digest(original[0] + original[1]),
                    nextSha256: digest(original[0] + original[1] + original[0]), run: identity), in: folder)
            recorded = 2
        case "another run's files":
            let other = base.appending(path: "other")
            _ = try await fullRun(other, [stepThenStop, stepThenStop, stepThenStop], inputs, role: "w8")
            for name in ["outputs.jsonl", "progress.json"] {
                try FileManager.default.removeItem(at: folder.appending(path: name))
                try FileManager.default.copyItem(at: other.appending(path: name), to: folder.appending(path: name))
            }
            lines = completeLines(try Data(contentsOf: folder.appending(path: "outputs.jsonl"))).map {
                String(decoding: $0, as: UTF8.self)
            }
            message = "progress.json belongs to another run; the run folder was changed outside this command."
        case "files of another run with the same conditions":
            let other = base.appending(path: "other")
            _ = try await fullRun(other, [stepThenStop, stepThenStop, stepThenStop], inputs, id: "run-b")
            for name in ["outputs.jsonl", "progress.json"] {
                try FileManager.default.removeItem(at: folder.appending(path: name))
                try FileManager.default.copyItem(at: other.appending(path: name), to: folder.appending(path: name))
            }
            lines = completeLines(try Data(contentsOf: folder.appending(path: "outputs.jsonl"))).map {
                String(decoding: $0, as: UTF8.self)
            }
            message = "progress.json belongs to another run; the run folder was changed outside this command."
        case "run.json of another run with the same conditions":
            let other = base.appending(path: "other")
            _ = try await fullRun(other, [stepThenStop, stepThenStop, stepThenStop], inputs, id: "run-b")
            try FileManager.default.removeItem(at: folder.appending(path: "run.json"))
            try FileManager.default.copyItem(
                at: other.appending(path: "run.json"), to: folder.appending(path: "run.json"))
            message = "progress.json belongs to another run; the run folder was changed outside this command."
        default:
            try FileManager.default.removeItem(at: folder.appending(path: "progress.json"))
            message = "progress.json is missing; the run folder was changed outside this command."
        }
        let changed = Data(lines.joined().utf8)
        let outputs = folder.appending(path: "outputs.jsonl")
        try changed.write(to: outputs)
        let progress = try? Data(contentsOf: folder.appending(path: "progress.json"))
        let runFile = try Data(contentsOf: folder.appending(path: "run.json"))
        let source = FakeEvaluationLogitsSource([stepThenStop])
        let error = await #expect(throws: ValidationError.self) {
            try await Evaluation.run(inputs, in: folder, record: runRecord(count: 3), source: source, stopTokens: [3])
        }
        #expect(
            error?.message
                == message
                ?? "outputs.jsonl has \(counted) complete lines, which match neither the \(recorded) lines that progress.json records nor the line that it announced; the run folder was changed outside this command."
        )
        #expect(source.calls.isEmpty)
        #expect(try Data(contentsOf: outputs) == changed)
        #expect((try? Data(contentsOf: folder.appending(path: "progress.json"))) == progress)
        #expect(try Data(contentsOf: folder.appending(path: "run.json")) == runFile)
    }

    @Test(
        "A run folder that finished four items, with outputs.jsonl and progress.json put back together to the pair written at two items, stops the resumed run without generating or removing anything"
    )
    func rejectsAnEarlierPairOfTheSameRun() async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = base.appending(path: "run")
        let inputs = items(4)
        _ = try await Evaluation.run(
            inputs, limit: 2, in: folder, record: runRecord(count: 4),
            source: FakeEvaluationLogitsSource([stepThenStop, stepThenStop]), stopTokens: [3])
        let pair = ["outputs.jsonl", "progress.json"]
        let earlier = try pair.map { try Data(contentsOf: folder.appending(path: $0)) }
        _ = try await Evaluation.run(
            inputs, in: folder, record: runRecord(count: 4),
            source: FakeEvaluationLogitsSource([stepThenStop, stepThenStop]), stopTokens: [3])
        #expect(try storedRun(folder).items == 4)
        for (name, data) in zip(pair, earlier) {
            try data.write(to: folder.appending(path: name))
        }
        let names = pair + ["run.json"]
        let before = try names.map { try Data(contentsOf: folder.appending(path: $0)) }
        let source = FakeEvaluationLogitsSource([stepThenStop, stepThenStop])
        let error = await #expect(throws: ValidationError.self) {
            try await Evaluation.run(inputs, in: folder, record: runRecord(count: 4), source: source, stopTokens: [3])
        }
        #expect(
            error?.message
                == "outputs.jsonl has 2 complete lines, fewer than the 4 finished items that run.json records; the run folder was changed outside this command."
        )
        #expect(source.calls.isEmpty)
        #expect(try names.map { try Data(contentsOf: folder.appending(path: $0)) } == before)
    }

    @Test(
        "Recorded lines with an id that is not in the input records or an id twice stop the resumed run",
        arguments: [false, true])
    func rejectsUnknownOrRepeatedIDs(repeated: Bool) async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = base.appending(path: "run")
        let inputs = items(3)
        _ = try await Evaluation.run(
            inputs, limit: 1, in: folder, record: runRecord(count: 3), source: FakeEvaluationLogitsSource([stopAtOnce]),
            stopTokens: [3])
        let outputs = folder.appending(path: "outputs.jsonl")
        let written = try Data(contentsOf: outputs)
        let stranger =
            try JSONEncoder.records.encode(OutputRecord(id: "other", inputSha256: "", inputTokens: 1)) + Data("\n".utf8)
        let changed = written + (repeated ? written : stranger)
        try changed.write(to: outputs)
        try Evaluation.write(
            EvaluationCheckpoint(lines: 2, sha256: digest(changed), run: runRecord(count: 3).identityDigest()),
            in: folder)
        let source = FakeEvaluationLogitsSource([stopAtOnce, stopAtOnce])
        let error = await #expect(throws: ValidationError.self) {
            try await Evaluation.run(inputs, in: folder, record: runRecord(count: 3), source: source, stopTokens: [3])
        }
        #expect(
            error?.message
                == (repeated
                    ? "outputs.jsonl line 2 (id item-1) repeats an id."
                    : "outputs.jsonl line 2 (id other) has no id of the input records."))
        #expect(source.calls.isEmpty)
        #expect(try Data(contentsOf: outputs) == changed)
    }

    @Test(
        "A progress.json whose lines is the largest integer or negative, and a run.json whose items is negative or more than the input records, stop the resumed run without crashing or removing anything",
        arguments: [("lines", Int.max), ("lines", -1), ("items", -1), ("items", 4)])
    func rejectsCountsOutOfRange(field: String, value: Int) async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = base.appending(path: "run")
        let inputs = items(3)
        let cancelled = FakeEvaluationLogitsSource([.init(rows: [], cancelsAtPrefill: true)])
        await #expect(throws: CancellationError.self) {
            try await Evaluation.run(
                inputs, in: folder, record: runRecord(count: 3), source: cancelled, stopTokens: [3])
        }
        var stored = try storedRun(folder)
        var message =
            "outputs.jsonl has 0 complete lines, which match neither the \(value) lines that progress.json records nor the line that it announced; the run folder was changed outside this command."
        if field == "lines" {
            try Evaluation.write(
                EvaluationCheckpoint(
                    lines: value, sha256: EvaluationCheckpoint.emptyDigest,
                    nextSha256: EvaluationCheckpoint.emptyDigest,
                    run: stored.identityDigest()), in: folder)
        } else {
            stored.items = value
            try Evaluation.write(stored, in: folder)
            message =
                "run.json records \(value) finished items, not a count from 0 to 3; the run folder was changed outside this command."
        }
        let names = ["outputs.jsonl", "progress.json", "run.json"]
        let before = try names.map { try Data(contentsOf: folder.appending(path: $0)) }
        let source = FakeEvaluationLogitsSource([stopAtOnce, stopAtOnce, stopAtOnce])
        let error = await #expect(throws: ValidationError.self) {
            try await Evaluation.run(inputs, in: folder, record: runRecord(count: 3), source: source, stopTokens: [3])
        }
        #expect(error?.message == message)
        #expect(source.calls.isEmpty)
        #expect(try names.map { try Data(contentsOf: folder.appending(path: $0)) } == before)
    }

    @Test(
        "A run.json without an id, written by an earlier version of this command, stops the resumed run without removing anything"
    )
    func rejectsRunRecordWithoutID() async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = base.appending(path: "run")
        let inputs = items(3)
        _ = try await fullRun(folder, [stepThenStop, stepThenStop, stepThenStop], inputs)
        let runURL = folder.appending(path: "run.json")
        var fields = try jsonObject(Data(contentsOf: runURL))
        fields["id"] = nil
        try JSONSerialization.data(withJSONObject: fields).write(to: runURL)
        let names = ["outputs.jsonl", "progress.json", "run.json"]
        let before = try names.map { try Data(contentsOf: folder.appending(path: $0)) }
        let source = FakeEvaluationLogitsSource([stepThenStop])
        let error = await #expect(throws: ValidationError.self) {
            try await Evaluation.run(inputs, in: folder, record: runRecord(count: 3), source: source, stopTokens: [3])
        }
        #expect(
            error?.message.hasPrefix(
                "run.json could not be read as a run record of this command, and a run folder of an earlier version cannot be resumed"
            ) == true)
        #expect(source.calls.isEmpty)
        #expect(try names.map { try Data(contentsOf: folder.appending(path: $0)) } == before)
    }

    @Test("Once the time limit has passed, the run ends after the item in progress and records its finish")
    func endsAfterTheItemInProgress() async throws {
        let base = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = base.appending(path: "run")
        let source = FakeEvaluationLogitsSource([stopAtOnce, stopAtOnce, stopAtOnce])
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
            "id": "run-a", "contract": ["version": "G31-QC-v1", "sha256": "contract"], "set": "primary", "role": "fp16",
            "asset": ["name": "bundle", "main_hash": "hash", "source_model_id": "org/model"],
            "implementation": [
                "command": "coreai-verify evaluate", "commit": "c0ffee", "os_version": implementation.osVersion,
                "os_build": implementation.osBuild,
            ],
            "call_shape": [
                "prefill_chunk": 256, "prefill_function": true, "expect_frequent_reshapes": true,
                "requested_compute": "default", "context_limit": 262_144,
            ],
            "inputs": ["sha256": "inputs", "count": 4], "items": 2, "started": "1970-01-01T00:00:00Z",
            "finished": "1970-01-01T00:01:00Z", "load": ["cached": true, "specialize_s": 0, "load_s": 1.5],
        ]
        #expect(encoded == expected)
        #expect(!implementation.osVersion.isEmpty && !implementation.osBuild.isEmpty)
    }
}
