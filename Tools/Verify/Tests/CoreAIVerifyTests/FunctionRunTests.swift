import Darwin
import Foundation
import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct FunctionRunTests {
    @Test(
        "An output that exists as a folder, a file, a link to a folder or a link to nothing is rejected before anything is opened, read or written",
        arguments: ["folder", "file", "link", "dangling"])
    func rejectsAnExistingOutput(kind: String) async throws {
        let fixture = try FunctionFixture(listText: "not json")
        defer { fixture.remove() }
        switch kind {
        case "folder": try FileManager.default.createDirectory(at: fixture.output, withIntermediateDirectories: false)
        case "file": try Data([1]).write(to: fixture.output)
        case "link": try FileManager.default.createSymbolicLink(at: fixture.output, withDestinationURL: fixture.folder)
        default:
            try FileManager.default.createSymbolicLink(at: fixture.output, withDestinationURL: fixture.url("missing"))
        }
        let source = FakeFunctionSource()
        await #expect(throws: FunctionError.outputExists(fixture.output.path(percentEncoded: false))) {
            try await fixture.run(source)
        }
        #expect(source.calls.isEmpty)
        if kind == "folder" { #expect(try fixture.names().isEmpty) }
        if kind == "file" { #expect(try Data(contentsOf: fixture.output) == Data([1])) }
    }

    @Test("An inputs file that is not valid is rejected before the asset is opened, and no output is made")
    func rejectsAnInvalidList() async throws {
        let fixture = try FunctionFixture(listText: #"{"inputs":{}}"#)
        defer { fixture.remove() }
        let source = FakeFunctionSource()
        await #expect(throws: FunctionError.invalidList("inputs is not an array")) { try await fixture.run(source) }
        #expect(source.calls.isEmpty)
        #expect(!FunctionFixture.exists(fixture.output))
    }

    @Test("An inputs file that is a named pipe or missing is rejected without waiting, before the asset is opened")
    func rejectsAnUnreadableList() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let source = FakeFunctionSource()
        try FileManager.default.removeItem(at: fixture.list)
        await #expect(
            throws: FunctionError.cannot("read the inputs file", String(cString: strerror(ENOENT)))
        ) { try await fixture.run(source) }
        try #require(mkfifo(fixture.list.path(percentEncoded: false), 0o600) == 0)
        await #expect(throws: FunctionError.notRegularFile("the inputs file")) { try await fixture.run(source) }
        #expect(source.calls.isEmpty)
        #expect(!FunctionFixture.exists(fixture.output))
    }

    @Test(
        "An input file that is named through a link to another folder is rejected before the asset is opened, also when a combining mark follows the slash",
        arguments: ["keys.bin", "\u{0301}keys.bin"])
    func rejectsAFileThroughAFolderLink(name: String) async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let elsewhere = fixture.url("elsewhere")
        try FileManager.default.createDirectory(at: elsewhere, withIntermediateDirectories: false)
        let keys = FunctionFixture.inputs[2]
        try FunctionFixture.data(keys).write(to: elsewhere.appending(path: name))
        try FileManager.default.createSymbolicLink(at: fixture.url("link"), withDestinationURL: elsewhere)
        try fixture.writeList(
            Array(FunctionFixture.inputs.dropLast())
                + [FunctionInput(name: keys.name, dtype: keys.dtype, shape: keys.shape, file: "link/" + name)])
        let source = FakeFunctionSource()
        let error = await #expect(throws: FunctionError.self) { try await fixture.run(source) }
        guard case .invalidList(let reason) = try #require(error) else {
            Issue.record("The error is not an invalid list: \(String(describing: error))")
            return
        }
        #expect(reason.hasPrefix("inputs[2].file must be the name of a file in the folder of the inputs file"))
        #expect(source.calls.isEmpty)
        #expect(!FunctionFixture.exists(fixture.output))
    }

    @Test("A function that the asset does not have is rejected after the asset is opened and before it is loaded")
    func rejectsAMissingFunction() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let source = FakeFunctionSource()
        await #expect(throws: FunctionError.noSuchFunction("other", available: ["draft"])) {
            try await fixture.run(source, function: "other")
        }
        #expect(source.calls == ["describe"])
        #expect(!FunctionFixture.exists(fixture.output))
    }

    @Test("A function with state is rejected before the data files are read, and it is not loaded")
    func rejectsAFunctionWithState() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let stateful = FunctionSignature(name: "draft", inputs: FunctionFixture.signature.inputs, stateNames: ["cache"])
        let source = FakeFunctionSource(.init(functions: [stateful]))
        await #expect(throws: FunctionError.hasStates("draft", ["cache"])) { try await fixture.run(source) }
        #expect(source.calls == ["describe"])
        #expect(!FunctionFixture.exists(fixture.output))
    }

    @Test("A list that leaves out, adds or repeats an input is rejected before the function is loaded")
    func rejectsAnotherSetOfInputs() async throws {
        let extra = FunctionInput(name: "mask", dtype: .int32, shape: [1], file: "mask.bin")
        let cases: [([FunctionInput], FunctionError)] = [
            (Array(FunctionFixture.inputs.dropLast()), .missingInput("keys")),
            (FunctionFixture.inputs + [extra], .unknownInput("mask")),
            (FunctionFixture.inputs + [FunctionFixture.inputs[1]], .duplicateInput("hidden_states")),
        ]
        for (inputs, error) in cases {
            let fixture = try FunctionFixture(inputs: inputs)
            defer { fixture.remove() }
            let source = FakeFunctionSource()
            await #expect(throws: error) { try await fixture.run(source) }
            #expect(source.calls == ["describe"])
            #expect(!FunctionFixture.exists(fixture.output))
        }
    }

    @Test(
        "An input of another dtype, rank or fixed dimension than the asset is rejected before the data files are read")
    func rejectsAnotherType() async throws {
        let ids = FunctionFixture.inputs[0]
        let hidden = FunctionFixture.inputs[1]
        let keys = FunctionFixture.inputs[2]
        let cases: [(FunctionInput, FunctionError)] = [
            (
                FunctionInput(name: hidden.name, dtype: .float32, shape: hidden.shape, file: hidden.file),
                .dtypeMismatch(input: "hidden_states", asset: "Float16", given: .float32)
            ),
            (
                FunctionInput(name: keys.name, dtype: .float16, shape: [1, 2, 5], file: keys.file),
                .rankMismatch(input: "keys", asset: 4, given: 3)
            ),
            (
                FunctionInput(name: ids.name, dtype: .int32, shape: [1, 2], file: ids.file),
                .dimensionMismatch(input: "input_ids", axis: 1, asset: 1, given: 2)
            ),
        ]
        for (changed, error) in cases {
            let inputs = FunctionFixture.inputs.map { $0.name == changed.name ? changed : $0 }
            let fixture = try FunctionFixture(inputs: inputs)
            defer { fixture.remove() }
            try FileManager.default.removeItem(at: fixture.url("hidden_states.bin"))
            let source = FakeFunctionSource()
            await #expect(throws: error) { try await fixture.run(source) }
            #expect(source.calls == ["describe"])
            #expect(!FunctionFixture.exists(fixture.output))
        }
    }

    @Test(
        "A data file that is missing, of another size, a folder, a named pipe or a link is rejected before the output is made and the function is loaded",
        arguments: ["missing", "short", "long", "folder", "pipe", "link"])
    func rejectsABadDataFile(kind: String) async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let path = fixture.url("input_ids.bin")
        try FileManager.default.removeItem(at: path)
        var error = FunctionError.notRegularFile("input_ids.bin")
        switch kind {
        case "missing": error = .cannot("read input_ids.bin", String(cString: strerror(ENOENT)))
        case "short":
            try Data([1, 2, 3]).write(to: path)
            error = .fileSize("input_ids.bin", expected: 4, actual: 3)
        case "long":
            try Data([1, 2, 3, 4, 5]).write(to: path)
            error = .fileSize("input_ids.bin", expected: 4, actual: 5)
        case "folder": try FileManager.default.createDirectory(at: path, withIntermediateDirectories: false)
        case "pipe": try #require(mkfifo(path.path(percentEncoded: false), 0o600) == 0)
        default:
            try Data([1, 2, 3, 4]).write(to: fixture.url("target.bin"))
            try FileManager.default.createSymbolicLink(at: path, withDestinationURL: fixture.url("target.bin"))
        }
        let source = FakeFunctionSource()
        await #expect(throws: error) { try await fixture.run(source) }
        #expect(source.calls == ["describe"])
        #expect(!FunctionFixture.exists(fixture.output))
    }

    @Test("An output folder that cannot be made is rejected before the function is loaded")
    func rejectsAnOutputThatCannotBeMade() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let source = FakeFunctionSource()
        let output = fixture.url("missing-parent/out")
        await #expect(throws: FunctionError.cannot("create the output folder", String(cString: strerror(ENOENT)))) {
            try await FunctionRun.run(
                function: "draft", inputs: fixture.list, output: output, repeatCount: 1, source: source)
        }
        #expect(source.calls == ["describe"])
        #expect(!FunctionFixture.exists(output))
    }

    @Test(
        "The record and the files agree: the inputs and outputs have their byte counts and SHA-256 values, the output files hold the bytes of the last call, and the same record is in function.json"
    )
    func recordMatchesTheFiles() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let source = FakeFunctionSource()
        let record = try await fixture.run(source, repeatCount: 3)
        #expect(source.calls == ["describe", "load draft", "call", "call", "call"])
        let expectedInputs = FunctionFixture.inputs.map {
            FunctionTensor(name: $0.name, dtype: $0.dtype, shape: $0.shape, data: FunctionFixture.data($0))
        }
        #expect(source.received == [expectedInputs, expectedInputs, expectedInputs])
        #expect(record.inputs.map(\.name) == FunctionFixture.inputs.map(\.name))
        for (entry, input) in zip(record.inputs, FunctionFixture.inputs) {
            let bytes = try fixture.contents(of: input.file)
            #expect(entry.bytes == bytes.count && entry.bytes == input.shape.reduce(1, *) * input.dtype.width)
            #expect(entry.sha256 == digest(bytes))
            #expect(entry.dtype == input.dtype && entry.shape == input.shape && entry.file == nil)
        }
        #expect(record.outputs.map(\.name) == ["draft_tokens", "scores"])
        for entry in record.outputs {
            let file = try #require(entry.file)
            let bytes = try fixture.contents(of: "out/\(file)")
            #expect(file == "\(entry.name).bin")
            #expect(entry.bytes == bytes.count && entry.bytes == entry.shape.reduce(1, *) * entry.dtype.width)
            #expect(entry.sha256 == digest(bytes))
        }
        let expectedOutputs = FunctionFixture.outputs()
        #expect(record.outputs.map(\.dtype) == expectedOutputs.map(\.dtype))
        #expect(record.outputs.map(\.shape) == expectedOutputs.map(\.shape))
        #expect(try fixture.contents(of: "out/draft_tokens.bin") == FunctionFixture.integers([1, 2, 3]))
        #expect(try fixture.contents(of: "out/scores.bin") == Data([1, 2, 3, 4]))
        let written = try fixture.contents(of: "out/function.json")
        let line = try JSONEncoder.records.encode(record) + Data("\n".utf8)
        #expect(written == line)
        #expect(try JSONDecoder.snake.decode(FunctionRecord.self, from: written) == record)
        #expect(try fixture.names() == ["draft_tokens.bin", "function.json", "scores.bin"])
        #expect(!FunctionFixture.exists(fixture.output.appending(path: "failure.json")))
    }

    @Test("The record names the asset, the function and the load of the model, and has one time for each call")
    func recordsTheRun() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let load = FunctionLoad(
            requestedCompute: "gpu", expectFrequentReshapes: false, cached: true, cacheQueryError: "no entry",
            specializeS: 1.25, loadS: 2.5)
        let record = try await fixture.run(FakeFunctionSource(.init(mainHash: nil, load: load)), repeatCount: 4)
        #expect(record.mainHash == nil)
        #expect(record.function == "draft")
        #expect(record.requestedCompute == "gpu" && !record.expectFrequentReshapes)
        #expect(record.cached && record.cacheQueryError == "no entry")
        #expect(record.specializeS == 1.25 && record.loadS == 2.5)
        #expect(record.callS.count == 4 && record.callS.allSatisfy { $0 >= 0 && $0.isFinite })
    }

    @Test("The record carries the main hash that the asset reports")
    func recordsTheMainHash() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        #expect(try await fixture.run(FakeFunctionSource(.init(mainHash: "00ff"))).mainHash == "00ff")
    }

    @Test("A single call is the boundary of the repeat: one call is made and the outputs are identical")
    func runsOnce() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let source = FakeFunctionSource()
        let record = try await fixture.run(source, repeatCount: 1)
        #expect(source.calls == ["describe", "load draft", "call"])
        #expect(record.callS.count == 1)
        #expect(record.repeatIdentical && record.repeatDifferingCalls.isEmpty)
        #expect(!FunctionFixture.exists(fixture.output.appending(path: "failure.json")))
    }

    @Test(
        "Outputs that differ from the first call are named by their call number, the files hold the last call, and function.json and failure.json are both left",
        arguments: [
            ([1, 5, 1], [2], 1), ([1, 1, 5], [3], 5), ([1, 5, 9], [2, 3], 9), ([1, 5, 5], [2, 3], 5),
            ([5, 1, 1], [2, 3], 1),
        ] as [([Int32], [Int], Int32)])
    func reportsDifferingCalls(tokens: [Int32], differing: [Int], last: Int32) async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let source = FakeFunctionSource(.init(outputs: tokens.map { FunctionFixture.outputs(token: $0) }))
        let record = try await fixture.run(source, repeatCount: 3)
        #expect(source.calls == ["describe", "load draft", "call", "call", "call"])
        #expect(!record.repeatIdentical && record.repeatDifferingCalls == differing)
        #expect(
            try fixture.contents(of: "out/draft_tokens.bin") == FunctionFixture.integers([last, last + 1, last + 2]))
        let written = try JSONDecoder.snake.decode(FunctionRecord.self, from: fixture.contents(of: "out/function.json"))
        #expect(written == record)
        #expect(try fixture.failureKind() == "repeat_mismatch")
        #expect(try fixture.names() == ["draft_tokens.bin", "failure.json", "function.json", "scores.bin"])
    }

    @Test("A call whose output has the same bytes but another shape or dtype counts as different")
    func comparesTheShapeAndTheDtype() async throws {
        let base = FunctionFixture.outputs()
        let reshaped = [
            FunctionTensor(name: "draft_tokens", dtype: .int32, shape: [3, 1], data: base[0].data), base[1],
        ]
        let retyped = [
            FunctionTensor(name: "draft_tokens", dtype: .float32, shape: [1, 3], data: base[0].data), base[1],
        ]
        for other in [reshaped, retyped] {
            let fixture = try FunctionFixture()
            defer { fixture.remove() }
            let record = try await fixture.run(FakeFunctionSource(.init(outputs: [base, other])), repeatCount: 2)
            #expect(record.repeatDifferingCalls == [2])
        }
    }

    @Test("Calls with equal outputs leave no failure record")
    func leavesNoFailureRecordWhenIdentical() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let record = try await fixture.run(FakeFunctionSource(), repeatCount: 5)
        #expect(record.repeatIdentical && record.callS.count == 5)
        #expect(try fixture.names() == ["draft_tokens.bin", "function.json", "scores.bin"])
    }

    @Test("A load that fails leaves a failure record of the kind load, and no other file")
    func recordsALoadFailure() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let source = FakeFunctionSource(.init(failsAtLoad: true))
        await #expect(throws: FakeFunctionSource.Failure()) { try await fixture.run(source, repeatCount: 2) }
        #expect(source.calls == ["describe", "load draft"])
        #expect(try fixture.failureKind() == "load")
        #expect(try fixture.names() == ["failure.json"])
    }

    @Test(
        "A call that fails, on the first or a later repeat, leaves a failure record of the kind call, stops the repeats and writes no outputs",
        arguments: [1, 2, 3])
    func recordsACallFailure(at number: Int) async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let source = FakeFunctionSource(.init(failsAtCall: number))
        await #expect(throws: FakeFunctionSource.Failure()) { try await fixture.run(source, repeatCount: 3) }
        #expect(source.calls == ["describe", "load draft"] + Array(repeating: "call", count: number))
        #expect(try fixture.failureKind() == "call")
        #expect(try fixture.names() == ["failure.json"])
    }

    @Test("A cancellation during a call leaves a failure record of the kind cancelled and is thrown again")
    func recordsACancellation() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let source = FakeFunctionSource(.init(cancelsAtCall: 2))
        await #expect(throws: CancellationError.self) { try await fixture.run(source, repeatCount: 3) }
        #expect(source.calls == ["describe", "load draft", "call", "call"])
        #expect(try fixture.failureKind() == "cancelled")
        #expect(try fixture.names() == ["failure.json"])
    }

    @Test("A cancellation while the function is loaded leaves a failure record of the kind cancelled")
    func recordsACancellationAtLoad() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let source = FakeFunctionSource(.init(cancelsAtLoad: true))
        await #expect(throws: CancellationError.self) { try await fixture.run(source) }
        #expect(source.calls == ["describe", "load draft"])
        #expect(try fixture.failureKind() == "cancelled")
        #expect(try fixture.names() == ["failure.json"])
    }

    @Test("An output file that cannot be written leaves a failure record of the kind write and no function.json")
    func recordsAWriteFailure() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let blocked = fixture.output.appending(path: "draft_tokens.bin")
        let source = FakeFunctionSource(
            .init(beforeCall: { _ in
                try? FileManager.default.createDirectory(at: blocked, withIntermediateDirectories: false)
            }))
        await #expect(throws: (any Error).self) { try await fixture.run(source) }
        #expect(try fixture.failureKind() == "write")
        #expect(try fixture.names() == ["draft_tokens.bin", "failure.json"])
    }

    @Test(
        "An output name that cannot be a file name, native or bridged from Objective-C, is rejected after the call, before any output file is written, with a failure record of the kind write",
        arguments: [
            "a/b", "", "x\u{0}y", "/abs", "a/\u{0301}b", "/\u{0301}", "x\u{0}\u{0301}",
            String(repeating: "a", count: 252),
        ], [false, true])
    func rejectsAnUnusableOutputName(text: String, bridged: Bool) async throws {
        let name = bridged ? NSString(string: text) as String : text
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let valid = FunctionTensor(name: "ok", dtype: .int32, shape: [1], data: FunctionFixture.integers([1]))
        let output = FunctionTensor(name: name, dtype: .int32, shape: [1], data: FunctionFixture.integers([1]))
        let source = FakeFunctionSource(.init(outputs: [[valid, output]]))
        await #expect(throws: FunctionError.unusableOutputName(name)) { try await fixture.run(source) }
        #expect(source.calls == ["describe", "load draft", "call"])
        #expect(try fixture.failureKind() == "write")
        #expect(try fixture.names() == ["failure.json"])
    }

    @Test(
        "An output name with dots, spaces or a leading dot, and the longest name that makes a file name, is written as it is"
    )
    func writesUnusualButValidNames() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let names = ["a.b", "with space", ".hidden", "...", ".", "..", "function", String(repeating: "n", count: 251)]
        let outputs = names.map {
            FunctionTensor(name: $0, dtype: .int32, shape: [1], data: FunctionFixture.integers([3]))
        }
        let record = try await fixture.run(FakeFunctionSource(.init(outputs: [outputs])))
        #expect(record.outputs.compactMap(\.file) == names.map { "\($0).bin" })
        #expect(try fixture.names() == (names.map { "\($0).bin" } + ["function.json"]).sorted())
    }
}
