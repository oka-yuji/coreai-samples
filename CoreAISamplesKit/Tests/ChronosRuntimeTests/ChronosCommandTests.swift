import ChronosCore
import Foundation
import Testing

@testable import ChronosRuntime

@Suite(.timeLimit(.minutes(1)))
struct ChronosCommandTests {
    @Test("The command accepts exactly one explicit input mode and help")
    func parsing() throws {
        #expect(try ChronosCommand.parse(["--help"]) == .help)
        let bundle = URL(fileURLWithPath: "bundle")
        let input = URL(fileURLWithPath: "input.json")
        let output = URL(fileURLWithPath: "output.json")
        #expect(
            try ChronosCommand.parse(["--bundle", "bundle", "--input", "input.json", "--output", "output.json"])
                == .run(bundle: bundle, mode: .input(input), output: output))
        #expect(
            try ChronosCommand.parse(["--output", "output.json", "--verify", "--bundle", "bundle"])
                == .run(bundle: bundle, mode: .verify, output: output))
        for arguments in [
            [], ["--bundle"], ["--help", "--verify"], ["--unknown"],
            ["--bundle", "b", "--output", "o"],
            ["--bundle", "b", "--verify", "--input", "i", "--output", "o"],
            ["--bundle", "b", "--verify", "--verify", "--output", "o"],
            ["--bundle", "b", "--bundle", "b", "--verify", "--output", "o"],
            ["--bundle", "", "--verify", "--output", "o"],
        ] {
            #expect(throws: ChronosError.invalidArguments) { try ChronosCommand.parse(arguments) }
        }
    }

    @Test("JSON null maps explicitly to missing while numeric values remain unscaled")
    func input() throws {
        let data = Data(#"{"context":[null,-12.5,0,1000,null]}"#.utf8)
        let input = try ChronosCommand.decodeInput(data)
        #expect(input.rawCount == 5)
        #expect(input.paddedValues[11].isNaN)
        #expect(Array(input.paddedValues[12...14]) == [-12.5, 0, 1000])
        #expect(input.paddedValues[15].isNaN)
    }

    @Test("Malformed boolean string nonfinite and oversized inputs are rejected")
    func invalidInput() {
        for json in [
            #"{"context":[true]}"#, #"{"context":["NaN",1]}"#, #"{"context":[null]}"#,
            #"{"context":[1e100]}"#, #"{"context":[]}"#, #"{"context":[[1]]}"#,
            #"{"context":[1],"extra":1}"#, #"[1]"#, #"{"context":[NaN,1]}"#,
        ] {
            #expect(throws: (any Error).self) { try ChronosCommand.decodeInput(Data(json.utf8)) }
        }
        let values = Array(repeating: 1, count: 8193)
        #expect(throws: (any Error).self) {
            try ChronosCommand.decodeInput(JSONSerialization.data(withJSONObject: ["context": values]))
        }
        #expect(throws: (any Error).self) {
            try ChronosCommand.decodeInput(Data(repeating: 32, count: ChronosFiles.inputLimit + 1))
        }
    }

    @Test("Reports are complete JSON and refuse existing files directories and dangling links")
    func save() throws {
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        let output = folder.url.appendingPathComponent("result.json")
        try ChronosReportFile.save(["value": [1, 2, 3]], to: output)
        let saved = try Data(contentsOf: output)
        #expect(try JSONDecoder().decode([String: [Int]].self, from: saved) == ["value": [1, 2, 3]])
        #expect(throws: ChronosError.outputExists) { try ChronosReportFile.save(["value": 4], to: output) }
        #expect(try Data(contentsOf: output) == saved)
        #expect(throws: ChronosError.outputExists) { try ChronosReportFile.save(["value": 4], to: folder.url) }
        let link = folder.url.appendingPathComponent("link.json")
        try FileManager.default.createSymbolicLink(
            at: link, withDestinationURL: folder.url.appendingPathComponent("absent"))
        #expect(throws: ChronosError.outputExists) { try ChronosReportFile.save(["value": 4], to: link) }
        #expect(
            try FileManager.default.contentsOfDirectory(atPath: folder.url.path).sorted() == [
                "link.json", "result.json",
            ])
    }

    @Test("Encoding and output path failures leave no partial report")
    func saveFailures() throws {
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        let output = folder.url.appendingPathComponent("result.json")
        #expect(throws: (any Error).self) { try ChronosReportFile.save(["value": Float.nan], to: output) }
        #expect(!FileManager.default.fileExists(atPath: output.path))
        #expect(throws: (any Error).self) {
            try ChronosReportFile.save(["value": 1], to: folder.url.appendingPathComponent("missing/result.json"))
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: folder.url.path).isEmpty)
    }

    @Test("Competing report writers publish exactly one complete file")
    func competingWriters() async throws {
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        let output = folder.url.appendingPathComponent("result.json")
        let winners = await withTaskGroup(of: Bool.self) { group in
            for value in 0..<8 {
                group.addTask {
                    do {
                        try ChronosReportFile.save(["value": value], to: output)
                        return true
                    } catch {
                        #expect(error as? ChronosError == .outputExists)
                        return false
                    }
                }
            }
            var count = 0
            for await won in group { if won { count += 1 } }
            return count
        }
        #expect(winners == 1)
        let value = try JSONDecoder().decode([String: Int].self, from: Data(contentsOf: output))
        #expect((0..<8).contains(value["value"] ?? -1))
        #expect(try FileManager.default.contentsOfDirectory(atPath: folder.url.path) == ["result.json"])
    }

    @Test("Help and invalid input finish without loading a model or saving output")
    func earlyFailures() async throws {
        try await ChronosCommand.execute(.help)
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        let input = folder.url.appendingPathComponent("input.json")
        try Data(#"{"context":[null]}"#.utf8).write(to: input)
        let output = folder.url.appendingPathComponent("output.json")
        await #expect(throws: ChronosError.invalidContext) {
            try await ChronosCommand.execute(.run(bundle: folder.url, mode: .input(input), output: output))
        }
        #expect(!FileManager.default.fileExists(atPath: output.path))
        try Data([7]).write(to: output)
        await #expect(throws: ChronosError.outputExists) {
            try await ChronosCommand.execute(.run(bundle: folder.url, mode: .verify, output: output))
        }
        #expect(try Data(contentsOf: output) == Data([7]))
    }

    @Test("The input command saves all predictions and releases before returning")
    func completeInput() async throws {
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        let input = folder.url.appendingPathComponent("input.json")
        try Data(#"{"context":[null,1]}"#.utf8).write(to: input)
        let output = folder.url.appendingPathComponent("output.json")
        let runtime = ChronosRuntime(factory: { ChronosFakePredictor(identity: ChronosTestFolder.identity) })
        try await ChronosCommand.execute(
            .run(bundle: folder.url, mode: .input(input), output: output), runtimeFactory: { _ in runtime })
        let saved = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: output)) as? [String: Any])
        #expect(saved["shape"] as? [Int] == [1, 21, 16])
        let quantiles = try #require(saved["quantiles"] as? [[[Float]]])
        #expect(quantiles == [ChronosTestFolder.rows()])
        #expect(saved["raw_count"] as? Int == 2)
        #expect(saved["loaded"] as? Bool == true)
        #expect(try await runtime.predict(ChronosContext([0])).loaded)
        await runtime.release()
    }

    @Test("A report created during inference is preserved and saving failure still joins release")
    func lateSaveFailure() async throws {
        let folder = try ChronosTestFolder()
        defer { folder.clean() }
        let input = folder.url.appendingPathComponent("input.json")
        try Data(#"{"context":[1]}"#.utf8).write(to: input)
        let output = folder.url.appendingPathComponent("output.json")
        let gate = ChronosGate()
        defer { gate.finish(.failure(CancellationError())) }
        let events = AsyncStream<Bool>.makeStream()
        defer { events.continuation.finish() }
        var iterator = events.stream.makeAsyncIterator()
        let runtime = ChronosRuntime(factory: {
            ChronosFakePredictor(
                identity: ChronosTestFolder.identity,
                run: { _ in
                    events.continuation.yield(true)
                    try await gate.wait()
                    return try ChronosPrediction(quantiles: ChronosTestFolder.rows())
                })
        })
        let operation = Task {
            try await ChronosCommand.execute(
                .run(bundle: folder.url, mode: .input(input), output: output), runtimeFactory: { _ in runtime })
        }
        #expect(await iterator.next() == true)
        do {
            try Data([8]).write(to: output)
        } catch {
            gate.finish(.failure(error))
            _ = await operation.result
            throw error
        }
        gate.finish()
        switch await operation.result {
        case .success: Issue.record("Saving over the competing file must fail")
        case .failure(let error): #expect(error as? ChronosError == .outputExists)
        }
        #expect(try Data(contentsOf: output) == Data([8]))
        #expect(try await runtime.predict(ChronosContext([0])).loaded)
        await runtime.release()
    }
}
