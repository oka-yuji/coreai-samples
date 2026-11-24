import ArgumentParser
import Foundation
import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct RunFunctionTests {
    private func arguments(_ fixture: FunctionFixture, repeatOption: String? = nil) -> [String] {
        let base = [
            "--model", "/not-loaded", "--function", "draft", "--inputs", fixture.list.path(percentEncoded: false),
            "--output", fixture.output.path(percentEncoded: false),
        ]
        return base + (repeatOption.map { [$0] } ?? [])
    }

    @Test("The options of the command are read, and the repeat count is 1 when it is not given")
    func readsOptions() throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let command = try RunFunction.parse(arguments(fixture))
        #expect(command.model == "/not-loaded" && command.function == "draft")
        #expect(command.inputs == fixture.list.path(percentEncoded: false))
        #expect(command.output == fixture.output.path(percentEncoded: false))
        #expect(command.repeatCount == 1)
        #expect(try RunFunction.parse(arguments(fixture, repeatOption: "--repeat=3")).repeatCount == 3)
    }

    @Test(
        "A repeat count of zero or less is rejected and one or more is accepted",
        arguments: [("--repeat=0", false), ("--repeat=-1", false), ("--repeat=1", true), ("--repeat=2", true)])
    func checksTheRepeatCount(option: String, accepted: Bool) throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        if accepted {
            _ = try RunFunction.parse(arguments(fixture, repeatOption: option))
        } else {
            let error = #expect(throws: (any Error).self) {
                try RunFunction.parse(arguments(fixture, repeatOption: option))
            }
            #expect(RunFunction.message(for: try #require(error)).hasPrefix("--repeat takes a count of one or more."))
        }
    }

    @Test("A run whose repeated calls agree leaves its record and returns")
    func succeeds() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        try await RunFunction.parse(arguments(fixture, repeatOption: "--repeat=2")).execute(
            source: FakeFunctionSource())
        #expect(FileManager.default.fileExists(atPath: fixture.output.appending(path: "function.json").path))
    }

    @Test(
        "A run whose repeated calls differ ends with the exit code 1 after it has left its record and its failure record"
    )
    func failsWhenTheCallsDiffer() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let source = FakeFunctionSource(
            .init(outputs: [FunctionFixture.outputs(token: 1), FunctionFixture.outputs(token: 2)]))
        let command = try RunFunction.parse(arguments(fixture, repeatOption: "--repeat=2"))
        await #expect(throws: ExitCode.failure) { try await command.execute(source: source) }
        for name in ["function.json", "failure.json"] {
            #expect(FileManager.default.fileExists(atPath: fixture.output.appending(path: name).path))
        }
    }

    @Test("A rejected input is thrown as a validation error that carries the message of the check")
    func throwsAValidationError() async throws {
        let fixture = try FunctionFixture(listText: "[]")
        defer { fixture.remove() }
        let command = try RunFunction.parse(arguments(fixture))
        let error = await #expect(throws: ValidationError.self) {
            try await command.execute(source: FakeFunctionSource())
        }
        #expect(error?.message == FunctionError.invalidList("the top level is not a JSON object").description)
    }

    @Test("A failure after the output was made is thrown as it is")
    func throwsAFailureAsItIs() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let command = try RunFunction.parse(arguments(fixture))
        await #expect(throws: FakeFunctionSource.Failure()) {
            try await command.execute(source: FakeFunctionSource(.init(failsAtCall: 1)))
        }
    }
}
