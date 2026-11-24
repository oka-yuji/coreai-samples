import Darwin
import Foundation
import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct FunctionRunFailureRecordTests {
    @Test(
        "An output folder that cannot hold the first failure record is removed with the reason, and the function is not loaded"
    )
    func removesAnOutputThatCannotHoldTheFirstRecord() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let source = FakeFunctionSource()
        let error = await #expect(throws: FunctionError.self) { try await fixture.run(source, folderMode: 0) }
        guard case .cannot(let action, let reason) = try #require(error) else {
            Issue.record("The error is not a refusal to write: \(String(describing: error))")
            return
        }
        #expect(action == "write the failure record")
        #expect(!reason.contains("could not be removed"))
        #expect(source.calls == ["describe"])
        #expect(!FunctionFixture.exists(fixture.output))
    }

    @Test(
        "When nothing more can be written into the output folder, the failure record of the kind incomplete stays and there is no function.json"
    )
    func keepsTheIncompleteRecord() async throws {
        let fixture = try FunctionFixture()
        let output = fixture.output.path(percentEncoded: false)
        defer {
            chmod(output, 0o755)
            fixture.remove()
        }
        let source = FakeFunctionSource(.init(beforeCall: { _ in chmod(output, 0o555) }))
        await #expect(throws: (any Error).self) { try await fixture.run(source) }
        #expect(try fixture.failureKind() == "incomplete")
        #expect(try fixture.names() == ["failure.json"])
    }

    @Test(
        "A failure record that cannot be removed after the record is written makes the run fail, and both records stay"
    )
    func failsWhenTheFailureRecordCannotBeRemoved() async throws {
        let fixture = try FunctionFixture()
        let record = fixture.output.appending(path: "failure.json").path(percentEncoded: false)
        defer {
            chflags(record, 0)
            fixture.remove()
        }
        let source = FakeFunctionSource(.init(beforeCall: { _ in chflags(record, UInt32(UF_IMMUTABLE)) }))
        await #expect(throws: (any Error).self) { try await fixture.run(source) }
        #expect(try fixture.names().contains("function.json"))
        #expect(try fixture.failureKind() == "incomplete")
    }

    @Test("A run that succeeds leaves no failure record, and a repeat mismatch replaces the first record")
    func replacesOrRemovesTheFirstRecord() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        _ = try await fixture.run(FakeFunctionSource(), repeatCount: 2)
        #expect(try fixture.names() == ["draft_tokens.bin", "function.json", "scores.bin"])
        let other = try FunctionFixture()
        defer { other.remove() }
        let outputs = [FunctionFixture.outputs(token: 1), FunctionFixture.outputs(token: 2)]
        let record = try await other.run(FakeFunctionSource(.init(outputs: outputs)), repeatCount: 2)
        #expect(!record.repeatIdentical)
        #expect(try other.failureKind() == "repeat_mismatch")
    }
}
