import Foundation
import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct FunctionRunCancellationTests {
    private func run(
        _ fixture: FunctionFixture, _ source: FakeFunctionSource, repeatCount: Int = 1, cancelFirst: Bool = false
    )
        async -> (any Error)?
    {
        let task = Task {
            if cancelFirst { _ = withUnsafeCurrentTask { $0?.cancel() } }
            return try await fixture.run(source, repeatCount: repeatCount)
        }
        do {
            _ = try await task.value
            return nil
        } catch {
            return error
        }
    }

    @Test("A task that is cancelled when the output is made stops before the function is loaded")
    func stopsBeforeTheLoad() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let source = FakeFunctionSource()
        #expect(await run(fixture, source, cancelFirst: true) is CancellationError)
        #expect(source.calls == ["describe"])
        #expect(try fixture.failureKind() == "cancelled")
        #expect(try fixture.names() == ["failure.json"])
    }

    @Test("A load that cancels the task and returns a value is followed by no call, no output and no function.json")
    func stopsAfterALoadThatCancelsTheTask() async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let source = FakeFunctionSource(.init(cancelsTaskAtLoad: true))
        #expect(await run(fixture, source, repeatCount: 3) is CancellationError)
        #expect(source.calls == ["describe", "load draft"])
        #expect(try fixture.failureKind() == "cancelled")
        #expect(try fixture.names() == ["failure.json"])
    }

    @Test(
        "A call that cancels the task and returns a value is followed by no other call, no output and no function.json, including when it was the last call",
        arguments: [(3, 1), (3, 2), (1, 1), (2, 2)])
    func stopsAfterACallThatCancelsTheTask(repeatCount: Int, cancelAt: Int) async throws {
        let fixture = try FunctionFixture()
        defer { fixture.remove() }
        let source = FakeFunctionSource(.init(cancelsTaskAtCall: cancelAt))
        #expect(await run(fixture, source, repeatCount: repeatCount) is CancellationError)
        #expect(source.calls == ["describe", "load draft"] + Array(repeating: "call", count: cancelAt))
        #expect(try fixture.failureKind() == "cancelled")
        #expect(try fixture.names() == ["failure.json"])
    }
}
