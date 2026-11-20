import Foundation
import SpokenSearchFeature
import Testing

@testable import SpokenSearchRuntime

@MainActor
struct SpokenRunnerTests {
    @Test("Closing a session joins an uncooperative transcription and revokes a later open")
    func closingJoinsWork() async {
        let slow = SuspendedTranscription()
        let runner = SpokenSearchRunner(isPermitted: { true }, speech: slow)
        var received = 0
        let open = Task {
            await runner.run(.open(1, URL(filePath: "/one.wav"))) { _ in received += 1 }
        }
        await slow.waitForStart()
        var didClose = false
        let close = Task {
            await runner.close()
            didClose = true
        }
        await slow.waitForCancel()
        #expect(!didClose)
        slow.finish()
        await open.value
        await close.value
        #expect(received == 0 && didClose)
        await runner.run(.open(2, URL(filePath: "/two.wav"))) { _ in received += 1 }
        #expect(slow.starts == 1 && received == 0)
    }
}
