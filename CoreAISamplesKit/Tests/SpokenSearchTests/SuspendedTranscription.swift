import Foundation
import SpokenSearchFeature

@testable import SpokenSearchRuntime

@MainActor
final class SuspendedTranscription: SpokenTranscribing {
    private(set) var starts = 0
    private var canceled = false
    private var waitStart: CheckedContinuation<Void, Never>?
    private var waitCancel: CheckedContinuation<Void, Never>?
    private var finishing: CheckedContinuation<Void, Never>?

    func transcribe(_ url: URL) async throws -> SpokenTranscript {
        starts += 1
        waitStart?.resume()
        waitStart = nil
        await withCheckedContinuation { finishing = $0 }
        return SpokenTranscript(runs: [SpokenRun(text: "day", start: 0, end: 1)])
    }

    func cancel() async {
        guard finishing != nil else { return }
        canceled = true
        waitCancel?.resume()
        waitCancel = nil
    }

    func waitForStart() async {
        if starts == 0 { await withCheckedContinuation { waitStart = $0 } }
    }

    func waitForCancel() async {
        if !canceled { await withCheckedContinuation { waitCancel = $0 } }
    }

    func finish() {
        finishing?.resume()
        finishing = nil
    }
}
