#if os(macOS)
    import Darwin
    import Foundation
    import SpeechCore

    @testable import SpeechRuntime

    enum RecordingRemovalScenarios {
        static func failed(_ optional: MacSpeechReferenceRecorder?, _ ticket: Int) async throws {
            guard let recorder = optional else {
                throw RecordingRemovalExpectation(detail: "test recorder was released early")
            }
            do {
                _ = try await recorder.record(ticket: ticket) {}
                throw RecordingRemovalExpectation(detail: "a failed recording returned a URL")
            } catch SpeechReferenceError.recordingFailed {}
        }
        static func ownership() async throws {
            let f = RecordingRemovalFixture()
            defer { f.cleanFixture() }
            let cleanup = MacSpeechRecordingCleanup(remove: f.remove)
            var previous: MacSpeechReferenceRecorder? = MacSpeechReferenceRecorder(backend: f.backend, cleanup: cleanup)
            try await failed(previous, 1)
            guard let url = f.state.reservations.first else {
                throw RecordingRemovalExpectation(detail: "no actual reservation")
            }
            try requireRemoval(
                f.state.stops == 1 && f.state.removals == [url] && FileManager.default.fileExists(atPath: url.path),
                "first failure retained the file after one stopped device")
            await previous?.stop()
            try requireRemoval(f.state.removals == [url, url], "idle stop must retry the same owned path")
            try await failed(previous, 2)
            try requireRemoval(
                f.state.permissions == 1 && f.state.reservations == [url] && f.state.removals == [url, url, url],
                "no permission or new file while cleanup fails")
            weak let expired = previous
            previous = nil
            try requireRemoval(expired == nil, "cleanup owner must not keep the old recorder alive")
            let next = MacSpeechReferenceRecorder(backend: f.backend, cleanup: cleanup)
            f.allowRemoval()
            await next.stop()
            try requireRemoval(
                !FileManager.default.fileExists(atPath: url.path) && f.state.removals == [url, url, url, url],
                "a later owner retires the exact pending file")
            f.mode("success")
            let handed = try await next.record(ticket: 3) {}
            try requireRemoval(
                handed != url && FileManager.default.fileExists(atPath: handed.path),
                "recovery transfers the new complete file")
            await next.stop()
            try requireRemoval(
                f.state.removals.allSatisfy { $0 == url } && FileManager.default.fileExists(atPath: handed.path),
                "a successful handoff is no longer owned by the recorder")
        }
        static func cancelled() async throws {
            let f = RecordingRemovalFixture()
            f.mode("waiting")
            defer { f.cleanFixture() }
            let cleanup = MacSpeechRecordingCleanup(remove: f.remove)
            let recorder = MacSpeechReferenceRecorder(backend: f.backend, cleanup: cleanup)
            let (starts, sink) = AsyncStream<Void>.makeStream()
            async let captured: URL = recorder.record(ticket: 5) {
                sink.yield(())
                sink.finish()
            }
            var events = starts.makeAsyncIterator()
            guard await events.next() != nil else { throw RecordingRemovalExpectation(detail: "device did not start") }
            await recorder.stop()
            do {
                _ = try await captured
                throw RecordingRemovalExpectation(detail: "cancel lost precedence")
            } catch is CancellationError {}
            let first = f.state
            try requireRemoval(
                first.permissions == 1 && first.stops == 1 && first.removals == first.reservations
                    && first.removals.count == 1, "nonthrow stop must attempt cleanup after stopping capture")
            guard let url = first.reservations.first else {
                throw RecordingRemovalExpectation(detail: "no reservation")
            }
            try requireRemoval(
                FileManager.default.fileExists(atPath: url.path), "injected removal failure was not exercised")
            let before = f.state.removals
            f.allowRemoval()
            f.mode("success")
            let separate = try await recorder.record(ticket: 6) {}
            try requireRemoval(
                !FileManager.default.fileExists(atPath: url.path) && f.state.removals == before + [url]
                    && f.state.permissions == 2 && separate != url, "retry at next record restores progress")
        }
        static func alreadyGone() async throws {
            let f = RecordingRemovalFixture()
            defer { f.cleanFixture() }
            let cleanup = MacSpeechRecordingCleanup(remove: f.remove)
            let rec = MacSpeechReferenceRecorder(backend: f.backend, cleanup: cleanup)
            try await failed(rec, 7)
            guard let pending = f.state.reservations.first else {
                throw RecordingRemovalExpectation(detail: "missing failed file")
            }
            try FileManager.default.removeItem(at: pending)
            f.allowRemoval()
            f.mode("success")
            let fresh = try await rec.record(ticket: 8) {}
            try requireRemoval(
                fresh != pending && f.state.removals == [pending, pending] && f.state.permissions == 2,
                "confirmed missing file must not block a new capture")
        }
    }
#endif
