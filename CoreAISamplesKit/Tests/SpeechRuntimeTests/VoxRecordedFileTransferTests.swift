#if os(macOS)
    import Foundation
    import SpeechCore
    import Synchronization
    import Testing

    @testable import SpeechRuntime

    struct VoxRecordedFileTransferTests {
        @Test(
            "VRC-04 A prepared Vox reference stays valid while another recorder retries the transferred file's deletion"
        )
        func sharesRetriedFile() async throws {
            let url = URL.temporaryDirectory.appending(path: "irodori-reference-" + UUID().uuidString + ".wav")
            defer { try? FileManager.default.removeItem(at: url) }
            let original = try SpeechReferenceAudio(
                samples: Array(repeating: 0.25, count: 80_000), source: .recording, format: .voxcpm2)
            let wav = try await referenceWAV(original)
            try wav.write(to: url, options: .withoutOverwriting)
            let denied = Mutex(true)
            let attempts = Mutex<[URL]>([])
            let cleanup = MacSpeechRecordingCleanup(remove: { path in
                attempts.withLock { $0.append(path) }
                if denied.withLock({ $0 }) { throw POSIXError(.EACCES) }
                try FileManager.default.removeItem(at: path)
            })
            func prepare() async throws -> SpeechReferenceAudio {
                let reader = SpeechReferenceReader(format: .voxcpm2, cleanup: cleanup)
                let value = try await reader.read(url, source: .recording)
                await reader.removeRecording(url)
                return value
            }
            let prepared = try await prepare()
            #expect(prepared == original && FileManager.default.fileExists(atPath: url.path))
            #expect(attempts.withLock { $0 } == [url])
            denied.withLock { $0 = false }
            let backend = MacSpeechRecordingBackend(
                requestPermission: { _ in Issue.record("Stopping an idle recorder requested hardware.") },
                reserveFile: { throw SpeechReferenceError.recordingFailed },
                makeDevice: { _, _, _ in throw SpeechReferenceError.recordingFailed })
            await MacSpeechReferenceRecorder(backend: backend, cleanup: cleanup).stop()
            #expect(attempts.withLock { $0 } == [url, url])
            #expect(!FileManager.default.fileExists(atPath: url.path) && prepared.samples == original.samples)
        }
    }
#endif
