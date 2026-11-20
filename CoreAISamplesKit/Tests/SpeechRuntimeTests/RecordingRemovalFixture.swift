#if os(macOS)
    import Darwin
    import Foundation
    import SpeechCore
    import Synchronization

    @testable import SpeechRuntime

    struct RecordingRemovalExpectation: Error {
        let detail: String
    }

    func requireRemoval(_ value: Bool, _ message: String) throws {
        if !value { throw RecordingRemovalExpectation(detail: message) }
    }

    final class RecordingRemovalFixture: Sendable {
        struct DataBook {
            var reservations: [URL] = []
            var removals: [URL] = []
            var permissions = 0
            var stops = 0
            var removalFails = true
            var mode = "failure"
        }
        let book = Mutex(DataBook())
        let immutable: Bool
        init(immutable: Bool = false) { self.immutable = immutable }
        var state: DataBook { book.withLock { $0 } }
        func mode(_ mode: String) { book.withLock { $0.mode = mode } }
        func allowRemoval() { book.withLock { $0.removalFails = false } }
        func cleanFixture() {
            for url in state.reservations {
                if immutable { _ = chflags(url.path, 0) }
                try? FileManager.default.removeItem(at: url)
            }
        }
        func remove(_ url: URL) throws {
            let fails = book.withLock { s in
                s.removals.append(url)
                return s.removalFails
            }
            if fails { throw POSIXError(.EACCES) }
            try FileManager.default.removeItem(at: url)
        }
        func reserve() throws -> URL {
            let url = URL.temporaryDirectory.appending(path: "irodori-reference-" + UUID().uuidString + ".wav")
            try Data("synthetic test bytes".utf8).write(to: url, options: .withoutOverwriting)
            book.withLock { $0.reservations.append(url) }
            if immutable, chflags(url.path, UInt32(UF_IMMUTABLE)) != 0 {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            return url
        }
        var backend: MacSpeechRecordingBackend {
            MacSpeechRecordingBackend(
                requestPermission: { reply in
                    self.book.withLock { $0.permissions += 1 }
                    reply(true)
                },
                reserveFile: reserve, makeDevice: { _, _, answer in Device(fixture: self, answer: answer) })
        }
        private final class Device: MacSpeechRecordingDevice {
            let fixture: RecordingRemovalFixture
            let answer: @Sendable (Bool) -> Void
            init(fixture: RecordingRemovalFixture, answer: @escaping @Sendable (Bool) -> Void) {
                self.fixture = fixture
                self.answer = answer
            }
            func prepare() -> Bool { true }
            func start(duration: TimeInterval) -> Bool {
                switch fixture.state.mode {
                case "failure": answer(false)
                case "success": answer(true)
                default: break
                }
                return true
            }
            func stop() { fixture.book.withLock { $0.stops += 1 } }
        }
    }
#endif
