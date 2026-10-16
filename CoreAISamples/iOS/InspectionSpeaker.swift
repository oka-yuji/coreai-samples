#if os(iOS)
    import AVFoundation
    import Foundation
    import Synchronization
    import TruckInspectionCore
    import TruckInspectionRuntime

    private struct InspectionUtterance: Sendable {
        let id: ObjectIdentifier
        let continuation: AsyncStream<Bool>.Continuation
    }

    private final class InspectionUtteranceState: Sendable {
        let pending = Mutex<InspectionUtterance?>(nil)
    }

    @MainActor
    final class InspectionSpeaker: NSObject, AVSpeechSynthesizerDelegate {
        private let synthesizer = AVSpeechSynthesizer()
        private nonisolated let storage = InspectionUtteranceState()

        override init() {
            super.init()
            synthesizer.delegate = self
        }

        static var voice: AVSpeechSynthesisVoice? {
            AVSpeechSynthesisVoice.speechVoices().filter { $0.language == "ja-JP" }
                .sorted { $0.quality.rawValue > $1.quality.rawValue }.first
        }

        func speak(_ text: String) async throws {
            try Task.checkCancellation()
            guard let voice = Self.voice else { throw InspectionSpeechError.noJapaneseVoice }
            let utterance = AVSpeechUtterance(string: text)
            utterance.voice = voice
            let id = ObjectIdentifier(utterance)
            let pair = AsyncStream<Bool>.makeStream(bufferingPolicy: .bufferingNewest(1))
            let previous = storage.pending.withLock { value in
                let old = value
                value = InspectionUtterance(id: id, continuation: pair.continuation)
                return old
            }
            previous?.continuation.finish()
            let box = storage
            let watchdog = DispatchSource.makeTimerSource(queue: .global(qos: .userInitiated))
            watchdog.schedule(deadline: .now() + InspectionRules.speechDeadline(for: text))
            watchdog.setEventHandler {
                let completion = box.pending.withLock { value -> AsyncStream<Bool>.Continuation? in
                    guard value?.id == id else { return nil }
                    let result = value?.continuation
                    value = nil
                    return result
                }
                completion?.yield(false)
                completion?.finish()
            }
            watchdog.resume()
            synthesizer.speak(utterance)
            var finished = false
            for await result in pair.stream {
                finished = result
                break
            }
            watchdog.cancel()
            complete(id, finished: false)
            if !finished { synthesizer.stopSpeaking(at: .immediate) }
            try Task.checkCancellation()
            if !finished { throw InspectionSpeechError.playbackTimedOut }
        }

        func stop() {
            let id = storage.pending.withLock { $0?.id }
            if let id { complete(id, finished: false) }
            synthesizer.stopSpeaking(at: .immediate)
        }

        nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
            complete(ObjectIdentifier(utterance), finished: true)
        }

        nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
            complete(ObjectIdentifier(utterance), finished: false)
        }

        private nonisolated func complete(_ id: ObjectIdentifier, finished: Bool) {
            let continuation = storage.pending.withLock { value -> AsyncStream<Bool>.Continuation? in
                guard value?.id == id else { return nil }
                let continuation = value?.continuation
                value = nil
                return continuation
            }
            continuation?.yield(finished)
            continuation?.finish()
        }
    }
#endif
