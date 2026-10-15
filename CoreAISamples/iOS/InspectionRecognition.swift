#if os(iOS)
    import AVFoundation
    import Foundation
    import Speech
    import Synchronization
    import TruckInspectionRuntime

    final class InspectionRecognition: Sendable {
        private struct State {
            var recognizer: SFSpeechRecognizer?
            var request: SFSpeechAudioBufferRecognitionRequest?
            var task: SFSpeechRecognitionTask?
        }
        private let storage = Mutex(State())

        func start(
            hints: [String], sink: AsyncStream<InspectionSpeechEvent>.Continuation, started: ContinuousClock.Instant
        ) throws {
            try storage.withLock { state in
                let recognizer = try Self.makeRecognizer()
                let request = SFSpeechAudioBufferRecognitionRequest()
                request.requiresOnDeviceRecognition = true
                request.shouldReportPartialResults = true
                request.taskHint = .dictation
                request.contextualStrings = Array(Set(hints)).sorted()
                state.recognizer = recognizer
                state.request = request
                state.task = recognizer.recognitionTask(with: request) { @Sendable result, error in
                    if let result {
                        let text = result.bestTranscription.formattedString
                        if result.isFinal {
                            sink.yield(.final(text))
                            sink.finish()
                            return
                        }
                        if !text.isEmpty {
                            sink.yield(.partial(text, elapsed: (ContinuousClock.now - started) / .seconds(1)))
                        }
                    }
                    if let error {
                        sink.yield(.failure(String(describing: error)))
                        sink.finish()
                    }
                }
            }
        }

        static func makeRecognizer() throws -> SFSpeechRecognizer {
            guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "ja-JP")),
                recognizer.locale.identifier.replacingOccurrences(of: "_", with: "-").lowercased() == "ja-jp",
                recognizer.supportsOnDeviceRecognition, recognizer.isAvailable
            else { throw InspectionSpeechError.unavailable }
            return recognizer
        }

        func append(_ buffer: AVReadOnlyAudioPCMBuffer) {
            storage.withLock { state in
                guard let request = state.request else { return }
                request.append(AVAudioPCMBuffer(copying: buffer))
            }
        }

        func endAudio() {
            storage.withLock { state in
                state.request?.endAudio()
                state.request = nil
            }
        }

        func cancel() {
            storage.withLock { state in
                state.task?.cancel()
                state.task = nil
                state.request = nil
                state.recognizer = nil
            }
        }
    }
#endif
