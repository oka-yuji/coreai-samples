#if os(iOS)
    import AVFoundation
    import Foundation
    import Speech
    import Synchronization
    import TruckInspectionRuntime

    actor InspectionAudioCapture {
        private let engine = AVAudioEngine()
        private var request: InspectionRecognition?
        private var continuation: AsyncStream<InspectionSpeechEvent>.Continuation?
        private var timer: DispatchSourceTimer?
        private var tapInstalled = false
        private var routeID: String?
        private var sessionGeneration = 0
        private var activated = false

        func configure() async throws -> String {
            try Task.checkCancellation()
            sessionGeneration += 1
            let generation = sessionGeneration
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .default, options: [.allowBluetoothHFP, .duckOthers])
            activated = true
            try session.setActive(true)
            let bluetooth = session.availableInputs?.first {
                $0.portType == .bluetoothHFP || $0.portType == .bluetoothLE
            }
            var fallback = false
            if let bluetooth {
                do {
                    try session.setPreferredInput(bluetooth)
                    if try await waitForRoute(bluetooth.uid, generation: generation) {
                        routeID = bluetooth.uid
                    } else {
                        fallback = true
                    }
                } catch is CancellationError { throw CancellationError() } catch { fallback = true }
            }
            if bluetooth == nil || fallback {
                try Task.checkCancellation()
                guard sessionGeneration == generation else { throw CancellationError() }
                try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .duckOthers])
                guard let input = session.availableInputs?.first(where: { $0.portType == .builtInMic }) else {
                    throw InspectionSpeechError.routeUnavailable
                }
                try session.setPreferredInput(input)
                try session.setActive(true)
                guard try await waitForRoute(input.uid, generation: generation) else {
                    throw InspectionSpeechError.routeUnavailable
                }
                routeID = input.uid
            }
            return "route: \(routeDescription()) fallback=\(fallback)"
        }

        func start(hints: [String]) throws -> AsyncStream<InspectionSpeechEvent> {
            try Task.checkCancellation()
            guard continuation == nil, let routeID,
                AVAudioSession.sharedInstance().currentRoute.inputs.first?.uid == routeID
            else { throw InspectionSpeechError.routeUnavailable }
            let input = engine.inputNode
            let format = input.outputFormat(forBus: 0)
            guard format.sampleRate.isFinite, format.sampleRate > 0, format.channelCount > 0 else {
                throw InspectionSpeechError.invalidAudioFormat
            }
            let pair = AsyncStream<InspectionSpeechEvent>.makeStream(bufferingPolicy: .bufferingNewest(128))
            continuation = pair.continuation
            let sink = pair.continuation
            sink.yield(.route(routeDescription()))
            let started = ContinuousClock.now
            let box = InspectionRecognition()
            request = box
            let lastReport = Mutex<Double>(-1)
            do {
                try input.installAudioTap(
                    onBus: 0, bufferSize: AVAudioFrameCount(format.sampleRate * 0.1), format: format
                ) { @Sendable buffer, _ in
                    box.append(buffer)
                    let elapsed = (ContinuousClock.now - started) / .seconds(1)
                    let report = lastReport.withLock { previous in
                        guard elapsed - previous >= 0.5 else { return false }
                        previous = elapsed
                        return true
                    }
                    if report { sink.yield(.level(Self.level(AVAudioPCMBuffer(copying: buffer)), elapsed: elapsed)) }
                }
                tapInstalled = true
                try box.start(hints: hints, sink: sink, started: started)
                engine.prepare()
                try engine.start()
                let ticker = DispatchSource.makeTimerSource(queue: .global(qos: .userInitiated))
                ticker.schedule(deadline: .now() + .milliseconds(100), repeating: .milliseconds(100))
                ticker.setEventHandler { sink.yield(.tick((ContinuousClock.now - started) / .seconds(1))) }
                ticker.resume()
                timer = ticker
                return pair.stream
            } catch {
                finish()
                throw error
            }
        }

        func endAudio() {
            engine.stop()
            if tapInstalled {
                engine.inputNode.removeTap(onBus: 0)
                tapInstalled = false
            }
            request?.endAudio()
        }

        func finish() {
            timer?.cancel()
            timer = nil
            endAudio()
            continuation?.finish()
            continuation = nil
            request?.cancel()
            request = nil
            engine.reset()
        }

        func stopSession() throws {
            sessionGeneration += 1
            finish()
            routeID = nil
            guard activated else { return }
            try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
            activated = false
        }

        private func waitForRoute(_ uid: String, generation: Int) async throws -> Bool {
            for _ in 0..<15 {
                try Task.checkCancellation()
                guard sessionGeneration == generation else { throw CancellationError() }
                if AVAudioSession.sharedInstance().currentRoute.inputs.first?.uid == uid { return true }
                try await Task.sleep(for: .milliseconds(100))
            }
            try Task.checkCancellation()
            guard sessionGeneration == generation else { throw CancellationError() }
            return AVAudioSession.sharedInstance().currentRoute.inputs.first?.uid == uid
        }

        private func routeDescription() -> String {
            let session = AVAudioSession.sharedInstance()
            let input = session.currentRoute.inputs.map { "\($0.portType.rawValue)/\($0.portName)" }.joined(
                separator: ",")
            let output = session.currentRoute.outputs.map { "\($0.portType.rawValue)/\($0.portName)" }.joined(
                separator: ",")
            return "in=[\(input)] out=[\(output)] \(session.sampleRate)Hz"
        }

        private nonisolated static func level(_ buffer: AVAudioPCMBuffer) -> Double {
            let count = Int(buffer.frameLength)
            guard count > 0 else { return -120 }
            var sum = 0.0
            for index in 0..<count {
                let offset = index * buffer.stride
                let value: Double
                if let data = buffer.floatChannelData {
                    value = Double(data[0][offset])
                } else if let data = buffer.int16ChannelData {
                    value = Double(data[0][offset]) / 32768
                } else if let data = buffer.int32ChannelData {
                    value = Double(data[0][offset]) / 2_147_483_648
                } else {
                    return -120
                }
                sum += value * value
            }
            let rms = (sum / Double(count)).squareRoot()
            return rms > 0 ? 20 * log10(rms) : -120
        }
    }
#endif
