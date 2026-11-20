#if os(macOS)
    import Foundation
    import SpeechCore
    import Synchronization

    @testable import SpeechRuntime

    final class VoxCPMWorkflowCapture: Sendable {
        enum Fault: Error, Equatable, Sendable { case none, failureAfterEncoder, cancelAfterEncoder }

        struct Stop: Sendable {
            let serial: Int
            let patches: Int
            let stoppedPatch: Int
            let samples: Int
            let rate: Int
        }

        private struct State {
            var created = 0
            var released = [Int]()
            var stops = [Stop]()
            var encoderCalls = 0
            var armed = Fault.none
            var injections = [Fault]()
        }

        private let state = Mutex(State())
        var created: Int { state.withLock { $0.created } }
        var released: [Int] { state.withLock { $0.released } }
        var stops: [Stop] { state.withLock { $0.stops } }
        var encoderCalls: Int { state.withLock { $0.encoderCalls } }
        var injections: [Fault] { state.withLock { $0.injections } }
        func arm(_ fault: Fault) { state.withLock { $0.armed = fault } }

        private func allocate() -> Int {
            state.withLock {
                $0.created += 1
                return $0.created
            }
        }

        private func finish(_ serial: Int, _ audio: VoxCPMSpeech) {
            let row = Stop(
                serial: serial, patches: audio.patchCount, stoppedPatch: audio.stoppedPatch,
                samples: audio.samples.count, rate: audio.sampleRate)
            state.withLock { $0.stops.append(row) }
        }

        private func afterEncoder() throws {
            let fault = state.withLock { value in
                value.encoderCalls += 1
                let next = value.armed
                value.armed = .none
                if next != .none { value.injections.append(next) }
                return next
            }
            switch fault {
            case .none: break
            case .failureAfterEncoder: throw fault
            case .cancelAfterEncoder: withUnsafeCurrentTask { $0?.cancel() }
            }
        }

        func dependencies() -> VoxCPMDependencies {
            var result = VoxCPMDependencies.live
            result.pipeline = { [self] opened in
                let loaded = try await VoxCPMPipeline<VoxCPMAssets>.load(opened)
                let parts = Parts(assets: loaded.components, capture: self)
                let wrapped = VoxCPMPipeline(components: parts, contract: loaded.contract)
                return Observed(pipeline: wrapped, capture: self, serial: allocate())
            }
            return result
        }

        private struct Parts: VoxCPMComponents {
            let assets: VoxCPMAssets
            let capture: VoxCPMWorkflowCapture

            func makeStates() throws -> VoxCPMAssetStates { try assets.makeStates() }

            @concurrent
            func call(_ function: VoxCPMFunction, _ inputs: [String: VoxCPMInput], states: inout VoxCPMAssetStates)
                async throws -> [String: [Float]]
            {
                let outputs = try await assets.call(function, inputs, states: &states)
                if function.rawValue == "refenc.main" { try capture.afterEncoder() }
                return outputs
            }
        }

        private final class Observed: VoxCPMSpeechPipeline, Sendable {
            let pipeline: VoxCPMPipeline<Parts>
            let capture: VoxCPMWorkflowCapture
            let serial: Int

            init(pipeline: VoxCPMPipeline<Parts>, capture: VoxCPMWorkflowCapture, serial: Int) {
                self.pipeline = pipeline
                self.capture = capture
                self.serial = serial
            }

            deinit { capture.state.withLock { $0.released.append(serial) } }

            @concurrent
            func generate(ids: [Int32], seed: UInt64) async throws -> VoxCPMSpeech {
                try await generate(ids: ids, seed: seed, reference: nil)
            }

            @concurrent
            func generate(ids: [Int32], seed: UInt64, reference: SpeechReferenceAudio?) async throws -> VoxCPMSpeech {
                let production: any VoxCPMSpeechPipeline = pipeline
                let speech = try await production.generate(ids: ids, seed: seed, reference: reference)
                capture.finish(serial, speech)
                return speech
            }
        }
    }
#endif
